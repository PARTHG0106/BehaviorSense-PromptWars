"""Experimental joint/bone fusion with ordered, masked detector observations.

This model has its own artifact format. It neither changes the baseline model
nor imports its validation-selected weights. Symmetric context is an offline
prediction protocol with lookahead, recorded by the training manifest.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from behaviorsense.models.stgcnpp import STGCNpp, make_stream
from behaviorsense.models.visual_context import CONTEXT_DIM, CONTEXT_SCHEMA


TEMPORAL_FORMAT = "behaviorsense_multilabel_temporal_v1"


class _MaskedTemporalBlock(nn.Module):
    """A temporal residual block that cannot propagate padded token contents."""

    def __init__(self, hidden: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        # Mask after normalization too: its learned bias must not turn a gap into
        # an input for the next temporal convolution.
        normalized = self.norm(values).masked_fill(~observed, 0)
        update = self.conv(normalized.transpose(1, 2)).transpose(1, 2)
        return (values + self.drop(self.activation(update))).masked_fill(~observed, 0)


class TemporalMultilabelFusion(nn.Module):
    """Two independent skeleton backbones plus an ordered context residual.

    Inputs are normalized joint coordinates ``[N,3,30,17,2]`` and detector
    observations ``[N,2*context_radius+1,401]``. Bone conversion happens here,
    exactly once. Context feature zero is the observed indicator. Padded values
    are discarded before encoding and after every temporal residual block.

    Skeleton logits average the two learned classifiers. The final context
    layer starts at zero, making the initial model exactly the skeleton model;
    an all-missing context always retains that fallback after training too.
    """

    def __init__(self, n_classes=22, base_channels=64, context_hidden=128,
                 context_radius=6, dropout=.2):
        super().__init__()
        for name, value, minimum in (("n_classes", n_classes, 1),
                                     ("base_channels", base_channels, 4),
                                     ("context_hidden", context_hidden, 1),
                                     ("context_radius", context_radius, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        self.context_radius = context_radius
        self.context_length = 2 * context_radius + 1
        self.context_dim = CONTEXT_DIM
        self.context_schema = CONTEXT_SCHEMA
        self.context_hidden = context_hidden
        self.n_classes = n_classes
        kwargs = dict(n_classes=n_classes, base_channels=base_channels, dropout=dropout)
        self.joint_backbone = STGCNpp(**kwargs)
        self.bone_backbone = STGCNpp(**kwargs)
        self.feature_dim = self.joint_backbone.feature_dim + self.bone_backbone.feature_dim
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(CONTEXT_DIM), nn.Linear(CONTEXT_DIM, context_hidden),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.position = nn.Parameter(torch.empty(1, self.context_length, context_hidden))
        nn.init.normal_(self.position, mean=0., std=.02)
        # Dilations 1,2,4 give the default center token a 15-token receptive
        # field, covering the full 13-observation sequence before final pooling.
        self.temporal_blocks = nn.ModuleList(
            _MaskedTemporalBlock(context_hidden, dilation, dropout) for dilation in (1, 2, 4))
        self.fusion_head = nn.Sequential(
            nn.Linear(self.feature_dim + 2 * context_hidden, context_hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(context_hidden, n_classes),
        )
        nn.init.zeros_(self.fusion_head[-1].weight)
        nn.init.zeros_(self.fusion_head[-1].bias)

    @staticmethod
    def _check_joints(joints: torch.Tensor) -> None:
        if joints.ndim != 5 or tuple(joints.shape[1:]) != (3, 30, 17, 2) or joints.shape[0] < 1:
            raise ValueError("joints must have shape [N,3,30,17,2] with N >= 1")

    def _skeleton(self, joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_joints(joints)
        joint_features = self.joint_backbone.forward_features(joints)
        bone_features = self.bone_backbone.forward_features(make_stream(joints, "bone"))
        joint_logits = self.joint_backbone.fc(self.joint_backbone.drop(joint_features))
        bone_logits = self.bone_backbone.fc(self.bone_backbone.drop(bone_features))
        return (joint_logits + bone_logits) * .5, torch.cat((joint_features, bone_features), dim=1)

    def _context_features(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observed = context[..., :1] > .5
        clean = context.masked_fill(~observed, 0)
        encoded = (self.context_encoder(clean) + self.position).masked_fill(~observed, 0)
        for block in self.temporal_blocks:
            encoded = block(encoded, observed)
        pooled = encoded.sum(dim=1) / observed.sum(dim=1).clamp_min(1)
        features = torch.cat((encoded[:, self.context_radius], pooled), dim=1)
        return features, observed.any(dim=1)

    def forward_skeleton(self, joints: torch.Tensor) -> torch.Tensor:
        """Learned joint/bone prediction with no detector contribution."""
        return self._skeleton(joints)[0]

    def forward(self, joints: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        self._check_joints(joints)
        if (context.ndim != 3 or context.shape[0] != joints.shape[0]
                or tuple(context.shape[1:]) != (self.context_length, self.context_dim)):
            raise ValueError(f"context must have shape [N,{self.context_length},{self.context_dim}]")
        skeleton_logits, skeleton_features = self._skeleton(joints)
        context_features, any_observed = self._context_features(context)
        delta = self.fusion_head(torch.cat((skeleton_features, context_features), dim=1))
        return skeleton_logits + delta.masked_fill(~any_observed, 0)


__all__ = ["TEMPORAL_FORMAT", "TemporalMultilabelFusion"]
