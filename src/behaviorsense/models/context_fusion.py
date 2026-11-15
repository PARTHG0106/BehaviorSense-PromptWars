"""Learned visual-context residual on the unchanged ST-GCN++ skeleton baseline."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from behaviorsense.models.stgcnpp import N_JOINTS, STGCNpp
from behaviorsense.models.visual_context import CONTEXT_DIM, CONTEXT_SCHEMA


class STGCNContextFusion(STGCNpp):
    """ST-GCN++ plus a trainable context encoder and cross-modal residual head.

    ``context`` is [N,D] for one exact source frame/window or [N,K,D] for K
    ordered observations. The first feature is the observed mask. Masked mean
    pooling makes padding inert. All-missing context returns the skeleton logits.

    Baseline parameter names stay unchanged for honest initialization/ablation.
    The final residual layer starts at zero, so importing a baseline initially
    reproduces its logits exactly. There are no per-activity context rules.
    """

    def __init__(self, n_classes: int = 20, in_channels: int = 3,
                 base_channels: int = 64, n_person: int = 2, dropout: float = 0.2,
                 n_joints: int = N_JOINTS, adjacency: np.ndarray | None = None,
                 context_dim: int = CONTEXT_DIM, context_hidden: int = 128) -> None:
        super().__init__(n_classes=n_classes, in_channels=in_channels,
                         base_channels=base_channels, n_person=n_person,
                         dropout=dropout, n_joints=n_joints, adjacency=adjacency)
        if context_dim != CONTEXT_DIM or context_hidden < 1:
            raise ValueError(f"{CONTEXT_SCHEMA} requires context_dim={CONTEXT_DIM} and positive context_hidden")
        self.context_dim = context_dim
        self.context_schema = CONTEXT_SCHEMA
        self.context_hidden = context_hidden
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, context_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(context_hidden, context_hidden), nn.GELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(self.feature_dim + context_hidden, context_hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(context_hidden, n_classes),
        )
        nn.init.zeros_(self.fusion_head[-1].weight)
        nn.init.zeros_(self.fusion_head[-1].bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.ndim == 2:
            context = context.unsqueeze(1)
        if (context.ndim != 3 or context.shape[0] != x.shape[0] or
                context.shape[1] < 1 or context.shape[2] != self.context_dim):
            raise ValueError(f"context must be [N,{self.context_dim}] or [N,K,{self.context_dim}]")
        features = self.forward_features(x)
        baseline = self.fc(self.drop(features))
        observed = (context[..., :1] > 0.5).to(dtype=context.dtype)
        # Discard padded feature values before encoding as well as before pooling.
        encoded = self.context_encoder(context * observed) * observed
        pooled = encoded.sum(dim=1) / observed.sum(dim=1).clamp_min(1)
        delta = self.fusion_head(torch.cat((features, pooled), dim=1))
        return baseline + delta * observed.any(dim=1).to(dtype=delta.dtype)

    def forward_skeleton(self, x: torch.Tensor) -> torch.Tensor:
        """Same trained skeleton branch, usable for paired context-removal evaluation."""
        return super().forward(x)
