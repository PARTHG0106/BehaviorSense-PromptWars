"""Losses for independent, partially observed activity labels.

ASL equations: Ridnik et al., ICCV 2021,
https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html
Reference implementation: https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
This implementation adds known-label masking and stable float32 log probabilities.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def masked_asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean ASL over known binary labels, with no additional class weights.

    For positive labels the likelihood is sigmoid(logit); for negative labels it
    is min(1, sigmoid(-logit) + clip). Each negative log likelihood is multiplied
    by (1 - likelihood) ** gamma, using the label's positive/negative gamma.
    Focal weights are detached, as in the standard reference implementation's
    default. Detaching the weights does not change the caller's gradient mode.

    Unknown entries are selected out before arithmetic and cannot contribute a
    loss or gradient, even when their stored values are nonfinite placeholders.
    Known targets and masks must be binary. Calculations use float32 even inside
    bf16 autocast. Unclipped log probabilities use log-sigmoid to retain finite,
    corrective gradients for confidently wrong predictions. Epsilon protects
    the log of the shifted negative probability when clipping is enabled.
    """
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("ASL logits, targets and known-label mask must have identical shapes")
    if logits.device != targets.device or logits.device != mask.device:
        raise ValueError("ASL logits, targets and mask must be on the same device")
    if not logits.is_floating_point():
        raise ValueError("ASL logits must have a floating-point dtype")
    for name, value in (("gamma_neg", gamma_neg), ("gamma_pos", gamma_pos)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(clip) or not 0 <= clip < 1:
        raise ValueError("clip must be finite and in [0, 1)")
    if not math.isfinite(eps) or not 0 < eps < 1:
        raise ValueError("eps must be finite and in (0, 1)")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("ASL known-label mask must be binary")
    known = mask.bool()
    if not bool(known.any()):
        raise ValueError("ASL batch contains no known labels")
    selected_targets = targets[known]
    if not bool(((selected_targets == 0) | (selected_targets == 1)).all()):
        raise ValueError("ASL known targets must be binary")
    selected_logits = logits[known]
    if not bool(torch.isfinite(selected_logits).all()):
        raise ValueError("ASL known logits must be finite")

    # Explicitly disable local autocast rather than relying on operator policies.
    # Boolean indexing also keeps unknown infinities out of sigmoid/log backward.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        x = selected_logits.float()
        positive = selected_targets.bool()
        positive_probability = torch.sigmoid(x)
        negative_probability = torch.sigmoid(-x)
        positive_log_likelihood = F.logsigmoid(x)
        if clip > 0:
            negative_probability = (negative_probability + clip).clamp(max=1)
            negative_log_likelihood = negative_probability.clamp_min(eps).log()
        else:
            negative_log_likelihood = F.logsigmoid(-x)
        log_likelihood = torch.where(
            positive, positive_log_likelihood, negative_log_likelihood)
        if gamma_neg != 0 or gamma_pos != 0:
            likelihood = torch.where(positive, positive_probability, negative_probability)
            exponent = torch.where(positive, gamma_pos, gamma_neg)
            focal_weight = (1 - likelihood.detach()).pow(exponent)
            log_likelihood = log_likelihood * focal_weight
        return -log_likelihood.mean()
