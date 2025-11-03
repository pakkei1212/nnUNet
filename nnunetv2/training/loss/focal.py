import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RobustFocalLoss(nn.Module):
    """
    Focal loss for dense segmentation that accepts targets shaped like nnU-Net expects.
    Input must be logits, not probabilities.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        reduction: str = "mean",
        ignore_index: int | None = -100,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.eps = eps

        if alpha is not None:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha_tensor.ndim == 0:
                alpha_tensor = alpha_tensor.view(1)
            self.register_buffer("alpha", alpha_tensor, persistent=False)
        else:
            self.alpha = None

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1, "Expected channel dim of size 1 for dense targets."
            target = target[:, 0]
        target = target.long()

        log_probs = F.log_softmax(input, dim=1)
        probs = log_probs.exp()

        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            if not torch.any(valid_mask):
                return input.new_zeros(())
            safe_target = torch.where(valid_mask, target, torch.zeros_like(target))
        else:
            valid_mask = None
            safe_target = target

        gather_idx = safe_target.unsqueeze(1)
        log_pt = log_probs.gather(1, gather_idx).squeeze(1)
        pt = probs.gather(1, gather_idx).squeeze(1).clamp_min(self.eps)

        focal_factor = (1.0 - pt) ** self.gamma

        if getattr(self, "alpha", None) is not None:
            alpha = self.alpha.to(input.device)
            if alpha.numel() == 1:
                alpha_t = alpha[0]
            else:
                alpha_t = alpha[safe_target]
        else:
            alpha_t = 1.0

        if not torch.is_tensor(alpha_t):
            alpha_t = torch.tensor(alpha_t, device=input.device, dtype=pt.dtype)
        if alpha_t.ndim == 0:
            alpha_t = alpha_t.expand_as(pt)

        loss = -alpha_t * focal_factor * log_pt

        if valid_mask is not None:
            loss = loss * valid_mask
            if self.reduction == "mean":
                denom = valid_mask.sum()
                if denom == 0:
                    return loss.sum() * 0
                return loss.sum() / denom
            if self.reduction == "sum":
                return loss.sum()
            return loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
