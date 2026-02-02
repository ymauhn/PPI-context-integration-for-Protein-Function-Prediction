from __future__ import annotations

import torch
import torch.nn as nn


class ProteinLoss(nn.Module):
    """
    ProteinLoss used in the SUPERMAGOv2-style setup.

    Final loss:
        CE(y_pred, y_true) * (1 - weighted_F1_GO) * (1 - weighted_F1_protein)

    Notes:
    - `weight_tensor` should be a 1D array-like of shape (n_classes,), e.g. IC values.
    - If `weight_tensor` is None, uniform weights are used.
    """

    def __init__(self, weight_tensor=None, device: str | torch.device = "cpu"):
        super().__init__()
        self._init_weights(weight_tensor=weight_tensor, device=device)

    def _init_weights(self, weight_tensor, device):
        # Default: uniform weights (scalar 1.0) — will broadcast correctly.
        if weight_tensor is None:
            self.register_buffer("weight_tensor", torch.tensor(1.0, dtype=torch.float32))
            return

        # Convert to tensor on the requested device.
        w = torch.as_tensor(weight_tensor, dtype=torch.float32, device=device)

        # If it's a scalar, keep scalar; otherwise ensure 1D.
        if w.ndim > 1:
            w = w.view(-1)

        self.register_buffer("weight_tensor", w)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: logits, shape (B, C)
            y_true: binary labels, shape (B, C)
        """
        sig_y_pred = torch.sigmoid(y_pred)
        ce = self._multilabel_categorical_crossentropy(y_pred, y_true)
        go_f = self._weighted_f1_loss(sig_y_pred, y_true, centric="go")
        pr_f = self._weighted_f1_loss(sig_y_pred, y_true, centric="protein")
        return ce * go_f * pr_f

    @staticmethod
    def _multilabel_categorical_crossentropy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Numerically-stable multilabel categorical cross-entropy (logits form).
        y_pred = (1.0 - 2.0 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e16
        y_pred_pos = y_pred - (1.0 - y_true) * 1e16
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return torch.mean(neg_loss + pos_loss)

    def _weighted_f1_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        beta: float = 1.0,
        centric: str = "protein",
        eps: float = 1e-16,
    ) -> torch.Tensor:
        """
        Computes 1 - mean(weighted F1).

        centric="protein": compute per-protein F1 (over classes), average over batch
        centric="go":      compute per-GO-term F1 (over proteins), average over classes
        """
        dim = 1 if centric == "protein" else 0
        w = self.weight_tensor

        tp = torch.sum(y_true * y_pred * w, dim=dim)
        fp = torch.sum((1.0 - y_true) * y_pred * w, dim=dim)
        fn = torch.sum(y_true * (1.0 - y_pred) * w, dim=dim)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        f1 = (1.0 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

        return 1.0 - torch.mean(f1)
