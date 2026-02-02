# ppi_v31/engine.py (v3.1)
from __future__ import annotations

import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .losses import ProteinLoss


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Sets RNG seeds for reproducibility.

    Note: deterministic=True can reduce performance on GPU.
    """
    import numpy as _np
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = bool(deterministic)
    cudnn.benchmark = not bool(deterministic)


class EarlyStopper:
    def __init__(self, patience: int = 25, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.count = 0

    def step(self, val: float) -> bool:
        if val < self.best - self.min_delta:
            self.best = val
            self.count = 0
            return False
        self.count += 1
        return self.count > self.patience


def build_criterion(
    loss_name: str,
    ic_vector: np.ndarray | None,
    device: torch.device,
    pos_weight_mode: str = "none",
):
    loss_name = (loss_name or "protein").lower()

    if loss_name == "protein":
        return ProteinLoss(weight_tensor=ic_vector, device=device)

    if loss_name == "bce":
        if pos_weight_mode == "ic" and ic_vector is not None:
            pw = torch.tensor(ic_vector, dtype=torch.float32, device=device)
            return nn.BCEWithLogitsLoss(pos_weight=pw)
        return nn.BCEWithLogitsLoss()

    raise ValueError(f"Unknown loss_name: {loss_name}")


def run_one_epoch(
    loader,
    model,
    criterion,
    optimizer,
    device: torch.device,
    train: bool = True,
    use_amp: bool = True,
):
    model.train(mode=train)
    total = 0.0

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                logits = model(xb)
                loss = criterion(logits, yb)

        total += float(loss.item())

    return total / max(1, len(loader))


def _plot_losses(tr_hist, vl_hist, save_path: Path, title: str = "Loss"):
    try:
        import matplotlib.pyplot as plt  # local import: avoid hard dependency

        plt.figure()
        plt.plot(range(1, len(tr_hist) + 1), tr_hist, label="train")
        plt.plot(range(1, len(vl_hist) + 1), vl_hist, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"[warn] plot_losses failed: {e}")


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device: torch.device,
    max_epochs: int,
    patience: int,
    run_tag: str,
    save_dir: Path,
    seed: int | None = None,
):
    if seed is not None:
        set_global_seed(int(seed), deterministic=True)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    es = EarlyStopper(patience=patience)
    best_val = float("inf")
    best_path = save_dir / f"{run_tag}_best.pt"

    tr_hist, vl_hist = [], []
    t0 = time.time()
    saved_any = False

    for ep in range(1, int(max_epochs) + 1):
        tr = run_one_epoch(train_loader, model, criterion, optimizer, device, train=True)
        vl = run_one_epoch(val_loader, model, criterion, optimizer, device, train=False)

        tr_hist.append(tr)
        vl_hist.append(vl)

        print(f"Epoch {ep:03d} | train/loss={tr:.4f} | val/loss={vl:.4f}")

        if np.isfinite(vl) and vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_path)
            saved_any = True
            print(f"  ✔ Saved new best: {best_path}")

        if es.step(vl):
            print(f"Early stopping (patience={patience})")
            break

    hist_json = {"train": tr_hist, "val": vl_hist, "best_val": best_val}
    (save_dir / f"{run_tag}_loss_hist.json").write_text(
        json.dumps(hist_json, indent=2),
        encoding="utf-8",
    )
    _plot_losses(tr_hist, vl_hist, save_dir / f"{run_tag}_losses.png", title=run_tag)

    print(f"Done in {time.time() - t0:.1f}s | Best val: {best_val:.4f}")

    return best_path if saved_any and best_path.exists() else None


def predict_proba(loader, model, device: torch.device):
    model.eval()
    all_p, all_y = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            y_np = yb.detach().cpu().numpy()
            all_p.append(probs)
            all_y.append(y_np)

    return np.vstack(all_p), np.vstack(all_y)
