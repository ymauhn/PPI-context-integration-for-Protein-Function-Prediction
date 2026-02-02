# ppi_v31/PPIImageFixed_v31.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from .models import build_image_backbone


# =========================
# Utilities: embeddings
# =========================
def _load_channel(
    raw_dir: Path,
    ids: List[str],
    ch: str,
    device: torch.device,
    *,
    strict: bool = False,
    dim: int = 1024,
) -> torch.Tensor:
    """
    Loads a single ProtT5 channel for all protein IDs.

    Expected file name:
        <raw_dir>/<PROTEIN_ID>-<ch>.npy

    Args:
        raw_dir: directory containing .npy embeddings
        ids: list of protein IDs
        ch: channel suffix ("24", "23", "22")
        device: torch device
        strict: if True, raise if any embedding is missing
        dim: embedding dimension (default: 1024)

    Returns:
        Tensor of shape (N, dim) on `device`, dtype float32.
    """
    raw_dir = Path(raw_dir)
    X = []
    missing = 0

    for pid in ids:
        p = raw_dir / f"{pid}-{ch}.npy"
        if p.exists():
            x = np.load(p).astype(np.float32, copy=False)
            if x.ndim != 1 or x.shape[0] != dim:
                raise ValueError(f"Invalid embedding shape in {p}: expected ({dim},), got {x.shape}")
        else:
            missing += 1
            if strict:
                raise FileNotFoundError(f"Missing embedding file: {p}")
            x = np.zeros(dim, dtype=np.float32)

        X.append(x)

    if missing > 0 and not strict:
        # Keep this as a warning (not an exception) to preserve your current behavior.
        print(f"[warn] Missing {missing} embeddings for channel {ch}; filled with zeros.")

    return torch.tensor(np.stack(X, axis=0), dtype=torch.float32, device=device)


# =========================
# Utilities: sparse graph
# =========================
def _coalesce_undirected_from_csv(
    ppi_csv: Path,
    name2idx: dict,
    thr: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds an undirected COO edge list (idx, val) from a PPI CSV.

    The CSV is expected to have:
      - column 0: protein id u
      - column 1: protein id v
      - column 2: confidence/weight (float)

    Edges with weight < thr are dropped.
    Undirected edges are created by duplicating (u,v) and (v,u).
    Multiple edges are coalesced by summing weights.

    Returns:
        idx: LongTensor of shape (2, E)
        val: FloatTensor of shape (E,)
    """
    import pandas as pd

    ppi_csv = Path(ppi_csv)
    if not ppi_csv.exists():
        raise FileNotFoundError(f"PPI CSV not found: {ppi_csv}")

    df = pd.read_csv(ppi_csv)

    if df.shape[1] < 3:
        raise ValueError(
            f"{ppi_csv} must have at least 3 columns: (u, v, confidence). "
            f"Found {df.shape[1]}."
        )

    a = df.iloc[:, 0].astype(str).to_numpy()
    b = df.iloc[:, 1].astype(str).to_numpy()
    s = pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    src, dst, val = [], [], []
    kept = 0
    dropped_thr = 0
    dropped_missing = 0

    for u, v, w in zip(a, b, s):
        w = float(w)
        if w < thr:
            dropped_thr += 1
            continue

        iu = name2idx.get(u)
        iv = name2idx.get(v)
        if iu is None or iv is None:
            dropped_missing += 1
            continue

        # add both directions
        src.extend([iu, iv])
        dst.extend([iv, iu])
        val.extend([w, w])
        kept += 1

    if len(src) == 0:
        idx_out = torch.empty((2, 0), dtype=torch.long)
        val_out = torch.empty((0,), dtype=torch.float32)
        return idx_out, val_out

    idx = torch.tensor([src, dst], dtype=torch.long)
    val = torch.tensor(val, dtype=torch.float32)

    idx_coal, val_coal = _coalesce(idx, val)

    # Optional but helpful log (kept small)
    print(
        f"[ppi] edges kept={kept} (undirected={len(val)}) | "
        f"coalesced={val_coal.numel()} | dropped_thr={dropped_thr} | dropped_missing_ids={dropped_missing}"
    )

    return idx_coal, val_coal


def _coalesce(idx: torch.Tensor, val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch/Numpy coalesce for COO (sum duplicate entries).
    Similar to torch_sparse.coalesce, but without external deps.
    """
    if idx.numel() == 0:
        return idx, val

    i = idx[0].cpu().numpy().astype(np.int64, copy=False)
    j = idx[1].cpu().numpy().astype(np.int64, copy=False)
    v = val.cpu().numpy().astype(np.float32, copy=False)

    if i.size == 0:
        return idx, val

    N = int(max(i.max(), j.max()) + 1)

    # stable hashing key for sorting
    key = i * (N + 1) + j
    order = np.argsort(key, kind="mergesort")

    i2 = i[order]
    j2 = j[order]
    v2 = v[order]

    _, start = np.unique(key[order], return_index=True)
    sums = np.add.reduceat(v2, start)

    out_idx = torch.tensor(np.vstack([i2[start], j2[start]]), dtype=torch.long)
    out_val = torch.tensor(sums, dtype=torch.float32)
    return out_idx, out_val


def _normalize(
    idx: torch.Tensor,
    val: torch.Tensor,
    N: int,
    *,
    norm: str = "col",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Returns a normalized sparse adjacency A (COO, coalesced).

    - norm="col": divide each edge weight by column sum (denominator indexed by j)
    - norm="row": divide each edge weight by row sum    (denominator indexed by i)
    """
    if idx.numel() == 0:
        return torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()

    if device is None:
        device = val.device

    i, j = idx[0].to(device), idx[1].to(device)
    val = val.to(device)

    if norm == "row":
        denom = torch.zeros(N, dtype=val.dtype, device=device).index_add(0, i, val).clamp_min_(1e-12)
        val_n = val / denom[i]
    else:  # "col"
        denom = torch.zeros(N, dtype=val.dtype, device=device).index_add(0, j, val).clamp_min_(1e-12)
        val_n = val / denom[j]

    A = torch.sparse_coo_tensor(torch.stack([i, j], dim=0), val_n, (N, N))
    return A.coalesce()


def _convexize(idx: torch.Tensor, val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds B = A + A^T in COO format (then coalesces).
    """
    if idx.numel() == 0:
        return idx, val

    i, j = idx
    idx_sym = torch.cat([idx, torch.stack([j, i], dim=0)], dim=1)
    val_sym = torch.cat([val, val], dim=0)
    return _coalesce(idx_sym, val_sym)


# =========================
# Fixed-alpha message passing
# =========================
class FixedAlphaSPMM(nn.Module):
    """
    H^t = alpha * H^{t-1} + (1 - alpha) * (A^T @ H^{t-1})

    - alpha is a fixed scalar (non-trainable).
    - A^T is stored as a sparse COO buffer (float32).
    - Runs `hops` propagation steps.

    Note: torch.sparse.mm on CUDA does not support half for sparse inputs reliably,
    so we force the internal computation to float32.
    """

    def __init__(self, A_T: torch.Tensor, hops: int, alpha: float = 0.5):
        super().__init__()
        self.hops = int(hops)

        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        self.register_buffer("A_T", A_T.coalesce().to(torch.float32))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        out_dtype = H.dtype
        H_work = H.to(torch.float32)

        A_T = self.A_T.to(H_work.device)
        alpha = self.alpha.to(H_work.device)

        if self.hops <= 0:
            return H_work.to(out_dtype)

        if H_work.device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=False):
                for _ in range(self.hops):
                    agg = torch.sparse.mm(A_T, H_work)
                    H_work = alpha * H_work + (1.0 - alpha) * agg
        else:
            for _ in range(self.hops):
                agg = torch.sparse.mm(A_T, H_work)
                H_work = alpha * H_work + (1.0 - alpha) * agg

        return H_work.to(out_dtype)


# =========================
# Vector -> image
# =========================
class _ReshapeToImage(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-50.0, 50.0).sigmoid()
        return x.view(-1, 3, 32, 32).contiguous()


# =========================
# Top-level model (fixed-only)
# =========================
class PPIImageModelFixedV31(nn.Module):
    def __init__(
        self,
        ids: List[str],
        raw_dir: Path,
        ppi_csv: Path,
        hops: int = 3,
        alpha_init: float = 0.5,
        thr: float = 0.05,
        arch: str = "resnet50",
        n_classes: int = 100,
        pretrained: bool = True,
        unfreeze: str = "last2",
        norm: str = "col",
        convexize: bool = False,
        device: Optional[torch.device] = None,
        *,
        strict_raw: bool = False,
    ):
        super().__init__()

        self.ids = list(ids)
        self.name2idx = {pid: i for i, pid in enumerate(self.ids)}
        self.device_ = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------
        # 1) Load raw embeddings H0
        # ------------------------
        X24 = _load_channel(raw_dir, self.ids, "24", self.device_, strict=strict_raw)
        X23 = _load_channel(raw_dir, self.ids, "23", self.device_, strict=strict_raw)
        X22 = _load_channel(raw_dir, self.ids, "22", self.device_, strict=strict_raw)
        H0 = torch.cat([X24, X23, X22], dim=1)  # (N, 3072)
        self.register_buffer("H0", H0.to(torch.float32))

        # ------------------------
        # 2) Build sparse PPI graph
        # ------------------------
        idx, val = _coalesce_undirected_from_csv(ppi_csv, self.name2idx, thr)
        N = len(self.ids)

        if convexize:
            idx, val = _convexize(idx, val)

        A = _normalize(idx, val, N, norm=norm, device=self.device_)
        self.A_T = A.transpose(0, 1).coalesce()

        # ------------------------
        # 3) Fixed message passing + cached propagation
        # ------------------------
        self.hops = int(hops)
        self.mp = FixedAlphaSPMM(self.A_T, hops=self.hops, alpha=alpha_init)

        if self.hops > 0:
            H_prop_once = self.mp(self.H0)
        else:
            H_prop_once = self.H0

        self.register_buffer("H_prop_cached", H_prop_once.to(torch.float32))

        # ------------------------
        # 4) Vision backbone
        # ------------------------
        self.to_image = _ReshapeToImage()
        self.backbone = build_image_backbone(
            arch,
            n_classes,
            unfreeze=unfreeze,
            pretrained=pretrained,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B,) or (B,1) of global indices
        idx = idx.view(-1)
        x_vec = self.H_prop_cached.index_select(0, idx)
        x = self.to_image(x_vec)
        return self.backbone(x)


def build_ppi_image_model_fixed_v31(**kwargs) -> PPIImageModelFixedV31:
    return PPIImageModelFixedV31(**kwargs)
