from __future__ import annotations

import argparse
import json
import time
import random
from pathlib import Path

import numpy as np
import torch

from ppi_v31.config import DEFAULT_CFG, RUNS_DIR, DOM_INFO
from ppi_v31.engine import build_criterion, fit, predict_proba
from ppi_v31.metrics import evaluate_collect, generate_ontology
from ppi_v31.data_utils import load_domain_csvs
from ppi_v31.PPIImageFixed_v31 import build_ppi_image_model_fixed_v31


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(step: str, msg: str) -> None:
    # Always print with flush so logs appear in real time.
    print(f"[{_now()}] STEP {step}: {msg}", flush=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_index_loaders(
    cfg: dict,
    data_dir: Path,
    all_ids: list[str],
    eval_split: str,
):
    """
    Builds DataLoaders for:
      - training: <domain>_train.csv
      - evaluation: <domain>_val.csv OR <domain>_test.csv (selected by --split)

    Returns:
      dl_tr, dl_ev, n_classes, terms, ic_vec, ic_dict, ontology, train_ids, eval_ids
    """
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader

    class _IndexDS(Dataset):
        def __init__(self, ids, labels, name2idx):
            self.ids = list(ids)
            self.y = labels.astype(np.float32)
            self.idx = np.array([name2idx[i] for i in self.ids], dtype=np.int64)

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            return (
                torch.tensor(self.idx[i], dtype=torch.long),
                torch.from_numpy(self.y[i]),
            )

    domain = cfg["domain"]

    # Load train/val/ic/go via existing helper
    df_tr, df_val_default, ic_df, go_path = load_domain_csvs(domain, data_dir)

    # If --split test is requested, replace validation df with <domain>_test.csv
    if eval_split == "test":
        df_eval = pd.read_csv(Path(data_dir) / f"{domain}_test.csv")
    else:
        df_eval = df_val_default

    # Check label columns match
    terms = df_tr.columns[2:]
    if list(terms) != list(df_eval.columns[2:]):
        raise ValueError(
            "Label term columns differ between train and the selected evaluation split."
        )

    # Label matrices
    y_tr = df_tr.iloc[:, 2:].to_numpy(np.float32)
    y_ev = df_eval.iloc[:, 2:].to_numpy(np.float32)

    train_ids = df_tr["ID"].astype(str).values
    eval_ids = df_eval["ID"].astype(str).values

    # IC vector aligned with `terms`
    ic_vec = (
        ic_df.set_index("terms")
        .reindex(terms)["IC"]
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    ic_dict = ic_df.set_index("terms")["IC"].to_dict()

    # Ontology object for evaluation
    ont = generate_ontology(
        go_path,
        specific_space=True,
        name_specific_space=DOM_INFO[domain]["type"],
    )

    # Map protein ID -> global index in H0 tensor (order = all_ids)
    name2idx = {pid: i for i, pid in enumerate(all_ids)}

    ds_tr = _IndexDS(train_ids, y_tr, name2idx)
    ds_ev = _IndexDS(eval_ids, y_ev, name2idx)

    use_gpu = torch.cuda.is_available()

    dl_kwargs = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=use_gpu
    )

    # persistent_workers only valid/useful when num_workers > 0
    if cfg["num_workers"] > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    # Defensive: ensure `shuffle` is not in dl_kwargs (prevents duplicate keyword error)
    dl_kwargs.pop("shuffle", None)

    dl_tr = DataLoader(ds_tr, **{**dl_kwargs, "shuffle": True})
    dl_ev = DataLoader(ds_ev, **{**dl_kwargs, "shuffle": False})
 
    return (
        dl_tr,
        dl_ev,
        len(terms),
        terms,
        ic_vec,
        ic_dict,
        ont,
        train_ids,
        eval_ids,
    )


def main() -> None:
    # Public-repo friendly defaults (relative paths).
    project_root = Path(".").resolve()
    default_data_dir = project_root / "data"
    default_ppi_csv = default_data_dir / "ppi.csv"
    default_raw_bp = project_root / "raw" / "raw_bp"

    ap = argparse.ArgumentParser()

    # Data paths
    ap.add_argument("--domain", choices=["bp", "cc", "mf"], default="bp")
    ap.add_argument("--data-dir", type=Path, default=default_data_dir)
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_bp,
        help="Directory containing {ID}-24/23/22.npy (e.g., raw/raw_bp).",
    )
    ap.add_argument(
        "--ppi-csv",
        type=Path,
        default=default_ppi_csv,
        help="PPI CSV with (u, v, confidence).",
    )

    # Which split becomes the evaluation split
    ap.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Which split to use for final evaluation (default: val).",
    )

    # Fixed message passing
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--alpha-init", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument(
        "--norm",
        choices=["row", "col"],
        default="col",
        help="Adjacency normalization (default: col).",
    )
    ap.add_argument(
        "--convexize",
        type=int,
        default=0,
        help="1 = use A <- A + A^T before normalization.",
    )

    # Vision backbone
    ap.add_argument(
        "--arch",
        type=str,
        choices=["resnet50", "convnext_tiny", "efficientnet_b0", "vit_b_16"],
        default="resnet50",
    )
    ap.add_argument(
        "--unfreeze",
        choices=["none", "last2", "all"],
        default=DEFAULT_CFG["resnet_unfreeze"],
    )
    ap.add_argument("--pretrained", type=int, default=1)

    # Training
    ap.add_argument("--batch-size", type=int, default=DEFAULT_CFG["batch_size"])
    ap.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_CFG["max_epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULT_CFG["patience"])
    ap.add_argument("--num-workers", type=int, default=DEFAULT_CFG["num_workers"])
    ap.add_argument("--loss-name", choices=["protein", "bce"], default=DEFAULT_CFG["loss_name"])
    ap.add_argument("--pos-weight-mode", choices=["none", "ic"], default=DEFAULT_CFG["pos_weight_mode"])
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()

    # Build cfg used internally
    cfg = DEFAULT_CFG.copy()
    cfg.update(
        dict(
            domain=args.domain,
            batch_size=args.batch_size,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            loss_name=args.loss_name,
            pos_weight_mode=args.pos_weight_mode,
            resnet_unfreeze=args.unfreeze,
            seed=args.seed,
        )
    )

    args.runs_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log("0/5", f"device={device}")
    _log("0/5", f"domain={args.domain} split={args.split}")
    _log("0/5", f"data-dir={args.data_dir}")
    _log("0/5", f"raw-dir={args.raw_dir}")
    _log("0/5", f"ppi-csv={args.ppi_csv}")

    # -------------------------------------------------
    # 1) Build a stable global protein ID list = train ∪ eval_split
    # -------------------------------------------------
    _log("1/5", "Loading train/eval CSVs and building protein id list...")
    import pandas as pd

    df_tr_full, df_val_default, _, _ = load_domain_csvs(args.domain, args.data_dir)

    if args.split == "test":
        df_eval_full = pd.read_csv(args.data_dir / f"{args.domain}_test.csv")
    else:
        df_eval_full = df_val_default

    tr_ids_list = df_tr_full["ID"].astype(str).tolist()
    ev_ids_list = df_eval_full["ID"].astype(str).tolist()

    seen = set()
    all_ids: list[str] = []
    for pid in tr_ids_list + ev_ids_list:
        if pid not in seen:
            seen.add(pid)
            all_ids.append(pid)

    _log(
        "1/5",
        f"train IDs: {len(tr_ids_list)} | {args.split} IDs: {len(ev_ids_list)} | union: {len(all_ids)}",
    )

    # -------------------------------------------------
    # 2) Build DataLoaders
    # -------------------------------------------------
    _log("2/5", "Creating DataLoaders...")
    (
        dl_tr,
        dl_ev,
        ncls,
        terms,
        ic_vec,
        ic_dict,
        ont,
        _,
        _,
    ) = _build_index_loaders(cfg, args.data_dir, all_ids, args.split)
    _log(
        "2/5",
        f"DataLoaders ready | n_classes={ncls} | train_batches={len(dl_tr)} | eval_batches={len(dl_ev)}",
    )

    # -------------------------------------------------
    # 3) Build model (loads .npy embeddings and PPI graph)
    # -------------------------------------------------
    _log("3/5", "Building model and loading embeddings/PPI... (may take a while)")
    model = build_ppi_image_model_fixed_v31(
        ids=all_ids,
        raw_dir=args.raw_dir,
        ppi_csv=args.ppi_csv,
        hops=args.hops,
        alpha_init=args.alpha_init,
        thr=args.thr,
        arch=args.arch,
        n_classes=ncls,
        pretrained=bool(args.pretrained),
        unfreeze=args.unfreeze,
        norm=args.norm,
        convexize=bool(args.convexize),
        device=device,
    ).to(device)
    _log("3/5", "Model ready ✔")

    # -------------------------------------------------
    # 4) Train
    # -------------------------------------------------
    _log("4/5", "Starting training / early stopping...")
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg["lr"])

    crit = build_criterion(cfg["loss_name"], ic_vec, device, pos_weight_mode=cfg["pos_weight_mode"])

    convex_tag = "_convex" if args.convexize else ""
    run_tag = (
        f"{args.domain}_{args.arch}_v31_fixed"
        f"_h{args.hops}_t{args.thr}_a{args.alpha_init:.2f}"
        f"_{args.norm}{convex_tag}"
        f"{'' if args.pretrained else '_scratch'}"
        f"_lr{args.lr}"
        f"_{args.split}"
    )
    _log("4/5", f"run_tag={run_tag}")

    best_path = fit(
        model,
        dl_tr,
        dl_ev,
        crit,
        opt,
        device,
        cfg["max_epochs"],
        cfg["patience"],
        run_tag,
        args.runs_dir,
    )

    if best_path and Path(best_path).exists():
        _log("4/5", f"Loading best checkpoint: {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=device))

    # -------------------------------------------------
    # 5) Final evaluation
    # -------------------------------------------------
    _log("5/5", f"Evaluating on split '{args.split}'...")
    probs, gts = predict_proba(dl_ev, model, device)

    metrics = evaluate_collect(
        probs,
        gts,
        ont_names=list(terms),  # IMPORTANT: terms is an Index; pass as list
        ontology=ont,
        ic=ic_dict,
        root=DOM_INFO[cfg["domain"]]["root"],
    )

    print(f"\nMetrics ({args.split}):", metrics)

    out_metrics = args.runs_dir / f"{run_tag}_metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _log("5/5", f"Saved metrics to {out_metrics}")

    if best_path:
        print("Saved best checkpoint:", best_path)


if __name__ == "__main__":
    main()
