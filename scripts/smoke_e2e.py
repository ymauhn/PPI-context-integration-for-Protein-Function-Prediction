# scripts/smoke_e2e.py
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


AA = "ACDEFGHIKLMNPQRSTVWY"


def _rand_seq(n: int) -> str:
    rng = np.random.default_rng(0)
    return "".join(rng.choice(list(AA), size=n))


def write_minimal_go_obo(path: Path, terms: list[str], domain_root: str = "GO:0008150") -> None:
    """
    Create a minimal GO OBO file that most OBO parsers can read.
    We attach all terms as children of the domain root (BP by default).
    """
    lines = []
    lines.append("format-version: 1.2")
    lines.append("")

    # Root
    lines.append("[Term]")
    lines.append(f"id: {domain_root}")
    lines.append("name: biological_process")
    lines.append("namespace: biological_process")
    lines.append("")

    for t in terms:
        if t == domain_root:
            continue
        lines.append("[Term]")
        lines.append(f"id: {t}")
        lines.append(f"name: synthetic_{t.replace(':','_')}")
        lines.append("namespace: biological_process")
        lines.append(f"is_a: {domain_root} ! biological_process")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def make_synthetic_layout(base: Path) -> tuple[Path, Path]:
    """
    Creates:
      base/data/   (csv splits, ic, ppi.csv, go.obo)
      base/raw_bp/ (per-protein embeddings)
    """
    data_dir = base / "data"
    raw_dir = base / "raw_bp"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic proteins + GO labels
    proteins = ["P00001", "P00002", "P00003", "P00004"]
    seqs = [_rand_seq(120), _rand_seq(240), _rand_seq(80), _rand_seq(160)]
    go_terms = ["GO:0008150", "GO:0009987"]  # include root + one child

    # Splits: small but valid
    def make_split_df():
        rng = np.random.default_rng(42)
        Y = rng.integers(0, 2, size=(len(proteins), len(go_terms)), dtype=np.int64)
        df = pd.DataFrame({"ID": proteins, "sequence": seqs})
        for j, t in enumerate(go_terms):
            df[t] = Y[:, j]
        return df

    for split in ["train", "val", "test"]:
        df = make_split_df()
        df.to_csv(data_dir / f"bp_{split}.csv", index=False)

    # IC file: must contain columns `terms` and `IC`
    ic_df = pd.DataFrame({"terms": go_terms, "IC": [1.0, 2.0]})
    ic_df.to_csv(data_dir / "bp_ic.csv", index=False)

    # PPI: your code requires 3 columns (u, v, score/confidence)
    ppi_df = pd.DataFrame(
        {
            "protein_A": ["P00001", "P00002", "P00003"],
            "protein_B": ["P00002", "P00003", "P00004"],
            "score": [0.9, 0.7, 0.8],
        }
    )
    ppi_df.to_csv(data_dir / "ppi.csv", index=False)

    # Minimal GO OBO
    write_minimal_go_obo(data_dir / "go.obo", terms=go_terms, domain_root="GO:0008150")

    # Raw embeddings: <ID>-24/-23/-22.npy with shape (1024,)
    rng = np.random.default_rng(123)
    for pid in proteins:
        for suf in ["-24", "-23", "-22"]:
            vec = rng.normal(size=(1024,)).astype(np.float32)
            np.save(raw_dir / f"{pid}{suf}.npy", vec)

    return data_dir, raw_dir


def run_e2e(repo_root: Path, data_dir: Path, raw_dir: Path) -> None:
    """
    Runs a tiny training to verify:
      - entrypoint works as module
      - reads CSV / PPI / GO OBO
      - loads embeddings
      - forward/backward completes
      - writes runs outputs
    """
    runs_dir = repo_root / "_smoke_runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "ppi_v31.fixed_e2e_v31",
        "--domain",
        "bp",
        "--data-dir",
        str(data_dir),
        "--raw-dir",
        str(raw_dir),
        "--ppi-csv",
        str(data_dir / "ppi.csv"),
        "--split",
        "val",
        "--hops",
        "1",
        "--alpha-init",
        "0.5",
        "--thr",
        "0.0",
        "--norm",
        "col",
        "--convexize",
        "0",
        "--arch",
        "convnext_tiny",
        "--unfreeze",
        "none",
        "--pretrained",
        "0",  # IMPORTANT: avoid downloading weights in Colab / CI
        "--batch-size",
        "2",
        "--lr",
        "1e-4",
        "--max-epochs",
        "1",
        "--patience",
        "1",
        "--num-workers",
        "0",
        "--loss-name",
        "protein",
        "--pos-weight-mode",
        "ic",
        "--runs-dir",
        str(runs_dir),
        "--seed",
        "1337",
    ]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo_root))


def main():
    repo_root = Path(__file__).resolve().parents[1]
    smoke_root = repo_root / "_smoke_data"

    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)

    data_dir, raw_dir = make_synthetic_layout(smoke_root)
    run_e2e(repo_root, data_dir, raw_dir)
    print("\n✅ Smoke test passed: E2E ran for 1 epoch with synthetic data.\n")


if __name__ == "__main__":
    main()
