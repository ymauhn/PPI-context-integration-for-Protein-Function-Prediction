# ppi_v31/extract_raw.py
import argparse
from pathlib import Path

from .prot_t5 import extract_and_save_raw_t5


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract ProtT5 embeddings (last 3 encoder layers) and save as per-protein .npy files."
    )
    ap.add_argument("--domain", required=True, choices=["bp", "cc", "mf"], help="GO domain (lowercase).")
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory containing <domain>_{train,val,test}.csv.")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for per-protein .npy files.")
    ap.add_argument("--batch-size", type=int, default=256, help="Initial batch size (auto-reduced on CUDA OOM).")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16", help="Model dtype (CUDA only).")
    args = ap.parse_args()

    names_tr, names_v, names_t = extract_and_save_raw_t5(
        args.domain,
        args.data_dir,
        args.out,
        batch_size=args.batch_size,
        precision=args.precision,
    )

    print(
        f"Saved to: {args.out} | "
        f"train={len(names_tr)} val={len(names_v)} test={len(names_t)}"
    )


if __name__ == "__main__":
    main()
