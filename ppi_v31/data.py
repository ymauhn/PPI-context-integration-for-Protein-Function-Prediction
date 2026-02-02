from __future__ import annotations

from pathlib import Path
import pandas as pd


def _read_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    # Ensure there is an ID column (expected name: "ID")
    if "ID" not in df.columns:
        # Fall back to using the first column as ID
        df = df.rename(columns={df.columns[0]: "ID"})

    # Frozen dataset contract:
    # column 0 = ID, column 1 = amino acid sequence, labels start at column 2
    if df.shape[1] < 3:
        raise ValueError(
            f"{path} must have at least 3 columns: "
            f"[ID, sequence, <GO label columns...>]. "
            f"Found only {df.shape[1]} column(s)."
        )

    return df


def load_domain_csvs(domain: str, data_dir: Path):
    """
    Expected files under `data_dir`:
      - {domain}_train.csv
      - {domain}_val.csv
      - {domain}_ic.csv
      - go.obo

    Optional:
      - {domain}_test.csv

    Returns:
      df_tr, df_val, ic_df (columns: 'terms','IC'), go_path
    """
    data_dir = Path(data_dir)

    train_path = data_dir / f"{domain}_train.csv"
    val_path = data_dir / f"{domain}_val.csv"
    test_path = data_dir / f"{domain}_test.csv"
    ic_path = data_dir / f"{domain}_ic.csv"
    go_path = data_dir / "go.obo"

    df_tr = _read_df(train_path)
    df_val = _read_df(val_path)

    # Test is optional here (some callers load it explicitly)
    if test_path.exists():
        _ = _read_df(test_path)

    if not ic_path.exists():
        raise FileNotFoundError(
            f"IC file not found: {ic_path} (expected columns: terms,IC)"
        )

    ic_df = pd.read_csv(ic_path)
    if not {"terms", "IC"}.issubset(set(ic_df.columns)):
        raise ValueError(f"{ic_path} must contain columns 'terms' and 'IC'.")

    return df_tr, df_val, ic_df, go_path
