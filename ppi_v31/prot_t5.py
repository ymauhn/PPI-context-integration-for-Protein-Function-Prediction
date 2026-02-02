# ppi_v31/prot_t5.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm


def _pool_tokens(matrix_embs: np.ndarray, method: str) -> np.ndarray:
    """Pool token embeddings along axis=0."""
    if method == "mean":
        return np.mean(matrix_embs, axis=0)
    raise ValueError(f"Unknown pooling method: {method!r}")


def protein_embedding(token_embs: np.ndarray, pos: List[int], method: str = "mean") -> np.ndarray:
    """
    Collapse chunk-level embeddings back to per-protein embeddings.

    token_embs: (num_chunks, D)
    pos: list of length num_chunks mapping each chunk to the original protein index.
    """
    if len(pos) == 0:
        return np.empty((0, token_embs.shape[-1]), dtype=np.float32)

    out: List[np.ndarray] = []
    last_pos = pos[0]
    cur: List[np.ndarray] = []

    for i in range(len(token_embs)):
        cur_pos = pos[i]
        if cur_pos == last_pos:
            cur.append(token_embs[i])
        else:
            out.append(_pool_tokens(np.asarray(cur), method))
            last_pos = cur_pos
            cur = [token_embs[i]]

    out.append(_pool_tokens(np.asarray(cur), method))
    return np.asarray(out)


def preprocess(df, subseq: int = 1022) -> Tuple[List[str], List[int], List[str]]:
    """
    Convert a split CSV into chunked sequences for ProtT5.

    Expected CSV format:
      - column 0: protein ID
      - column 1: amino acid sequence

    Returns:
      prot_list: list of subsequences (strings)
      positions: mapping each subsequence to original protein row index
      names: list of protein IDs (one per original protein)
    """
    sequences = df.iloc[:, 1].astype(str).values
    names = df.iloc[:, 0].astype(str).values.tolist()

    prot_list: List[str] = []
    positions: List[int] = []

    for i, seq in enumerate(sequences):
        # Chunk into segments of length `subseq`.
        if len(seq) == 0:
            # Keep empty sequence as a single chunk to avoid dropping entries.
            prot_list.append("")
            positions.append(i)
            continue

        n_chunks = int(np.ceil(len(seq) / subseq))
        for idx in range(n_chunks):
            positions.append(i)
            start = idx * subseq
            end = (idx + 1) * subseq
            prot_list.append(seq[start:end])

    return prot_list, positions, names


def get_embeddings_batch(
    seqs: List[str],
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Embed a batch of sequences with ProtT5, returning mean-pooled representations from
    the last 3 encoder layers: [-1], [-2], [-3].

    Returns 3 arrays of shape (B, 1024), dtype float32.
    """
    toks = [" ".join(list(s)) for s in seqs]
    enc = tokenizer(toks, padding=True, truncation=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attention_mask = enc["attention_mask"].to(device, non_blocking=True)

    with torch.no_grad():
        if autocast_dtype is not None and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        h1, h2, h3 = out.hidden_states[-1], out.hidden_states[-2], out.hidden_states[-3]

        # Mean pooling over tokens, masking padding.
        mask = attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        e1 = (h1 * mask).sum(dim=1) / denom
        e2 = (h2 * mask).sum(dim=1) / denom
        e3 = (h3 * mask).sum(dim=1) / denom

    return (
        e1.detach().cpu().numpy().astype(np.float32),
        e2.detach().cpu().numpy().astype(np.float32),
        e3.detach().cpu().numpy().astype(np.float32),
    )


def embed_and_collapse_T5_batched(
    seqs: List[str],
    pos: List[int],
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    batch_size: int = 256,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Embed chunked sequences in batches (with adaptive batch-size reduction on CUDA OOM),
    then collapse chunk-level embeddings back to per-protein embeddings using `pos`.
    """
    n = len(seqs)
    D = 1024

    l1 = np.zeros((n, D), dtype=np.float32)
    l2 = np.zeros((n, D), dtype=np.float32)
    l3 = np.zeros((n, D), dtype=np.float32)

    i = 0
    pbar = tqdm(total=n, desc=f"Embedding {n} chunks (adaptive batch)", leave=False)

    while i < n:
        bs = min(batch_size, n - i)
        try:
            e1, e2, e3 = get_embeddings_batch(seqs[i : i + bs], tokenizer, model, device, autocast_dtype)
            l1[i : i + bs], l2[i : i + bs], l3[i : i + bs] = e1, e2, e3
            i += bs
            pbar.update(bs)

            if device.type == "cuda":
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if bs == 1:
                raise
            batch_size = max(1, bs // 2)

    pbar.close()

    n1 = protein_embedding(l1, pos, method="mean")
    n2 = protein_embedding(l2, pos, method="mean")
    n3 = protein_embedding(l3, pos, method="mean")
    return n1, n2, n3


def load_t5(precision: str = "bf16", device: Optional[torch.device] = None):
    """
    Load ProtT5 encoder model and tokenizer.

    precision: 'fp32' | 'fp16' | 'bf16'
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"

    tok = T5Tokenizer.from_pretrained(model_name)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in dtype_map:
        raise ValueError(f"Invalid precision: {precision!r}. Choose from {list(dtype_map.keys())}.")

    dtype = dtype_map[precision]
    model = T5EncoderModel.from_pretrained(model_name, torch_dtype=dtype).to(device)

    autocast_dtype = dtype if device.type == "cuda" else None
    model.eval()
    return tok, model, device, autocast_dtype


def extract_and_save_raw_t5(
    domain: str,
    data_dir: Path,
    out_dir: Path,
    batch_size: int = 256,
    precision: str = "bf16",
):
    """
    Extract per-protein embeddings for train/val/test of a domain.

    Reads split CSVs from:
      <data_dir>/{domain}_train.csv, {domain}_val.csv, {domain}_test.csv

    Saves three files per protein to out_dir:
      <PID>-24.npy, <PID>-23.npy, <PID>-22.npy
    """
    import pandas as pd
    from .data import load_domain_csvs

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_tr, df_v, df_ic, go_path = load_domain_csvs(domain, data_dir)
    df_t = pd.read_csv(data_dir / f"{domain}_test.csv")

    tok, model, device, autocast_dtype = load_t5(precision=precision)

    def _do_split(df):
        seqs, pos, names = preprocess(df)
        n1, n2, n3 = embed_and_collapse_T5_batched(
            seqs, pos, tok, model, device, batch_size=batch_size, autocast_dtype=autocast_dtype
        )
        for i, pid in enumerate(names):
            np.save(out_dir / f"{pid}-24.npy", n1[i].astype(np.float32))
            np.save(out_dir / f"{pid}-23.npy", n2[i].astype(np.float32))
            np.save(out_dir / f"{pid}-22.npy", n3[i].astype(np.float32))
        return names

    names_tr = _do_split(df_tr)
    names_v = _do_split(df_v)
    names_t = _do_split(df_t)

    if device.type == "cuda":
        del model
        torch.cuda.empty_cache()

    return names_tr, names_v, names_t
