# Dataset & File Layout (Not Included)

This repository does **not** include the dataset used in the paper
(CSV splits, PPI edge list, GO ontology, or raw embeddings),
as these files are large and may have licensing or redistribution constraints.

The code assumes the **exact directory layout and file formats** described below.

---

## 1) Directory layout

### (A) `--data-dir`

This directory must contain all CSV splits, IC files, the PPI edge list, and the GO ontology:

```text
<data-dir>/
  ppi.csv
  go.obo

  bp_train.csv
  bp_val.csv
  bp_test.csv
  bp_ic.csv

  cc_train.csv
  cc_val.csv
  cc_test.csv
  cc_ic.csv

  mf_train.csv
  mf_val.csv
  mf_test.csv
  mf_ic.csv
```

Domains are expected as lowercase strings: `bp`, `cc`, `mf`.

---

### (B) `--raw-dir` (per domain)

This directory contains **per-protein ProtT5 embeddings** saved as `.npy` files.

Example:

```text
<raw-dir>/
  P12345-24.npy
  P12345-23.npy
  P12345-22.npy
  Q9XXXX-24.npy
  Q9XXXX-23.npy
  Q9XXXX-22.npy
  ...
```

Each protein must have **three files**, corresponding to the last three encoder layers
of ProtT5.

---

## 2) Split CSV format (train / val / test)

Each split CSV (e.g., `bp_train.csv`) must follow **exactly** this structure:

| Column index | Description |
|-------------|-------------|
| 0 | **Protein ID** (string, e.g. UniProt accession) |
| 1 | **Amino acid sequence** (string) |
| 2..end | Binary GO labels (one column per GO term) |

Example header:

```text
ID,sequence,GO:0008150,GO:0009987,GO:...
```

Example row:

```text
P12345,MAAAAGVVK...,0,1,...
```

Important notes:
- Labels are read from **column index 2 onward**.
- Protein IDs must match:
  - the embedding filenames in `--raw-dir`
  - the node identifiers used in `ppi.csv`.

---

## 3) IC file format (`<domain>_ic.csv`)

Each IC file must contain at least the following columns:

| Column | Description |
|------|-------------|
| `terms` | GO term identifier (e.g. `GO:0008150`) |
| `IC` | Information content (float) |

Example:

```text
terms,IC
GO:0008150,2.134
GO:0009987,4.012
```

These values are used when running with:

```bash
--pos-weight-mode ic
```

---

## 4) PPI edge list format (`ppi.csv`) — confidence is USED

The PPI file must define interactions between proteins and use the **same protein IDs**
as the split CSVs.

### Required columns
Your pipeline uses the **confidence score**, so `ppi.csv` must include:

| Column index | Description |
|------------|-------------|
| 0 | protein ID of node A |
| 1 | protein ID of node B |
| 2 | **confidence / weight** (float) |

Example:

```text
protein1,protein2,confidence
P12345,Q9XXXX,0.82
Q9XXXX,A0A0B4...,0.11
```

### How it is used
- The confidence value is used as an **edge weight** in the graph contextualization step.
- The `--thr` argument is applied as a **minimum confidence threshold** when building/using the graph (e.g., `--thr 0.0` keeps all edges).

If your dataset uses a different header name (e.g., `score`, `combined_score`, `weight`), that is fine as long as:
- the first two columns are protein IDs
- the third column is the numeric confidence.

---

## 5) Gene Ontology file (`go.obo`)

The Gene Ontology OBO file must be placed at:

```text
<data-dir>/go.obo
```

It is used for GO hierarchy handling and evaluation.

---

## 6) Extracting ProtT5 embeddings (raw features)

Raw embeddings are extracted using the provided `prot_t5.py` script,
which relies on the **ProtT5-XL UniRef50 encoder**.

### Model
- HuggingFace model:  
  `Rostlab/prot_t5_xl_half_uniref50-enc`
- Encoder-only (no decoder)
- Hidden size: 1024

### Procedure
1. Protein sequences are read from the **second column** of the split CSVs.
2. Long sequences are chunked into subsequences of length 1022.
3. Each chunk is embedded with ProtT5.
4. Mean pooling over residues is applied.
5. Chunk embeddings belonging to the same protein are averaged.
6. The **last three encoder layers** are extracted and saved separately.

### Output files
For each protein ID `<PID>`, three files are generated:

```text
<PID>-24.npy   # last encoder layer
<PID>-23.npy   # second-to-last encoder layer
<PID>-22.npy   # third-to-last encoder layer
```

Each file stores a 1D NumPy array of shape `(1024,)` with `float32` dtype.

### Example extraction snippet (conceptual)

```python
from pathlib import Path
from ppi_v31.prot_t5 import extract_and_save_raw_t5

extract_and_save_raw_t5(
    domain="bp",
    data_dir=Path("/path/to/data"),
    out_dir=Path("/path/to/raw/raw_bp"),
    batch_size=256,
    precision="bf16"
)
```

---

## Notes
- Do **not** commit data files or embeddings to the repository.
- The pipeline assumes this exact layout; deviations may cause runtime errors.
- If you wish to share data publicly, consider hosting it on Zenodo or OSF and linking it here.
