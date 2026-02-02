 \# PPI v3.1 — Protein Function Classification (E2E)

 

 This repository contains the \*\*core code\*\* used in our paper experiments for \*\*protein function prediction\*\*

 by combining:

 

 - \*\*Sequence embeddings\*\* from \*\*ProtT5\*\* (last 3 encoder layers),

 - \*\*PPI-based context integration\*\* via fixed, non-parametric message passing,

 - A \*\*vision backbone\*\* (e.g., ConvNeXt-Tiny / ResNet-50) trained on a compact \*\*3×32×32\*\* image-like representation.

 

 > The code is provided as a \*\*core-only\*\* release focused on the E2E pipeline used in the article.

 



 
## What is included

 - End-to-end entrypoint used in the paper:

 &nbsp; - `python -m ppi\_v31.fixed\_e2e\_v31 ...`

 - Fixed PPI contextualization + image construction (`PPIImageFixed\_v31.py`)

 - Training loop (`engine.py`), loss (`losses.py`), metrics (`metrics.py`)

 - Vision backbones (`models.py`)

 - Optional utility to extract ProtT5 embeddings (`extract\_raw.py`, `prot\_t5.py`)

 

## What is NOT included

# Large files are not shipped:

 - CSV splits (`\*\_train.csv`, `\*\_val.csv`, `\*\_test.csv`)

 - Information content (`\*\_ic.csv`)

 - PPI edge list (`ppi.csv`)

 - Gene Ontology file (`go.obo`)

 - Extracted embeddings (`\*.npy`)

 - Training artifacts (checkpoints, logs, runs)



 See \*\*\[`DATASET.md`](DATASET.md)\*\* for the exact data layout and file formats expected by the code.

 





# Installation



## Requirements

 Python 3.10+

 PyTorch + torchvision (GPU recommended for training)

 NumPy / Pandas / tqdm

 (Optional) `transformers` + `sentencepiece` for ProtT5 extraction

 

 Install dependencies:

 

 ```bash

 pip install -r requirements.txt

 ```

 

 > Note: For CUDA-enabled PyTorch, you may prefer installing `torch`/`torchvision` via the official PyTorch selector,

 then install the remaining dependencies with `pip install -r requirements.txt`.

 



 

# Data layout (required)

 

 The E2E pipeline assumes:

 - split CSVs with columns: `ID`, `sequence`, then GO term labels (0/1)

 - per-protein embeddings stored under `--raw-dir` as:

 &nbsp; - `<ID>-24.npy`, `<ID>-23.npy`, `<ID>-22.npy`

 

Full details: \*\*\[`DATASET.md`](DATASET.md)\*\*.





 

# Run the E2E pipeline (paper entrypoint)

 

 Example command template (same style as in the paper):

# 

 ```bash

 python -m ppi\_v31.fixed\_e2e\_v31   --domain bp   --data-dir /path/to/ppi\_protein\_function\_pipeline/data   --raw-dir  /path/to/ppi\_protein\_function\_pipeline/raw/raw\_bp   --ppi-csv  /path/to/ppi\_protein\_function\_pipeline/data/ppi.csv   --split test   --hops 5   --alpha-init 0.5   --thr 0.0   --norm col   --convexize 0   --arch convnext\_tiny   --unfreeze all   --pretrained 1   --batch-size 64   --lr 1e-4   --max-epochs 100   --patience 20   --num-workers 4   --loss-name protein   --pos-weight-mode ic   --runs-dir /path/to/runs\_v31\_new   --seed 1337

 ```



All commands used for the experiments (hops sweep, alpha sweep, backbone comparison) are listed in:

 \*\*\[`ARTICLE\_RUNS.md`](ARTICLE\_RUNS.md)\*\*

 


 

# Extract ProtT5 embeddings (optional preprocessing)

 

 If you do \*\*not\*\* have embeddings yet, you can generate them from the `sequence` column in the CSV splits:

 

 ```bash

 python -m ppi\_v31.extract\_raw   --domain bp   --data-dir /path/to/ppi\_protein\_function\_pipeline/data   --out /path/to/ppi\_protein\_function\_pipeline/raw/raw\_bp   --batch-size 256   --precision bf16

 ```

 

 This will save, for each protein ID `<ID>`:

 

    - `<ID>-24.npy` (last encoder layer)

    - `<ID>-23.npy` (second-to-last)

    - `<ID>-22.npy` (third-to-last)




# Reproducibility notes

- This repo intentionally tracks \*\*code only\*\*. Data and large artifacts must be provided externally.

- Use `--seed` for deterministic behavior where applicable.

- Ensure your local environment matches the paper setup (PyTorch + CUDA version, etc.).





