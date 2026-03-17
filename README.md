# Protein Function Prediction based on PPI Context + ProtT5 (E2E)

The paper is available **here** (link coming soon).

## Description

PPI v3.1 is an end-to-end (E2E) pipeline for **multi-label protein function prediction (Gene Ontology)** that integrates:
- **Transformer-based sequence embeddings** (ProtT5; last three encoder layers),
- **PPI-based context integration** via **fixed (non-trainable) multi-hop message passing**, and
- a **vision backbone** (e.g., ConvNeXt-Tiny / ResNet-50) trained on a compact **3×32×32** representation.

In our setup, each protein is represented by three ProtT5 layer vectors (3 × 1024 = 3072), optionally contextualized on the PPI graph, reshaped into a 3×32×32 “image”, and fed to a vision model to predict GO terms for **BP**, **CC**, and **MF**.

## Installation

To install and set up the project, follow the steps below:

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

> **PyTorch/CUDA note:** If you use GPU, install `torch`/`torchvision` using the official PyTorch selector for your CUDA version, then install the remaining packages from `requirements.txt`.

## Dataset

This repository **does not ship** large datasets or embeddings.

The expected data layout and file formats are described in **[`DATASET.md`](DATASET.md)**.

If/when we publish the dataset externally, links will be added here:
- Dataset (CSV splits + PPI + GO OBO): **Zenodo link coming soon**
- IC values: **Zenodo link coming soon**

## Preprocessing (ProtT5 embeddings)

If you do not have raw embeddings yet, you can extract them from the **`sequence`** column of the split CSVs.

In the repository root, run:

```bash
python -m ppi_v31.extract_raw   --domain bp   --data-dir /path/to/data   --out /path/to/raw/raw_bp   --batch-size 256   --precision bf16
```

This will save, for each protein ID `<ID>`:
- `<ID>-24.npy` (last ProtT5 encoder layer)
- `<ID>-23.npy` (second-to-last layer)
- `<ID>-22.npy` (third-to-last layer)

## Reproducibility

To reproduce the E2E experiments:

1. Prepare the dataset following **[`DATASET.md`](DATASET.md)**.
2. (Optional) Extract ProtT5 embeddings using the command above.
3. Run the E2E entrypoint:

   ```bash
   python -m ppi_v31.fixed_e2e_v31 --help
   ```

A command template (same style as used in our experiments):

```bash
python -m ppi_v31.fixed_e2e_v31   --domain bp   --data-dir /path/to/data   --raw-dir  /path/to/raw/raw_bp   --ppi-csv  /path/to/data/ppi.csv   --split test   --hops 5   --alpha-init 0.5   --thr 0.0   --norm col   --convexize 0   --arch convnext_tiny   --unfreeze all   --pretrained 1   --batch-size 64   --lr 1e-4   --max-epochs 100   --patience 20   --num-workers 4   --loss-name protein   --pos-weight-mode ic   --runs-dir /path/to/runs_v31_new   --seed 1337
```

All commands used in the paper (hops sweep, alpha sweep, backbone comparison) are listed in **[`ARTICLE_RUNS.md`](ARTICLE_RUNS.md)**.

## Smoke test (no real data required)

To verify that the entrypoints and paths work end-to-end **without** real data, run:

```bash
python scripts/smoke_e2e.py
```

This generates a tiny synthetic dataset and runs 1 epoch with `--pretrained 0` to avoid downloads.

## Dataset Adaptation

If you need to run this pipeline on your own dataset, you must create a dataset with the **same structure** as ours:

- For each ontology (`bp`, `cc`, `mf`), provide:
  - `<domain>_train.csv`, `<domain>_val.csv`, `<domain>_test.csv`
  - First column: protein ID
  - Second column: protein sequence (amino acids)
  - Remaining columns: GO terms in one-hot encoding format (0/1)

- Provide:
  - `<domain>_ic.csv` with columns `terms, IC` (required when using `--pos-weight-mode ic`)
  - `ppi.csv` with **three columns**: `(protein_A, protein_B, score)`
  - `go.obo` (Gene Ontology OBO file)

See **[`DATASET.md`](DATASET.md)** for the exact expected layout.

## Citation

This repository contains the source code of **"From Graphs to Images: Non-Parametric PPI Context Integration for Vision-Based Protein Function Prediction"**, as presented in the paper:

Yeonatan Mauhnoom, Gabriel Bianchin de Oliveira, Helio Pedrini, Zanoni Dias.  
*"From Graphs to Images: Non-Parametric PPI Context Integration for Vision-Based Protein Function Prediction,"*  
in Proceedings of the **21st International Conference on Computer Vision Theory and Applications (VISAPP 2026)**, Volume 1,  
Marbella, Spain, March 9–11, 2026, pp. 157–166.

If you use this source code and/or its results, please cite our publication:
```bibtex
@inproceedings{mauhnoom2026visapp,
  author    = {Mauhnoom, Yeonatan and de Oliveira, Gabriel Bianchin and Pedrini, Helio and Dias, Zanoni},
  title     = {From Graphs to Images: Non-Parametric PPI Context Integration for Vision-Based Protein Function Prediction},
  booktitle = {Proceedings of the 21st International Conference on Computer Vision Theory and Applications (VISAPP 2026)},
  volume    = {1},
  pages     = {157--166},
  address   = {Marbella, Spain},
  month     = {Mar},
  year      = {2026},
  publisher = {SCITEPRESS -- Science and Technology Publications, Lda.},
  isbn      = {978-989-758-804-4},
  issn      = {2184-4321},
  DOI = {10.5220/0000216600004084}
}

