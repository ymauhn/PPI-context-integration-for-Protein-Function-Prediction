# ARTICLE_RUNS.md — Runs used in the paper (E2E only)

All experiments in the paper were performed using the direct end-to-end entrypoint:

```bash
python -m ppi_v31.fixed_e2e_v31   --domain bp   --data-dir /path/to/ppi_protein_function_pipeline/data   --raw-dir  /path/to/ppi_protein_function_pipeline/raw/raw_bp   --ppi-csv  /path/to/ppi_protein_function_pipeline/data/ppi.csv   --split test   --hops 5   --alpha-init 0.5   --thr 0.0   --norm col   --convexize 0   --arch convnext_tiny   --unfreeze all   --pretrained 1   --batch-size 64   --lr 1e-4   --max-epochs 100   --patience 20   --num-workers 4   --loss-name protein   --pos-weight-mode ic   --runs-dir /path/to/ppi_protein_function_pipeline/runs_v31_new   --seed 1337
```

---

## Fixed settings in the paper
Unless explicitly stated otherwise, all runs used:
- `--thr 0.0` (PPI confidence threshold; `0.0` keeps all edges)
- `--norm col`
- `--convexize 0`
- `--pretrained 1`
- `--unfreeze all`
- `--batch-size 64`
- `--lr 1e-4`
- `--max-epochs 100`
- `--patience 20`
- `--loss-name protein`
- `--pos-weight-mode ic`
- `--seed 1337`

---

## Sweep 1 — hops (0 to 10)
We swept the number of hops from 0 to 10 with:
- `alpha-init = 0.5`
- `arch = convnext_tiny`
- `thr = 0.0` (fixed)

Example:

```bash
for h in $(seq 0 10); do
  python -m ppi_v31.fixed_e2e_v31     --domain bp     --data-dir /path/to/ppi_protein_function_pipeline/data     --raw-dir  /path/to/ppi_protein_function_pipeline/raw/raw_bp     --ppi-csv  /path/to/ppi_protein_function_pipeline/data/ppi.csv     --split test     --hops $h     --alpha-init 0.5     --thr 0.0     --norm col     --convexize 0     --arch convnext_tiny     --unfreeze all     --pretrained 1     --batch-size 64     --lr 1e-4     --max-epochs 100     --patience 20     --num-workers 4     --loss-name protein     --pos-weight-mode ic     --runs-dir /path/to/ppi_protein_function_pipeline/runs_v31_new     --seed 1337
done
```

---

## Sweep 2 — alpha-init (0.1 to 0.9)
Using the best hop from Sweep 1, we swept:
- `alpha-init ∈ {0.1, 0.2, ..., 0.9}`
- `thr = 0.0` (fixed)
- `arch = convnext_tiny`

Example:

```bash
BEST_HOP=5  # replace with the best hop found in Sweep 1

for a in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  python -m ppi_v31.fixed_e2e_v31     --domain bp     --data-dir /path/to/ppi_protein_function_pipeline/data     --raw-dir  /path/to/ppi_protein_function_pipeline/raw/raw_bp     --ppi-csv  /path/to/ppi_protein_function_pipeline/data/ppi.csv     --split test     --hops $BEST_HOP     --alpha-init $a     --thr 0.0     --norm col     --convexize 0     --arch convnext_tiny     --unfreeze all     --pretrained 1     --batch-size 64     --lr 1e-4     --max-epochs 100     --patience 20     --num-workers 4     --loss-name protein     --pos-weight-mode ic     --runs-dir /path/to/ppi_protein_function_pipeline/runs_v31_new     --seed 1337
done
```

---

## Sweep 3 — backbone comparison (ResNet vs ConvNeXt Tiny)
Using the best settings from Sweeps 1 and 2 (best hop and best alpha-init), we compared:
- `arch = resnet50`
- `arch = convnext_tiny`

Example:

```bash
BEST_HOP=5     # replace with the best hop
BEST_ALPHA=0.5 # replace with the best alpha-init

for arch in resnet50 convnext_tiny; do
  python -m ppi_v31.fixed_e2e_v31     --domain bp     --data-dir /path/to/ppi_protein_function_pipeline/data     --raw-dir  /path/to/ppi_protein_function_pipeline/raw/raw_bp     --ppi-csv  /path/to/ppi_protein_function_pipeline/data/ppi.csv     --split test     --hops $BEST_HOP     --alpha-init $BEST_ALPHA     --thr 0.0     --norm col     --convexize 0     --arch $arch     --unfreeze all     --pretrained 1     --batch-size 64     --lr 1e-4     --max-epochs 100     --patience 20     --num-workers 4     --loss-name protein     --pos-weight-mode ic     --runs-dir /path/to/ppi_protein_function_pipeline/runs_v31_new     --seed 1337
done
```
