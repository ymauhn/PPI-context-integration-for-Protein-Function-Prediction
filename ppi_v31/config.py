from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

"""
Central configuration defaults for the PPI v3.1 E2E pipeline.

These values can be overridden via CLI flags in `fixed_e2e_v31.py`.
"""

# Default directory where runs (checkpoints/metrics/logs) are saved.
RUNS_DIR: Path = Path("runs_v31")

# Default training configuration
DEFAULT_CFG: Dict[str, Any] = dict(
    batch_size=64,
    lr=1e-4,
    max_epochs=50,
    patience=10,
    num_workers=4,
    loss_name="protein",       # "protein" | "bce"
    pos_weight_mode="ic",      # "none" | "ic"
    resnet_unfreeze="all",   # "none" | "last2" | "all"
)

# Gene Ontology domain metadata
DOM_INFO = {
    "bp": {"type": "biological_process", "root": "GO:0008150"},
    "cc": {"type": "cellular_component", "root": "GO:0005575"},
    "mf": {"type": "molecular_function", "root": "GO:0003674"},
}
