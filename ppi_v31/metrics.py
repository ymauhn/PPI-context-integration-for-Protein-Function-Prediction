# ppi_v31/metrics.py
from __future__ import annotations

import numpy as np


def get_ancestors(ontology: dict, term: str) -> list[str]:
    """
    Returns all ancestors of `term` (including itself) by traversing parent links.
    """
    queue = [term]
    out = []
    seen = set()

    while queue:
        t = queue.pop(0)
        if t in seen:
            continue
        seen.add(t)

        if t not in ontology:
            continue

        out.append(t)
        for parent in ontology[t].get("parents", []):
            if parent in ontology:
                queue.append(parent)

    return out


def generate_ontology(file, specific_space: bool = False, name_specific_space: str = "") -> dict:
    """
    Parses a GO OBO file and builds an ontology dict containing:
      - parents, children, ancestors, alt_ids, namespace, name
    """
    ontology = {}
    gene = {}
    flag = False

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line == "[Term]":
                if "id" in gene:
                    ontology[gene["id"]] = gene
                gene = {"parents": [], "alt_ids": []}
                flag = True
                continue

            if line == "[Typedef]":
                flag = False
                continue

            if not flag:
                continue

            items = line.split(": ", 1)
            if len(items) != 2:
                continue

            k, v = items[0], items[1]

            if k == "id":
                gene["id"] = v
            elif k == "alt_id":
                gene["alt_ids"].append(v)
            elif k == "namespace":
                if specific_space:
                    if name_specific_space == v:
                        gene["namespace"] = v
                    else:
                        gene = {}
                        flag = False
                else:
                    gene["namespace"] = v
            elif k == "is_a":
                gene["parents"].append(v.split(" ! ")[0])
            elif k == "name":
                gene["name"] = v
            elif k == "is_obsolete":
                gene = {}
                flag = False

        if "id" in gene:
            ontology[gene["id"]] = gene

    # ancestors + alt_ids
    key_list = list(ontology.keys())
    for key in key_list:
        ontology[key]["ancestors"] = get_ancestors(ontology, key)
        for alt_id in ontology[key].get("alt_ids", []):
            ontology[alt_id] = ontology[key]

    # children
    for key, value in ontology.items():
        value.setdefault("children", [])
        for p_id in value.get("parents", []):
            if p_id in ontology:
                ontology[p_id].setdefault("children", [])
                ontology[p_id]["children"].append(key)

    return ontology


def propagate_preds(predictions: np.ndarray, ont_names, ontology: dict) -> np.ndarray:
    """
    Propagates predictions upward: each parent term gets at least the child's score.
    This enforces GO consistency.

    Args:
        predictions: shape (N, C)
        ont_names: list/array of GO IDs with length C
        ontology: dict from generate_ontology

    Returns:
        propagated predictions (same array object is modified and returned)
    """
    ont_n = ont_names.tolist() if hasattr(ont_names, "tolist") else list(ont_names)
    idx_map = {go_id: i for i, go_id in enumerate(ont_n)}

    # Precompute, for each term index, the list of parent indices present in ont_n
    parents_idx = []
    for go_id in ont_n:
        plist = []
        if go_id in ontology:
            for parent in ontology[go_id].get("ancestors", []):
                j = idx_map.get(parent)
                if j is not None:
                    plist.append(j)
        parents_idx.append(list(set(plist)))

    # Apply propagation
    for i in range(predictions.shape[0]):
        row = predictions[i]
        for t_idx, p_list in enumerate(parents_idx):
            if not p_list:
                continue
            v = row[t_idx]
            for p_idx in p_list:
                if row[p_idx] < v:
                    row[p_idx] = v

    return predictions


def evaluate_collect(preds, gt, ont_names, ontology, ic: dict, root: str) -> dict:
    """
    Computes CAFA-style metrics:
      - Fmax
      - Fmax* (root-removed variant)
      - wFmax (IC-weighted)
      - Smin
      - AuPRC
      - IAuPRC
    """
    import math

    preds2 = propagate_preds(preds.copy(), ont_names, ontology)
    wfmax = 0.0
    fmax = 0.0
    fmax_s = 0.0
    smin = 1e100

    pr_arr, rc_arr = [], []

    # Make sure ont_names behaves like an array for boolean indexing
    ont_arr = np.array(ont_names.tolist() if hasattr(ont_names, "tolist") else list(ont_names), dtype=object)

    for tau in np.linspace(0, 1, 101):
        wpr = wrc = num_prot_w = 0.0
        pr_s = rc_s = num_prot_s = 0.0
        pr_n = rc_n = num_prot_n = 0.0
        ru = mi = 0.0

        for i, pred in enumerate(preds2):
            protein_pred = set(ont_arr[pred >= tau].tolist())
            protein_gt = set(ont_arr[gt[i] == 1].tolist())

            ic_pred = sum(ic.get(q, 0.0) for q in protein_pred)
            ic_gt = sum(ic.get(q, 0.0) for q in protein_gt)
            inter = protein_pred.intersection(protein_gt)
            ic_intersect = sum(ic.get(q, 0.0) for q in inter)

            if ic_pred > 0:
                num_prot_w += 1.0
                wpr += (ic_intersect / ic_pred)
            if ic_gt > 0:
                wrc += (ic_intersect / ic_gt)

            if len(protein_pred) > 0:
                num_prot_n += 1.0
                pr_n += len(inter) / len(protein_pred)
            rc_n += len(inter) / max(1, len(protein_gt))

            tp = inter
            fp = protein_pred - tp
            fn = protein_gt - tp
            for go_id in fp:
                mi += ic.get(go_id, 0.0)
            for go_id in fn:
                ru += ic.get(go_id, 0.0)

            # root-removed variant
            protein_pred.discard(root)
            protein_gt.discard(root)

            if len(protein_pred) > 0:
                num_prot_s += 1.0
                pr_s += len(protein_pred.intersection(protein_gt)) / len(protein_pred)
            if len(protein_gt) > 0:
                rc_s += len(protein_pred.intersection(protein_gt)) / len(protein_gt)

        tau_wpr = (wpr / num_prot_w) if num_prot_w > 0 else 0.0
        tau_wrc = wrc / len(preds2)
        if (tau_wrc + tau_wpr) > 0:
            wfmax = max(wfmax, (2 * tau_wpr * tau_wrc) / (tau_wpr + tau_wrc))

        tau_pr_n = (pr_n / num_prot_n) if num_prot_n > 0 else 0.0
        tau_rc_n = rc_n / len(preds2)
        if (tau_pr_n + tau_rc_n) > 0:
            fmax = max(fmax, (2 * tau_pr_n * tau_rc_n) / (tau_pr_n + tau_rc_n))

        pr_arr.append(tau_pr_n)
        rc_arr.append(tau_rc_n)

        ru = ru / len(preds2)
        mi = mi / len(preds2)
        smin = min(smin, math.sqrt((ru * ru) + (mi * mi)))

        tau_pr_s = (pr_s / num_prot_s) if num_prot_s > 0 else 0.0
        tau_rc_s = rc_s / len(preds2)
        if (tau_pr_s + tau_rc_s) > 0:
            fmax_s = max(fmax_s, (2 * tau_pr_s * tau_rc_s) / (tau_pr_s + tau_rc_s))

    pr_arr = np.array(pr_arr)
    rc_arr = np.array(rc_arr)

    idx = np.argsort(rc_arr)
    rc_arr = rc_arr[idx]
    pr_arr = pr_arr[idx]
    auprc = np.trapz(pr_arr, rc_arr)

    ipr_arr, irc_arr = [], []
    for tau in np.linspace(0, 1, 101):
        jj = np.where(rc_arr >= tau)[0]
        if len(jj) != 0:
            irc_arr.append(tau)
            ipr_arr.append(float(np.max(pr_arr[jj[0] :])))
    iauprc = np.trapz(ipr_arr, irc_arr)

    return dict(fmax=fmax, fmax_star=fmax_s, wfmax=wfmax, smin=smin, auprc=auprc, iauprc=iauprc)
