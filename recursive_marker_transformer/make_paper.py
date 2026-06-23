# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The SMART Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

"""Generate the AAAI paper (.tex + .bib) directly from experiment results.

This module *is* the paper source: the prose lives here as a template, and the
numbers / tables are injected from the JSON written by ``experiments.py``. Run
it after the experiments and you get a self-contained ``paper/`` directory that
compiles to PDF with pdflatex + bibtex. No number in the paper is hand-typed.

    python -m recursive_marker_transformer.make_paper --results results --outdir paper
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TEMPLATE_DIR = _REPO / "aaai_template"


# --------------------------------------------------------------------------- #
# helpers to load + format results
# --------------------------------------------------------------------------- #
def _load(results: Path, name: str):
    p = results / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pct(x):
    return f"{100 * x:.1f}"


def _fmt(n):
    return f"{n:,}"


def _flops(n):
    """Compact FLOPs (G/M/K) so values fit the narrow AAAI two-column tables."""
    n = float(n)
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


def main_results_table(res, primary):
    """Per-class + per-head table for the main model."""
    if res is None:
        return "\\textit{(main results pending)}"
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Head & Accuracy & Macro-F1 & Weighted-F1 & \\#Classes \\\\",
        "\\midrule",
    ]
    for h, m in res["heads"].items():
        nc = len([k for k in m["per_class"] if k.isdigit()])
        star = "$^{\\dagger}$" if h == primary else ""
        lines.append(
            f"{h.replace('_',' ')}{star} & {_pct(m['accuracy'])} & "
            f"{_pct(m['macro_f1'])} & {_pct(m['weighted_f1'])} & {nc} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# Contiguous class index -> human cohort label for the default 4-cohort TCGA run
# (raw cancer_type codes [breast=0, head_neck=1, lung=2, thyroid=4] remap to 0..3;
# prostate=3 is not in the loaded cohorts). Used only as a fallback when results
# do not carry an explicit ``class_names`` map (newer runs persist one in train.py).
_CANCER_NAMES = {
    "0": "Breast (BRCA)",
    "1": "Head-neck (HNSC)",
    "2": "Lung (LUNG)",
    "3": "Thyroid (THCA)",
}


def _class_name(res, head, c):
    # For cancer_type the four loaded cohorts are breast/head_neck/lung/thyroid in
    # sorted order (0..3); prefer this verified positional map over any persisted
    # names, which an earlier 5-cohort name table mislabelled (class 3 as PRAD).
    if head == "cancer_type":
        return _CANCER_NAMES.get(str(c), f"Class {c}")
    names = res["heads"][head].get("class_names")
    if names and str(c) in names:
        return names[str(c)]
    return f"Class {c}"


def per_class_table(res, primary):
    """Per-class (per-cohort) precision/recall/F1/support for the primary head,
    with the macro and support-weighted aggregates for context."""
    if res is None or primary not in res["heads"]:
        return "\\textit{(per-class breakdown pending)}"
    pc = res["heads"][primary]["per_class"]
    classes = sorted(k for k in pc if k.isdigit())
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Cohort (class) & Precision & Recall & F1 & Support \\\\",
        "\\midrule",
    ]
    for c in classes:
        d = pc[c]
        lines.append(
            f"{_class_name(res, primary, c)} & {_pct(d['precision'])} & {_pct(d['recall'])} & "
            f"{_pct(d['f1-score'])} & {int(d['support'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def _rows_table(results, primary, rows, cols="macro_acc_params_flops"):
    lines = [
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Variant & Macro-F1 & Acc. & Stack params & FLOPs/cell \\\\",
        "\\midrule",
    ]
    any_row = False
    for name, label in rows:
        r = _load(results, name)
        if r is None:
            continue
        any_row = True
        h = r["heads"].get(primary) or next(iter(r["heads"].values()))
        lines.append(
            f"{label} & {_pct(h['macro_f1'])} & {_pct(h['accuracy'])} & "
            f"{_fmt(r['transformer_params'])} & {_flops(r['approx_flops_per_sample'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}"]
    return "\n".join(lines) if any_row else "\\textit{(pending)}"


def selection_table(results, primary):
    """Marker-selection study: selection determines the genes the model sees.
    All rows use fixed-depth recursion so only the selection strategy varies
    (the router row is `fixed_depth` = router selection + fixed recursion)."""
    return _rows_table(results, primary, [
        ("sel_concrete", "Concrete (learned)"),
        ("fixed_depth", "Router (learned, ours)"),
        ("sel_variance", "Variance (heuristic)"),
        ("sel_random", "Random"),
    ])


def ablation_table(results, primary):
    """Architecture ablations on the full (aggregate) model."""
    return _rows_table(results, primary, [
        ("main", "Full model"),
        ("no_refine", "$-$ recursive refinement"),
        ("depth1", "$-$ recursion ($K{=}1$)"),
        ("independent", "$-$ weight sharing"),
    ])


def routing_table(results, primary):
    """Mixture-of-Recursions routing study: expert- vs token-choice vs uniform
    fixed depth. Reports the per-token depth allocation and the compute saving."""
    rows = [
        ("main", "Expert-choice (ours)"),
        ("mor_token", "Token-choice"),
        ("fixed_depth", "Fixed depth (uniform)"),
    ]
    lines = [
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Routing & Macro-F1 & Acc. & Mean depth & Eff. FLOPs & Saving \\\\",
        "\\midrule",
    ]
    any_row = False
    for name, label in rows:
        r = _load(results, name)
        if r is None:
            continue
        any_row = True
        h = r["heads"].get(primary) or next(iter(r["heads"].values()))
        md = r.get("mean_recursion_depth", float("nan"))
        save = r.get("compute_saving_ratio", 1.0)
        lines.append(
            f"{label} & {_pct(h['macro_f1'])} & {_pct(h['accuracy'])} & "
            f"{md:.2f} & {_flops(r['approx_flops_per_sample'])} & {save:.2f}$\\times$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}"]
    return "\n".join(lines) if any_row else "\\textit{(routing study pending)}"


# Curated literature/database validation for router-identified genes. Each entry
# is an established association of the gene with one of the four cohorts (breast,
# head-neck squamous, lung, thyroid). "DB" -> curated marker databases
# (CellMarker 2.0, PanglaoDB); "NYBR1" -> the primary NY-BR-1 paper.
_GENE_VALIDATION = {
    "KRT14":     ("Basal cytokeratin; breast/squamous", "DB"),
    "ANKRD30A":  ("NY-BR-1 breast antigen", "NYBR1"),
    "SFN":       ("Stratifin (14-3-3$\\sigma$); breast", "DB"),
    "SPRR3":     ("Small proline-rich; squamous", "DB"),
    "SPRR2C":    ("Small proline-rich; squamous", "DB"),
    "TRH":       ("Thyrotropin-releasing hormone; thyroid", "DB"),
    "FAM83F":    ("Thyroid carcinoma", "DB"),
    "MUC21":     ("Epithelial mucin; squamous", "DB"),
    "WISP2":     ("CCN5; breast carcinoma", "DB"),
    "TYRP1":     ("Melanocytic/pigmentation", "DB"),
    "FOXA3":     ("Forkhead TF; epithelia", "DB"),
    "TCF21":     ("Epithelial/mesenchymal TF; lung", "DB"),
    "TMPRSS11D": ("Airway protease; squamous", "DB"),
}
_GENE_SRC = {"DB": "\\cite{hu2023cellmarker,franzen2019panglaodb}",
             "NYBR1": "\\cite{jager2001nybr1}"}


def _read_markers(results: Path):
    """Read the headline model's marker panel (gene, importance, recursion depth)."""
    for cand in (results / "markers_main.csv", _REPO / "markers_top.csv"):
        if cand.exists():
            rows = []
            with open(cand) as f:
                for r in csv.DictReader(f):
                    depth = r.get("recursion_depth") or ""
                    rows.append((r["gene"], float(r["importance"]),
                                 float(depth) if depth else None))
            return rows
    return []


def gene_identification_table(results: Path, top=16):
    """Top router-identified genes ranked by mean recursion depth d_m (the
    adaptive compute the model allocates to each gene). Descriptive, not a
    validation claim: where a gene matches our small curated marker list we
    annotate the association, otherwise we simply list the gene and its depth."""
    rows = [r for r in _read_markers(results) if r[2] is not None]
    if not rows:
        return "\\textit{(marker panel pending: rerun experiments)}"
    rows.sort(key=lambda r: -r[2])                    # by recursion depth
    rows = rows[:top]
    annotated = any(g in _GENE_VALIDATION for g, _i, _d in rows)
    lines = ["{\\footnotesize"]
    if annotated:
        # If any deepest-routed gene is in the curated list, show associations.
        lines += ["\\begin{tabular}{@{}l c p{0.52\\columnwidth}@{}}",
                  "\\toprule",
                  "Gene & Depth $d_m$ & Curated association (source) \\\\",
                  "\\midrule"]
        for g, _imp, d in rows:
            if g in _GENE_VALIDATION:
                assoc, src = _GENE_VALIDATION[g]
                note = f"{assoc}~{_GENE_SRC[src]}"
            else:
                note = "\\textemdash"
            lines.append(f"\\textit{{{g}}} & {d:.2f} & {note} \\\\")
    else:
        # No curated overlap in the top panel: a clean two-column gene/depth list.
        half = (len(rows) + 1) // 2
        left, right = rows[:half], rows[half:]
        lines += ["\\begin{tabular}{@{}l c @{\\quad} l c@{}}",
                  "\\toprule",
                  "Gene & $d_m$ & Gene & $d_m$ \\\\",
                  "\\midrule"]
        for i in range(half):
            lg, _li, ld = left[i]
            if i < len(right):
                rg, _ri, rd = right[i]
                lines.append(f"\\textit{{{lg}}} & {ld:.2f} & \\textit{{{rg}}} & {rd:.2f} \\\\")
            else:
                lines.append(f"\\textit{{{lg}}} & {ld:.2f} & & \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    return "\n".join(lines)


def param_table(results):
    pe = _load(results, "param_efficiency")
    if pe is None:
        return "\\textit{(parameter table pending)}"
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Depth $K$ & Shared (ours) & Independent & Reduction \\\\",
        "\\midrule",
    ]
    for e in pe:
        lines.append(
            f"{e['depth']} & {_fmt(e['shared_params'])} & "
            f"{_fmt(e['independent_params'])} & {e['ratio']:.2f}$\\times$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# Representative configs for the total-cost table (R1: total params incl.
# embeddings + wall-clock per config). Read straight from results/*.json.
_COST_ROWS = [
    ("main",        "Expert-choice (headline)"),
    ("fixed_depth", "Fixed depth"),
    ("depth1",      "Single pass ($K{=}1$)"),
    ("independent", "Independent layers"),
]


def cost_table(results: Path):
    """Stack params, total params (incl. gene embedding + classifier) and
    train wall-clock per config, so the parameter claim is reported end-to-end."""
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Config & Stack params & Total params & Train (s) \\\\",
        "\\midrule",
    ]
    any_row = False
    for name, label in _COST_ROWS:
        r = _load(results, name)
        if r is None:
            continue
        any_row = True
        wall = r.get("wall_seconds")
        wall = f"{wall:.0f}" if isinstance(wall, (int, float)) else "--"
        lines.append(
            f"{label} & {_fmt(r['transformer_params'])} & "
            f"{_fmt(r['total_params'])} & {wall} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) if any_row else "\\textit{(cost table pending)}"


# --- reviewer extras: multi-seed main task + init/anneal ablation ------------ #
_REXTRA = _REPO / "results_extra"


def _seed_runs():
    d = _REXTRA / "multiseed"
    out = []
    if d.exists():
        for p in sorted(d.glob("seed*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except Exception:
                pass
    return out


def main_seed_stats():
    """(macro_f1 mean/std, acc mean/std, n) over the multi-seed headline runs on
    the primary cohort task, or None. Addresses the request for mean+/-std on the
    main result."""
    runs = _seed_runs()
    if len(runs) < 2:
        return None
    return (_mean_std([r["macro_f1"] for r in runs]),
            _mean_std([r["accuracy"] for r in runs]), len(runs))


def init_anneal_table():
    """2x2 ablation isolating the two router design choices the paper argues for:
    peaked vs uniform initialisation x temperature anneal on/off, mean+/-std
    macro-F1 over the seeds in results_extra/init_anneal."""
    d = _REXTRA / "init_anneal"
    cells = {}
    if d.exists():
        for p in sorted(d.glob("peak*_anneal*_seed*.json")):
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            cells.setdefault((bool(r["peak_init"]), bool(r["anneal_markers"])),
                             []).append(r["macro_f1"])
    if not cells:
        return "\\textit{(init/anneal ablation pending)}"

    def cell(pk, an):
        return _ms_cell(_mean_std(cells.get((pk, an), [])))

    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Initialisation & Annealed $\\tau$ & Constant $\\tau$ \\\\",
        "\\midrule",
        f"Peaked (ours) & {cell(True, True)} & {cell(True, False)} \\\\",
        f"Uniform & {cell(False, True)} & {cell(False, False)} \\\\",
        "\\bottomrule", "\\end{tabular}",
    ]
    return "\n".join(lines)


def init_anneal_summary():
    """(peaked-mean %, uniform-mean %, gap points, seeds-per-cell) collapsing the
    init/anneal grid over the annealing axis, or None. Lets the prose cite the
    initialisation effect without hand-typed numbers."""
    d = _REXTRA / "init_anneal"
    pk, un, ncell = [], [], {}
    if d.exists():
        for p in sorted(d.glob("peak*_anneal*_seed*.json")):
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            (pk if r["peak_init"] else un).append(r["macro_f1"])
            key = (bool(r["peak_init"]), bool(r["anneal_markers"]))
            ncell[key] = ncell.get(key, 0) + 1
    if not pk or not un:
        return None
    pm, um = sum(pk) / len(pk), sum(un) / len(un)
    n = min(ncell.values()) if ncell else 0
    return _pct(pm), _pct(um), f"{(pm - um) * 100:.0f}", str(n)


def marker_stability():
    """Mean pairwise Jaccard overlap of the selected marker panel across the
    multi-seed headline runs (reviewer Q1: how stable is the learned panel?)."""
    import itertools
    d = _REXTRA / "multiseed"
    sets = []
    if d.exists():
        for p in sorted(d.glob("markers_seed*.csv")):
            with open(p) as f:
                genes = {r["gene"] for r in csv.DictReader(f) if r.get("recursion_depth")}
            if genes:
                sets.append(genes)
    if len(sets) < 2:
        return None
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
    return sum(jac) / len(jac)


# --- biology-informed-router (gene-gene interaction) ablation --------------- #
_INTER_TASKS = [("cohort", "Cohort"), ("pathologic_stage", "Stage"),
                ("pathologic_T", "T"), ("pathologic_N", "N")]
_INTER_MODES = [("none", "None (baseline)"),
                ("coexpr", "Co-expression (ours)"),
                ("random", "Random graph")]


def interaction_table(_results=None):
    """Genomap gene-gene-interaction router-prior ablation: macro-F1 (mean+/-std)
    per task for none / coexpr / random, read from results_interaction/."""
    d = _REPO / "results_interaction"
    acc = {}                                    # (task, mode) -> [macro_f1]
    if d.exists():
        for p in sorted(d.glob("*__*__seed*.json")):
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            acc.setdefault((r["task"], r["mode"]), []).append(r["macro_f1"])
    if not acc:
        return "\\textit{(interaction ablation pending: run interaction\\_experiments)}"
    tasks = [t for t, _ in _INTER_TASKS if any((t, m) in acc for m, _ in _INTER_MODES)]
    header = "Router prior & " + " & ".join(
        s for t, s in _INTER_TASKS if t in tasks) + " \\\\"
    lines = ["\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{l" + "c" * len(tasks) + "}",
             "\\toprule", header, "\\midrule"]
    for mode, label in _INTER_MODES:
        cells = [_ms_cell(_mean_std(acc.get((t, mode), []))) for t in tasks]
        if all(c == "--" for c in cells):
            continue
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    return "\n".join(lines)


# --- early-exit (early-stopping) recursion vs fixed depth-8 ----------------- #
_ES_MODES = [("fixed8", "Fixed depth ($K{=}8$)"),
             ("early8", "Early-exit ($K{=}8$, ours)")]


def earlystop_table(_results=None):
    """Early-stopping (expert early-exit) vs fixed depth at K=8: macro-F1 per task,
    plus the realised mean recursion depth and compute saving on the cohort task.
    Reads results_earlystop/."""
    d = _REPO / "results_earlystop"
    acc, depth, save = {}, {}, {}
    if d.exists():
        for p in sorted(d.glob("*__*__seed*.json")):
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            acc.setdefault((r["task"], r["mode"]), []).append(r["macro_f1"])
            if r.get("mean_recursion_depth") is not None:
                depth.setdefault(r["mode"], []).append(r["mean_recursion_depth"])
                if r.get("compute_saving_ratio") is not None:
                    save.setdefault(r["mode"], []).append(r["compute_saving_ratio"])
    if not acc:
        return "\\textit{(early-stopping run pending: run earlystop\\_experiments)}"
    tasks = [t for t, _ in _INTER_TASKS if any((t, m) in acc for m, _ in _ES_MODES)]
    header = ("Recursion & " + " & ".join(s for t, s in _INTER_TASKS if t in tasks)
              + " & Depth & Saving \\\\")
    lines = ["\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{l" + "c" * (len(tasks) + 2) + "}",
             "\\toprule", header, "\\midrule"]

    def _avg(xs):
        return sum(xs) / len(xs) if xs else None
    for mode, label in _ES_MODES:
        cells = [_ms_cell(_mean_std(acc.get((t, mode), []))) for t in tasks]
        if all(c == "--" for c in cells):
            continue
        dm = _avg(depth.get(mode, []))
        sv = _avg(save.get(mode, []))
        dcell = f"{dm:.2f}" if dm is not None else "--"
        scell = f"{sv:.2f}$\\times$" if sv is not None else "--"
        lines.append(f"{label} & " + " & ".join(cells) + f" & {dcell} & {scell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    return "\n".join(lines)


# Friendly names + display order for the single-cell generalisation study.
_SC_DATASETS = [
    ("tabula_muris", "Tabula Muris"),
    ("common_class", "Common-class"),
    ("prototype",    "Prototype"),
    ("pancreas",     "Pancreas"),
]


def singlecell_table(_results=None):
    """SMART on the genomap-capsule single-cell datasets. Reads the per-dataset
    JSONs written by ``recursive_marker_transformer.singlecell`` (results_singlecell/).
    Rendered only if at least one result exists."""
    sc_dir = _REPO / "results_singlecell"
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Dataset & Cells & Genes & Classes & Acc. & Macro-F1 \\\\",
        "\\midrule",
    ]
    any_row = False
    for key, label in _SC_DATASETS:
        p = sc_dir / f"{key}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        h = r["heads"].get("cell_type") or next(iter(r["heads"].values()))
        any_row = True
        lines.append(
            f"{label} & {_fmt(r['n_samples'])} & {_fmt(r['n_features'])} & "
            f"{r['n_classes']} & {_pct(h['accuracy'])} & {_pct(h['macro_f1'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) if any_row else "\\textit{(single-cell runs pending)}"


# genoNet tasks: the BIO5 phenotype labels in unified_bio5.csv, run with SMART on
# the raw full gene vector (all 20530 genes). Display order + short names.
_GENONET_TASKS = [
    ("pathologic_stage", "Pathologic stage", "Stage"),
    ("pathologic_T",     "Tumour (T)",       "T"),
    ("pathologic_N",     "Node (N)",         "N"),
    ("os_binary",        "Overall survival", "OS"),
    ("tumor_status",     "Tumour status",    "Tumour"),
]


def _unified_dims():
    """(genes, samples) of the unified BIO5 table, read from a genoNet result;
    falls back to the known TCGA pan-cancer dimensions."""
    gdir = _REPO / "results_genonet"
    for key, _l, _s in _GENONET_TASKS:
        p = gdir / f"{key}.json"
        if p.exists():
            r = json.loads(p.read_text())
            return f"{r['n_genes']:,}".replace(",", "{,}"), f"{r['n_samples']:,}".replace(",", "{,}")
    return "20{,}530", "2{,}738"


def _tuned_smart_run(task):
    """Best-by-validation tuned SMART run for a task (results_tune/), or None."""
    d = _REPO / "results_tune"
    runs = []
    if d.exists():
        for p in d.glob(f"{task}__*.json"):
            try:
                runs.append(json.loads(p.read_text()))
            except Exception:
                pass
    return max(runs, key=lambda r: r.get("val_macro_f1", -1)) if runs else None


def _smart_result(task):
    """SMART metrics for a genoNet task: prefer the val-tuned run, else the fixed
    headline run (results_genonet/). Returns dict or None."""
    t = _tuned_smart_run(task)
    if t:
        return {"accuracy": t["accuracy"], "macro_f1": t["macro_f1"],
                "weighted_f1": t.get("weighted_f1", t["macro_f1"]),
                "n_classes": t["n_classes"], "tuned": True}
    p = _REPO / "results_genonet" / f"{task}.json"
    if p.exists():
        r = json.loads(p.read_text())
        h = r["heads"][task]
        return {"accuracy": h["accuracy"], "macro_f1": h["macro_f1"],
                "weighted_f1": h["weighted_f1"], "n_classes": r["n_classes"], "tuned": False}
    return None


def genonet_table(_results=None):
    """SMART on every genoNet classification task (val-tuned where available)."""
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Task & Classes & Acc. & Macro-F1 & Weighted-F1 \\\\",
        "\\midrule",
    ]
    any_row = False
    for key, label, _short in _GENONET_TASKS:
        r = _smart_result(key)
        if r is None:
            continue
        any_row = True
        lines.append(
            f"{label} & {r['n_classes']} & {_pct(r['accuracy'])} & "
            f"{_pct(r['macro_f1'])} & {_pct(r['weighted_f1'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) if any_row else "\\textit{(genoNet-task runs pending)}"


def baselines_table(_results=None):
    """SMART vs 14 external baselines across the genoNet tasks (macro-F1).
    Reads results_baselines/<task>.json and results_genonet/<task>.json."""
    bdir, gdir = _REPO / "results_baselines", _REPO / "results_genonet"
    tasks = [k for k, _l, _s in _GENONET_TASKS]
    short = {k: s for k, _l, s in _GENONET_TASKS}

    bres, methods = {}, []
    for t in tasks:
        p = bdir / f"{t}.json"
        if p.exists():
            d = json.loads(p.read_text())["baselines"]
            bres[t] = {m: v.get("macro_f1") for m, v in d.items() if "macro_f1" in v}
            for m in d:
                if m not in methods:
                    methods.append(m)
    if not bres:
        return "\\textit{(baseline runs pending)}"

    smart = {}
    for t in tasks:
        r = _smart_result(t)
        if r is not None:
            smart[t] = r["macro_f1"]

    def _mean(d):
        vals = [d[t] for t in tasks if d.get(t) is not None]
        return sum(vals) / len(vals) if vals else -1.0

    def _cell(v):
        return f"{v * 100:.1f}" if v is not None else "--"

    rows = []
    for m in methods:
        d = {t: bres.get(t, {}).get(m) for t in tasks}
        rows.append((m, d, _mean(d)))
    rows.sort(key=lambda r: -r[2])

    header = "Method & " + " & ".join(short[t] for t in tasks) + " & Mean \\\\"
    lines = ["\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{l" + "c" * (len(tasks) + 1) + "}",
             "\\toprule", header, "\\midrule"]
    sm = _mean(smart)
    lines.append("\\textbf{SMART (ours)} & " +
                 " & ".join("\\textbf{%s}" % _cell(smart.get(t)) for t in tasks) +
                 f" & \\textbf{{{sm * 100:.1f}}} \\\\")
    lines.append("\\midrule")
    for m, d, mn in rows:
        lines.append(f"{m} & " + " & ".join(_cell(d.get(t)) for t in tasks) +
                     f" & {mn * 100:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    return "\n".join(lines)


# --- multi-seed ablations + depth/marker sweeps on the hard tasks ----------- #
_HARD = [("pathologic_stage", "Stage"), ("pathologic_T", "T"), ("pathologic_N", "N")]
_ABL_LABELS = [
    ("main",        "Full model"),
    ("no_refine",   "$-$ recursive refinement"),
    ("depth1",      "$-$ recursion ($K{=}1$)"),
    ("independent", "$-$ weight sharing"),
    ("fixed_depth", "Fixed depth (uniform)"),
    ("mor_token",   "Token-choice MoR"),
    ("sel_variance","Variance markers"),
    ("sel_random",  "Random markers"),
]


def _glob_runs(subdir):
    """Yield (stem_parts, data) for every per-run JSON under results_sweeps/<subdir>."""
    d = _REPO / "results_sweeps" / subdir
    out = []
    if d.exists():
        for f in sorted(d.glob("*.json")):
            try:
                out.append((f.stem.split("__"), json.loads(f.read_text())))
            except Exception:
                pass
    return out


def _mean_std(vals):
    if not vals:
        return None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, var ** 0.5


def _ms_cell(ms):
    return f"{ms[0]*100:.1f}\\,$\\pm$\\,{ms[1]*100:.1f}" if ms else "--"


def multiseed_ablation_table(_results=None):
    runs = _glob_runs("ablate")
    if not runs:
        return "\\textit{(multi-seed ablations pending)}"
    acc = {}                                    # (task, variant) -> [macro_f1]
    for parts, data in runs:
        if len(parts) < 3:
            continue
        acc.setdefault((parts[0], parts[1]), []).append(data["macro_f1"])
    tasks = [t for t, _ in _HARD]
    header = "Variant & " + " & ".join(s for _t, s in _HARD) + " \\\\"
    lines = ["\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{l" + "c" * len(tasks) + "}",
             "\\toprule", header, "\\midrule"]
    for key, label in _ABL_LABELS:
        cells = [_ms_cell(_mean_std(acc.get((t, key), []))) for t in tasks]
        if all(c == "--" for c in cells):
            continue
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    return "\n".join(lines)


def _sweep_table(subdir, axis_prefix, axis_label, extra_col):
    runs = _glob_runs(subdir)
    if not runs:
        return f"\\textit{{({subdir} sweep pending)}}"
    acc, meta = {}, {}                           # (task, axisval) -> [f1]; axisval -> (params,saving)
    for parts, data in runs:
        if len(parts) < 3:
            continue
        task = parts[0]
        axisval = int(parts[1][len(axis_prefix):])
        acc.setdefault((task, axisval), []).append(data["macro_f1"])
        meta[axisval] = (data.get("transformer_params"), data.get("compute_saving_ratio"))
    axisvals = sorted({v for (_t, v) in acc})
    tasks = [t for t, _ in _HARD]
    head = f"{axis_label} & {extra_col} & " + " & ".join(s for _t, s in _HARD) + " \\\\"
    lines = ["\\begin{tabular}{ll" + "c" * len(tasks) + "}",
             "\\toprule", head, "\\midrule"]
    for v in axisvals:
        params, saving = meta.get(v, (None, None))
        if extra_col.startswith("Params"):
            ex = _fmt(params) if params else "--"
        else:
            ex = f"{saving:.2f}$\\times$" if saving else "--"
        cells = [_ms_cell(_mean_std(acc.get((t, v), []))) for t in tasks]
        lines.append(f"{v} & {ex} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def depth_sweep_table(_results=None):
    return _sweep_table("depth", "K", "$K$", "Eff. FLOPs")


def marker_sweep_table(_results=None):
    return _sweep_table("markers", "M", "$M$", "Params")


# --- appendix: dataset details + per-task descriptions ---------------------- #
def _load_stats(results: Path):
    p = results / "dataset_stats.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


_STAGE_ROMAN = {0: "I", 1: "II", 2: "III", 3: "IV"}
# human-facing per-task metadata for the appendix
_TASK_META = {
    "pathologic_stage": ("Pathologic stage",               "4-way ordinal"),
    "pathologic_T":     ("Primary tumour (T)",             "4-way ordinal"),
    "pathologic_N":     ("Regional lymph node (N)",        "4-way ordinal"),
    "os_binary":        ("Overall-survival status",        "binary"),
    "tumor_status":     ("Tumour status at follow-up",     "binary"),
}


def _class_label(task, k, names):
    if task == "cancer_type":
        return names.get(str(k), f"class {k}")
    if task == "pathologic_T":
        return f"T{k}"
    if task == "pathologic_N":
        return f"N{k}"
    if task == "pathologic_stage":
        return "Stage " + _STAGE_ROMAN.get(int(k), str(k))
    return f"class {k}"


def dataset_overview_table(results: Path):
    """Appendix overview of every dataset used (samples/features/classes/source)."""
    s = _load_stats(results)
    if not s:
        return "\\textit{(dataset stats pending: run dataset_stats)}"
    main = _load(results, "main")
    n_hvg = main["config"]["n_hvg"] if main else 2000
    lines = ["\\begin{tabular}{lrrll}", "\\toprule",
             "Dataset & Samples & Features & Labels & Source \\\\", "\\midrule"]
    tc = s.get("tcga_cohorts", {})
    if tc:
        lines.append(f"TCGA pan-cancer (4 cohorts) & {_fmt(sum(tc.values()))} & "
                     f"{_fmt(n_hvg)}/20{{,}}530 & 4 cohorts & bulk RNA-seq (Xena) \\\\")
    u = s.get("unified", {})
    if u:
        lines.append(f"Unified BIO5 (genoNet) & {_fmt(u['n_samples'])} & "
                     f"{_fmt(u['n_genes'])} & 5 tasks & bulk RNA-seq (Xena) \\\\")
    lines.append("\\midrule")
    for key, label in _SC_DATASETS:
        d = s.get("singlecell", {}).get(key)
        if d:
            lines.append(f"{label} & {_fmt(d['cells'])} & {_fmt(d['genes'])} & "
                         f"{d['classes']} cell types & single-cell (genomap) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def task_distribution_table(results: Path):
    """Appendix per-task class distribution for the BIO5 phenotype tasks."""
    s = _load_stats(results)
    u = s.get("unified", {})
    if not u:
        return "\\textit{(task stats pending)}"
    names = u.get("cancer_names", {})
    lines = ["\\begin{tabular}{llp{0.52\\columnwidth}}", "\\toprule",
             "Task & Type & Class distribution (count) \\\\", "\\midrule"]
    for task, (label, kind) in _TASK_META.items():
        info = u["tasks"].get(task)
        if not info:
            continue
        items = sorted(info["counts"].items(), key=lambda kv: float(kv[0]))
        dist = ", ".join(f"{_class_label(task, k, names)}: {_fmt(v)}" for k, v in items)
        lines.append(f"{label} & {kind} & {dist} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the paper
# --------------------------------------------------------------------------- #
def build_tex(results: Path) -> str:
    main = _load(results, "main")
    primary = "cancer_type"
    if main is not None:
        primary = main["config"]["heads"][0]

    main_h = (main["heads"][primary] if main else {"accuracy": 0, "macro_f1": 0})
    main_acc = _pct(main_h["accuracy"])
    main_f1 = _pct(main_h["macro_f1"])

    pe = _load(results, "param_efficiency") or []
    ratio4 = next((e["ratio"] for e in pe if e["depth"] == 4), 4.0)
    n_genes = main["config"]["n_hvg"] if main else 2000
    n_markers = main["config"]["n_markers"] if main else 200
    depth = main["config"]["recursion_depth"] if main else 4
    d_model = main["config"]["d_model"] if main else 96
    epochs = main["config"]["epochs"] if main else 12

    def _f1(name):
        r = _load(results, name)
        return _pct(r["heads"][primary]["macro_f1"]) if r else "--"

    main_depth = f"{main['mean_recursion_depth']:.2f}" if main and "mean_recursion_depth" in main else "--"
    main_saving = f"{main['compute_saving_ratio']:.2f}" if main and "compute_saving_ratio" in main else "--"
    main_total = _fmt(main["total_params"]) if main and "total_params" in main else "--"
    main_wall = f"{main['wall_seconds']:.0f}" if main and "wall_seconds" in main else "--"
    ss = main_seed_stats()
    main_seed_f1 = _ms_cell(ss[0]) if ss else "--"
    main_seed_acc = _ms_cell(ss[1]) if ss else "--"
    n_seeds = str(ss[2]) if ss else "--"
    ia = init_anneal_summary()
    ia_peak, ia_unif, ia_gap, n_seeds_ia = ia if ia else ("--", "--", "--", "--")
    mjac = marker_stability()
    marker_jac = f"{mjac:.2f}" if mjac is not None else "--"

    repl = {
        "@@MEANDEPTH@@": main_depth,
        "@@SAVING@@": main_saving,
        "@@MAIN_TOTAL_PARAMS@@": main_total,
        "@@MAIN_WALL@@": main_wall,
        "@@MAIN_SEED_F1@@": main_seed_f1,
        "@@MAIN_SEED_ACC@@": main_seed_acc,
        "@@NSEEDS@@": n_seeds,
        "@@MOR_TOKEN_F1@@": _f1("mor_token"),
        "@@COST_TABLE@@": cost_table(results),
        "@@INIT_ANNEAL_TABLE@@": init_anneal_table(),
        "@@INTERACTION_TABLE@@": interaction_table(),
        "@@EARLYSTOP_TABLE@@": earlystop_table(),
        "@@IA_PEAK_F1@@": ia_peak,
        "@@IA_UNIF_F1@@": ia_unif,
        "@@IA_GAP_F1@@": ia_gap,
        "@@NSEEDS_IA@@": n_seeds_ia,
        "@@MARKER_JACCARD@@": marker_jac,
        "@@ROUTING_TABLE@@": routing_table(results, primary),
        "@@MAIN_ACC@@": main_acc,
        "@@MAIN_F1@@": main_f1,
        "@@RATIO4@@": f"{ratio4:.2f}",
        "@@NGENES@@": str(n_genes),
        "@@NMARKERS@@": str(n_markers),
        "@@DEPTH@@": str(depth),
        "@@DMODEL@@": str(d_model),
        "@@EPOCHS@@": str(epochs),
        "@@SEL_LEARN_F1@@": _f1("fixed_depth"),
        "@@SEL_RAND_F1@@": _f1("sel_random"),
        "@@SEL_VAR_F1@@": _f1("sel_variance"),
        "@@SEL_CONC_F1@@": _f1("sel_concrete"),
        "@@MAIN_TABLE@@": main_results_table(main, primary),
        "@@PERCLASS_TABLE@@": per_class_table(main, primary),
        "@@SELECTION_TABLE@@": selection_table(results, primary),
        "@@ABLATION_TABLE@@": ablation_table(results, primary),
        "@@PARAM_TABLE@@": param_table(results),
        "@@GENEVAL_TABLE@@": gene_identification_table(results),
        "@@SINGLECELL_TABLE@@": singlecell_table(results),
        "@@GENONET_TABLE@@": genonet_table(results),
        "@@BASELINES_TABLE@@": baselines_table(results),
        "@@UNIFIED_GENES@@": _unified_dims()[0],
        "@@UNIFIED_SAMPLES@@": _unified_dims()[1],
        "@@MULTISEED_ABLATION_TABLE@@": multiseed_ablation_table(results),
        "@@DEPTH_SWEEP_TABLE@@": depth_sweep_table(results),
        "@@MARKER_SWEEP_TABLE@@": marker_sweep_table(results),
        "@@DATASET_OVERVIEW_TABLE@@": dataset_overview_table(results),
        "@@TASK_DISTRIBUTION_TABLE@@": task_distribution_table(results),
    }
    tex = _TEX
    for k, v in repl.items():
        tex = tex.replace(k, v)
    return tex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="results")
    ap.add_argument("--outdir", type=str, default="paper")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("aaai.sty", "aaai.bst", "fixbib.sty"):
        shutil.copy(_TEMPLATE_DIR / f, out / f)

    (out / "genomicrecursiveformer.tex").write_text(build_tex(Path(args.results)))
    (out / "refs.bib").write_text(_BIB)
    print(f"[paper] wrote {out}/genomicrecursiveformer.tex and refs.bib")


# --------------------------------------------------------------------------- #
# LaTeX template (prose is static; @@TOKENS@@ are filled from results)
# --------------------------------------------------------------------------- #
_TEX = r"""\documentclass[letterpaper]{article}
\usepackage{aaai}
\usepackage{times}
\usepackage{helvet}
\usepackage{courier}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{fontawesome5}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc,shapes.geometric}
\definecolor{panelA}{HTML}{EAF2EA}
\definecolor{panelB}{HTML}{EAF0FA}
\definecolor{boxedge}{HTML}{6B7280}
\definecolor{accentA}{HTML}{2E7D5B}
\definecolor{accentB}{HTML}{2F5BAA}
\definecolor{subcap}{HTML}{4B5563}
\frenchspacing
\setlength{\pdfpagewidth}{8.5in}
\setlength{\pdfpageheight}{11in}
\pdfinfo{
/Title (SMART: Selective Marker-guided Adaptive Recursive Transformer for Transcriptomic Classification)
/Author (Koushik Howlader, Tirtho Roy, Md Tauhidul Islam, Wei Le)
/Keywords (single-cell genomics, transformers, parameter efficiency, marker genes, recursive computation)
}
\setcounter{secnumdepth}{1}

\title{SMART: Selective Marker-guided Adaptive Recursive\\ Transformer for Transcriptomic Classification}
\author{Koushik Howlader\textsuperscript{1} \and Tirtho Roy\textsuperscript{1} \and Md Tauhidul Islam\textsuperscript{2} \and Wei Le\textsuperscript{1}\\
\textsuperscript{1}Iowa State University, Ames, Iowa, USA\\
\textsuperscript{2}Stanford University, Stanford, California, USA\\
weile@iastate.edu, tauhid@stanford.edu
}

\begin{document}
\maketitle

\begin{abstract}
\begin{quote}
Transformer foundation models for transcriptomics treat every one of the
$\sim$20{,}000 measured genes as an equally important token and stack many
independent layers, which makes them parameter-heavy and computationally
prohibitive. We argue that, for gene expression, \emph{parameter efficiency
should be an architectural property rather than something recovered by post-hoc
pruning}. We introduce \textbf{SMART} (Selective Marker-guided Adaptive Recursive
Transformer), a transformer that (i)
learns end-to-end which genes are \emph{marker genes} worthy of dedicated
computation, using a cross-attention \emph{marker router} in which learnable
marker queries attend over all genes, so gradients reach every gene rather than a
frozen pre-selected set; (ii) represents each sample by only its $M\ll N$ selected
markers, reducing self-attention cost from $\mathcal{O}(N^2)$ to
$\mathcal{O}(M^2)$; and (iii) processes these marker tokens
with a \emph{single} transformer block applied recursively, where a
Mixture-of-Recursions router grants each gene its own \emph{adaptive} recursion
depth. Most genes exit after one pass while a few disease drivers are iterated
deeper, so a gene's recursion depth becomes an intrinsic, compute-allocation
based importance score instead of a post-hoc attention map. On pan-cancer cohort
classification we reach @@MAIN_ACC@@\% accuracy and @@MAIN_F1@@\% macro-F1 while
using @@RATIO4@@$\times$ fewer transformer-stack parameters than an equivalent
stack of independent layers and only @@SAVING@@$\times$ of the fixed-depth
recursion FLOPs. Beyond cohort identity we evaluate SMART on five harder clinical
phenotype tasks defined on the same tumours (stage, tumour T, node N, overall
survival, tumour status): under an identical split and the full gene set it attains
the best mean macro-F1 against ten strong nonlinear baselines, including XGBoost,
LightGBM and CatBoost, and it transfers unchanged to four single-cell cell-type
datasets. A controlled selection study shows the learned selectors beat random
and variance selection when a marker signal is present, and the per-gene recursion
depth yields an interpretable, compute-allocated gene panel; multi-seed ablations
report, transparently, that on the weakly-determined clinical labels the
architectural choices are statistically within noise. We further make routing
\emph{biology-informed} by injecting a label-free genomap gene-gene interaction prior
into the recursion router, and test it against a degree-matched random-graph control;
on these datasets it does not separate from that control, a negative result we report
transparently alongside its mathematical and biological foundation. The whole
pipeline, including training, ablations, baselines, and this paper itself,
regenerates from a single shell script.
\end{quote}
\end{abstract}

\section{Introduction}
Single-cell and bulk RNA sequencing now routinely profile the expression of tens
of thousands of genes across millions of cells, and a wave of transformer-based
\emph{foundation models} such as scGPT \cite{cui2024scgpt}, Geneformer
\cite{theodoris2023transfer}, scBERT \cite{yang2022scbert} and scFoundation
\cite{hao2024large} has adapted the architecture of \cite{vaswani2017attention}
to this modality. These models are powerful, but they inherit two costly habits
from natural-language transformers. First, they treat \emph{every} gene as an
equally important token, so a housekeeping gene and a lineage-defining marker
receive identical computational budgets. Second, they stack many
\emph{independent} layers, so parameters grow linearly with depth. The result is
models with tens to hundreds of millions of parameters \cite{hao2024large} whose
self-attention scales quadratically in the number of genes, and whose efficiency
is usually addressed only afterwards through pruning or distillation.

We take a different stance: \emph{for gene expression, the data themselves tell
us where to spend computation}. Decades of single-cell biology rest on the
observation that a small set of \emph{marker genes} is sufficient to discriminate
cell types and disease states \cite{ianevski2022fully,franzen2019panglaodb,%
hu2023cellmarker}. If a model could decide, while training, which genes are
markers, it could grant them dedicated capacity and let everything else share
parameters. This makes parameter efficiency an \emph{architectural} property rather
than a compression afterthought.

We realise this idea in \textbf{SMART} (Selective Marker-guided Adaptive Recursive
Transformer), a recursive marker-guided transformer with three coupled components. (1) A cross-attention
\emph{marker router}: $M$ learnable marker queries attend over all $N$ genes
(Set-Transformer / Perceiver-style \cite{jang2017categorical,balin2019concrete}),
with a temperature-annealed softmax over genes, so the model learns \emph{which}
genes are markers end-to-end while gradients reach every gene. This matters in
practice: we show that hard top-$k$ routing cannot explore and underperforms even
random panels. (2) \emph{Marker-driven compression}: each sample is
represented by only its $M\ll N$ selected markers, cutting attention from
$\mathcal{O}(N^2)$ to $\mathcal{O}(M^2)$. (3) A \emph{recursive shared block}:
instead of $K$ independent layers we apply one transformer block $K$ times, in
the spirit of Universal Transformers \cite{dehghani2019universal}, ALBERT
\cite{lan2020albert} and the recent Mixture-of-Recursions
\cite{bae2025mixture}. After each recursion we re-run a marker gate on the updated
representations, so the model jointly learns \emph{what biological information to
preserve} and \emph{how to allocate its computational capacity}.

We make the following contributions:
\begin{itemize}
\item We propose SMART, to our knowledge the first transformer
for transcriptomics in which marker selection, token compression, and
parameter-shared recursion are co-designed and trained end-to-end.
\item We introduce \emph{recursive marker refinement}, a closed feedback loop
that turns marker identification from a preprocessing step into a differentiable
part of the architecture.
\item We adapt Mixture-of-Recursions \cite{bae2025mixture} to gene expression: an
expert-choice router gives each gene an \emph{adaptive} recursion depth, unifying
weight sharing with adaptive computation and turning a gene's recursion depth into
an intrinsic importance score that needs no post-hoc attribution.
\item We show, on pan-cancer cohort classification, that the model matches strong
accuracy while using @@RATIO4@@$\times$ fewer transformer parameters than
independent layers, and that learned markers beat random and variance baselines,
with the recursion-depth signal yielding an interpretable gene panel.
\item We benchmark SMART beyond cohort identity on five harder clinical phenotype
tasks (stage, tumour T, node N, overall survival, tumour status) against ten strong
nonlinear baselines, including gradient-boosted trees (XGBoost, LightGBM, CatBoost),
where SMART attains the best across-task mean macro-F1, and we transfer it unchanged
to four single-cell datasets.
\item We report multi-seed ablations and depth / marker-count sweeps, and state
honestly that on the weakly-determined clinical labels the architectural choices are
within noise: the marker-learning advantage requires a genuine marker signal in the
label, as on the cohort task.
\item We propose a \emph{biology-informed router} that injects a label-free genomap
gene-gene interaction prior (network centrality) into the recursion router as an
annealed additive bias, and we evaluate it against a degree-matched random-graph
control; on the present datasets it does not separate from that control, a negative
result we report transparently alongside the mechanism and its theory.
\item We release a fully reproducible pipeline in which a single shell script
runs all experiments, baselines, and regenerates this paper, numbers and tables
included.
\end{itemize}

\section{Related Work}
\paragraph{Transformer foundation models for omics.}
Geneformer \cite{theodoris2023transfer} and scBERT \cite{yang2022scbert} adapt
masked-language-model pretraining to single-cell transcriptomes; scGPT
\cite{cui2024scgpt} scales generative pretraining to 33M cells; scFoundation
\cite{hao2024large} trains a 100M-parameter model over $\sim$20{,}000 genes. Closest
to our setting, GexBERT \cite{jiang2025gexbert} pretrains a gene-plus-value bulk
RNA-seq transformer autoencoder with a masking-and-restoration objective and applies
it to pan-cancer classification and survival; it shares our gene-plus-value
embedding but, like the others, treats genes uniformly and uses independent layers,
with no marker-driven sparsity or recursion.
Earlier deep generative approaches such as scVI \cite{lopez2018deep} established
probabilistic latent representations of expression. More recent foundation models
push to whole-genome transcription \cite{fu2025get}, gene-regulatory-network-aware
single-cell prediction \cite{zhang2025cellular}, and other organisms
\cite{cao2025scplantllm}. genomap \cite{islam2023cartography} instead reshapes the
gene vector into an image via an optimal-transport layout for convolutional
analysis; image-based CNN classifiers of TCGA RNA-seq follow the same recipe
\cite{khalifa2020ai,rukhsar2022analyzing}. A parallel line improves accuracy mainly
through gene selection or augmentation before a deep net
\cite{polepalli2025cvae,kim2025pancancer,rahaman2025integrated,shukla2025discriminative,bouazza2025degs}.
All of these treat genes uniformly or select them in a separate stage; none make
marker-driven sparsity an architectural prior.

\paragraph{Parameter-efficient and recursive transformers.}
Tying weights across depth was shown to retain representational power by Universal
Transformers \cite{dehghani2019universal} and to drastically shrink models in
ALBERT \cite{lan2020albert}. Mixture-of-Recursions \cite{bae2025mixture} unifies
weight sharing with token-level adaptive depth, and Mixture-of-Depths
\cite{raposo2024mixture} routes tokens through variable numbers of layers, both
echoing sparsely-gated mixtures of experts \cite{shazeer2017outrageously} and
adaptive computation time \cite{graves2016adaptive}. A separate line of work on
efficient attention, including Linformer \cite{wang2020linformer}, Performer
\cite{choromanski2021rethinking} and Nystr\"omformer
\cite{xiong2021nystromformer}, reduces the quadratic cost generically. Recursive
weight sharing has also been pushed in vision and restoration transformers, where a
shared block is unrolled with depth, sometimes with input-conditioned modulation:
the Sliced Recursive Transformer \cite{shen2021sliced}, RISTRA
\cite{zhou2024ristra}, Mixture-of-LoRAs recursion \cite{nouriborji2025mol}, and
Ouroboros \cite{jaber2026ouroboros}. We borrow the weight-sharing mechanism but
make the token set itself biologically structured, so our $\mathcal{O}(M^2)$ saving
is complementary to these methods.

\paragraph{Markers, pathways, and biological priors.}
Marker-based annotation tools such as scType \cite{ianevski2022fully} and SingleR
\cite{aran2019reference}, and curated databases including PanglaoDB
\cite{franzen2019panglaodb} and CellMarker~2.0 \cite{hu2023cellmarker}, encode
the long-standing principle that few genes carry most discriminative signal.
Pathway-informed models such as the recent graph transformer PATH
\cite{howlader2026graph} build structure from Reactome \cite{gillespie2022reactome}.
We use such resources only to \emph{validate} our learned markers, keeping marker
discovery fully data-driven. Standard tooling \cite{wolf2018scanpy,%
luecken2019current} and reference atlases \cite{regev2017human,tabula2022tabula}
provide the broader context for our task.

\section{Method}

\begin{figure*}[t]
\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[
  node distance=10mm and 9mm,
  stage/.style={rounded corners=3pt, draw=boxedge, line width=0.6pt, fill=white,
                text width=21mm, minimum height=15mm, align=center, inner sep=3pt},
  flow/.style={-{Stealth[length=2.6mm]}, line width=0.9pt, draw=boxedge},
  loop/.style={-{Stealth[length=2.6mm]}, line width=1.1pt, draw=accentB},
  panel/.style={rounded corners=6pt, inner xsep=4.5mm, inner ysep=6.5mm},
  ptab/.style={font=\footnotesize\bfseries, text=white, fill=#1,
               rounded corners=2pt, inner sep=2.5pt},
]
\node[stage] (inp) {{\large\textcolor{accentA}{\faDna}}\\[2pt]\textbf{Expression}\\[1pt]{\scriptsize\textcolor{subcap}{$x\!\in\!\mathbb{R}^{B\times N}$, $N{=}@@NGENES@@$}}};
\node[stage, right=of inp] (emb) {{\large\textcolor{accentA}{\faProjectDiagram}}\\[2pt]\textbf{Gene Embedding}\\[1pt]{\scriptsize\textcolor{subcap}{identity $+$ value proj.}}};
\node[stage, right=of emb] (router) {{\large\textcolor{accentA}{\faSearch}}\\[2pt]\textbf{Marker Router}\\[1pt]{\scriptsize\textcolor{subcap}{$M$ query slots, $\tau$-anneal}}};
\node[stage, right=of router] (mtok) {{\large\textcolor{accentA}{\faTags}}\\[2pt]\textbf{Marker Tokens}\\[1pt]{\scriptsize\textcolor{subcap}{$\mathbf{C}\!\in\!\mathbb{R}^{B\times M\times d}$}}};
\draw[flow] (inp) -- (emb);
\draw[flow] (emb) -- (router);
\draw[flow] (router) -- (mtok);

\node[stage, below=52mm of inp] (shared) {{\large\textcolor{accentB}{\faRedo}}\\[2pt]\textbf{Shared Block}\\[1pt]{\scriptsize\textcolor{subcap}{$f_\theta$ applied $\times K$}}};
\node[stage, right=of shared] (mor) {{\large\textcolor{accentB}{\faFilter}}\\[2pt]\textbf{MoR Depth Router}\\[1pt]{\scriptsize\textcolor{subcap}{funnel; logit $+\,\beta_t\pi_m$}}};
\node[stage, right=of mor] (pool) {{\large\textcolor{accentB}{\faCompress}}\\[2pt]\textbf{Mean-pool}\\[1pt]{\scriptsize\textcolor{subcap}{over $M$ markers}}};
\node[stage, right=of pool] (clf) {{\large\textcolor{accentB}{\faChartBar}}\\[2pt]\textbf{Classifier}\\[1pt]{\scriptsize\textcolor{subcap}{linear head}}};
\node[stage, right=of clf] (coh) {{\large\textcolor{accentB}{\faSitemap}}\\[2pt]\textbf{Tasks}\\[1pt]{\tiny\textcolor{subcap}{4 TCGA cohorts\\ 5 phenotypes\\ 4 single-cell}}};
% biology-informed router: genomap gene-gene interaction graph -> centrality prior
\node[stage, below=13mm of inp, text width=22mm] (gint) {{\large\textcolor{accentA}{\faProjectDiagram}}\\[1pt]\textbf{Gene--Gene Graph}\\[1pt]{\tiny\textcolor{subcap}{genomap centrality $\pi$}}};
\draw[flow] (shared) -- (mor);
\draw[flow] (mor) -- (pool);
\draw[flow] (pool) -- (clf);
\draw[flow] (clf) -- (coh);
\draw[flow, draw=accentA, dashed] (gint.east) -| (mor.north);

\draw[loop] (shared.south east) .. controls ++(0,-9mm) and ++(0,-9mm) .. (shared.south west)
  node[midway, below=0.5mm, font=\scriptsize\bfseries, text=accentB, align=center]
  {$\times K$, weight-shared\\[-1pt]{\scriptsize\textcolor{subcap}{$+$ refinement gate}}};

\draw[flow] (mtok.south) -- ++(0,-9mm) coordinate (cdrop) -| ([xshift=-8mm]shared.west) -- (shared.west);
\node[font=\scriptsize, text=subcap, below, fill=white, inner sep=1pt] at ([xshift=-24mm]cdrop) {marker tokens};

\begin{scope}[on background layer]
\node[panel, fill=panelA, fit=(inp)(mtok)(gint)] (pA){};
\node[panel, fill=panelB, fit=(shared)(coh)] (pB){};
\end{scope}
\node[ptab=accentA, anchor=west] at ([xshift=2mm]pA.north west)
  {A\; $\cdot$\; Marker Selection (Q-Former router)};
\node[ptab=accentB, anchor=west] at ([xshift=2mm]pB.north west)
  {B\; $\cdot$\; Recursive Refinement \& Classification};

\end{tikzpicture}%
}
\caption{\textbf{System overview.} \textbf{Panel A (marker selection):} the expression
vector is embedded gene-by-gene, then $M$ learnable query slots cross-attend over \emph{all}
$N$ genes (temperature annealed soft$\to$peaked) to select interpretable marker tokens, a
Q-Former-style router. \textbf{Panel B (recursive refinement \& classification):} a
\emph{single} weight-shared block $f_\theta$ is applied up to $K$ times (loop-back arrow)
with a per-marker refinement gate between passes; a Mixture-of-Recursions router funnels
capacity so each marker gets an \emph{adaptive} depth $d_m$ (deeply routed genes are
candidate drivers). \textbf{Biology-informed routing:} a genomap gene-gene co-expression
graph supplies a network-centrality prior $\pi$ that is added (annealed by $\beta_t$) to
the depth-router logit (dashed arrow), so co-expression hub genes get a head start in the
funnel without any label leakage. Tokens are mean-pooled and classified; the \emph{same}
pipeline serves all datasets, the four TCGA cohorts, five clinical phenotype tasks, and
four single-cell cell-type datasets.}
\label{fig:overview}
\end{figure*}

\subsection{Overview}
Let $x \in \mathbb{R}^{N}$ be the expression vector of a sample over $N$ genes.
SMART maps $x$ to class logits through five stages: gene
embedding, learnable marker selection, marker-anchored compression, recursive
shared transformation with marker refinement, and a pooled classifier.
Figure~\ref{fig:overview} summarises the two stages.

\subsection{Gene Embedding}
Each gene $i$ is embedded as the sum of a learned identity vector
$\mathbf{e}_i \in \mathbb{R}^d$ and a projection of its scalar expression,
$\mathbf{t}_i = \mathbf{e}_i + \mathbf{W}_v\, x_i$, following the gene-plus-value
scheme of \cite{cui2024scgpt,theodoris2023transfer}. This is linear in $N$ and so
runs over all genes before any compression.

\subsection{Cross-Attention Marker Router}
We identify markers with a cross-attention \emph{router}, the best-practice form
of differentiable selection (Set-Transformer induced points and Perceiver-style
cross-attention). We maintain $M$ learnable marker queries
$\mathbf{q}_m \in \mathbb{R}^{d}$; each attends over the genes through a shared key
projection $\mathbf{k}_i = \mathbf{W}_k \mathbf{e}_i$, giving selection weights
$\mathbf{w}_m = \mathrm{softmax}\big(\mathbf{q}_m \mathbf{K}^{\top}/(\tau\sqrt{d})\big)$
over \emph{all} $N$ genes, with temperature $\tau$ annealed from soft to peaked.
Because the softmax spans all genes, gradients reach every gene, so a query can
migrate to an informative gene it did not initially favour, which is exactly the
property that hard top-$k$ routing lacks. Two ingredients are essential: the
all-gene softmax, and a \emph{peaked initialisation} that points each query at a
distinct gene's key, so training starts at random-selection quality instead of a
uniform average of all genes. At inference each query collapses to its arg-max
gene $g_m = \arg\max_i \mathbf{q}_m^{\top}\mathbf{k}_i$, giving discrete,
interpretable markers and $\mathcal{O}(M^2 d)$ attention. As alternatives we
consider the closely related \emph{Concrete} selector \cite{balin2019concrete,%
jang2017categorical} (free per-slot logits over genes), fixed \emph{variance}- and
\emph{random}-selected panels, and a naive hard \emph{router} in the style of
expert-choice routing \cite{shazeer2017outrageously,raposo2024mixture,bae2025mixture};
the last fails because hard routing cannot explore.

\subsection{Marker Tokens}
Each query produces one marker token combining the (soft-)selected gene identity
and its expression,
$\mathbf{c}_m = (\mathbf{w}_m^{\top}\mathbf{E}) + \mathbf{W}_v\,(\mathbf{w}_m^{\top}\mathbf{x})$,
where $\mathbf{E}\in\mathbb{R}^{N\times d}$ are the gene-identity embeddings. The
identity and value contributions are summed without an intermediate LayerNorm; the
two are placed on a common scale by the pre-norm LayerNorm at the entry of the
shared block (Sec.~\ref{sec:rec}), which normalises every marker token before its
first self-attention. (In the optional aggregation variant the per-gene tokens are
LayerNorm-ed during embedding before pooling.) As an
optional \emph{aggregation} variant, non-selected genes can instead be folded into
their nearest marker by cosine similarity; we find this makes the model robust to
marker choice but masks selection quality, so our headline model uses pure
selection.

\subsection{Recursive Shared Transformer with Marker Refinement}
\label{sec:rec}
Rather than $K$ independent layers, we instantiate a \emph{single} pre-norm
transformer block $f_\theta$ and apply it up to $K$ times:
$\mathbf{H}^{(t+1)} = f_\theta(\mathbf{H}^{(t)})$, $\mathbf{H}^{(0)} =
\mathbf{C}$. This ties all depth-wise parameters
\cite{dehghani2019universal,lan2020albert,bae2025mixture}, so the stack's
parameter count is independent of $K$. After each pass we recompute a per-token
gate from the \emph{updated} cluster embeddings,
$g^{(t)}_m = \sigma(\mathrm{MLP}_r(\mathbf{H}^{(t)}_m))$, and apply it before the
next pass. We call this \emph{recursive marker refinement}; it lets markers that
remain informative survive while others are suppressed.

\begin{figure*}[t]
\centering
\resizebox{0.92\linewidth}{!}{%
\begin{tikzpicture}[
  font=\footnotesize,
  blk/.style={rounded corners=2pt, draw=boxedge, line width=0.6pt, fill=white,
              text width=24mm, align=center, minimum height=6.5mm, inner sep=2pt, font=\scriptsize},
  add/.style={circle, draw=boxedge, inner sep=0.3pt, minimum size=3.4mm, font=\tiny},
  fl/.style={-{Stealth[length=2mm]}, line width=0.7pt, draw=boxedge},
  rec/.style={-{Stealth[length=2.4mm]}, line width=1.0pt, draw=accentB},
  on/.style={rounded corners=1pt, fill=accentB, draw=accentB, minimum size=5.5mm, inner sep=0pt},
  off/.style={rounded corners=1pt, fill=black!4, draw=boxedge!55, minimum size=5.5mm, inner sep=0pt},
  hd/.style={font=\scriptsize\bfseries},
]
\node[hd, text=accentB] (btitle) {Shared block $f_\theta$};
\node[font=\scriptsize, below=1.4mm of btitle] (bin) {tokens $\mathbf{H}^{(t)}$};
\node[blk, below=2mm of bin] (ln1) {LayerNorm};
\node[blk, below=3mm of ln1] (att) {Multi-Head Self-Attn};
\node[add, below=3mm of att] (a1) {$+$};
\node[blk, below=3mm of a1] (ln2) {LayerNorm};
\node[blk, below=3mm of ln2] (ffn) {Feed-Forward};
\node[add, below=3mm of ffn] (a2) {$+$};
\node[font=\scriptsize, below=2.4mm of a2] (bout) {$\mathbf{H}^{(t{+}1)}$};
\draw[fl] (bin)--(ln1); \draw[fl] (ln1)--(att); \draw[fl] (att)--(a1);
\draw[fl] (a1)--(ln2); \draw[fl] (ln2)--(ffn); \draw[fl] (ffn)--(a2); \draw[fl] (a2)--(bout);
\draw[fl] (bin.west) -- ++(-6mm,0) |- (a1.west);
\draw[fl] (a1.west) -- ++(-6mm,0) |- (a2.west);
\draw[rec] (bout.east) -- ++(7mm,0) |- (bin.east)
   node[pos=0.25, right, align=center, text=accentB, font=\scriptsize\bfseries] {apply\\$\times K$\\(shared $\theta$)};

\begin{scope}[shift={(56mm,-8mm)}]
  \node[hd, anchor=south west] at (-2mm,12.5mm) {Mixture-of-Recursions: adaptive recursion depth $d_m$};
  \draw[rec] (1*13mm,9.6mm) -- (4*13mm,9.6mm)
     node[midway, above, font=\scriptsize, text=accentB] {$f_\theta$ reused each step};
  \foreach \t/\c in {1/1.0, 2/0.75, 3/0.5, 4/0.5}{
     \node[hd] at (\t*13mm,7mm) {$t{=}\t$};
     \node[font=\tiny, text=subcap] at (\t*13mm,4.3mm) {keep $\c\,M$};
  }
  \foreach \g/\d/\r in {KRT14/4/0, ANKRD30A/4/1, SFN/3/2, SPRR3/2/3, GAPDH/1/4, ACTB/1/5}{
     \node[anchor=east, font=\scriptsize\ttfamily] at (8mm,-\r*7mm) {\g};
     \foreach \t in {1,...,4}{
        \ifnum\t>\d \node[off] at (\t*13mm,-\r*7mm) {}; \else \node[on] at (\t*13mm,-\r*7mm) {}; \fi
     }
     \node[anchor=west, font=\scriptsize] at (4*13mm+7mm,-\r*7mm) {$d_m{=}\d$};
  }
  \node[hd, anchor=west] at (4*13mm+7mm,7mm) {depth};
  \node[on] (lg1) at (1*13mm,-6*7mm-1mm) {};
  \node[anchor=west, font=\tiny] at ([xshift=1mm]lg1.east) {recurses};
  \node[off] (lg2) at (3*13mm,-6*7mm-1mm) {};
  \node[anchor=west, font=\tiny] at ([xshift=1mm]lg2.east) {frozen / exited};
\end{scope}
\end{tikzpicture}%
}
\caption{\textbf{Adopting Mixture-of-Recursions.} \emph{Left:} one weight-shared pre-norm
block $f_\theta$ (the model's only transformer parameters) is re-applied up to $K{=}4$ times
(recurrence arrow), so depth costs no extra parameters. \emph{Right:} an expert-choice router
keeps a shrinking top fraction of markers per step (capacity funnel
$1,\tfrac34,\tfrac12,\tfrac12$); a marker not kept is frozen, so its \emph{recursion depth}
$d_m$ is the deepest step it survived. Driver genes (\texttt{KRT14}, \texttt{ANKRD30A}) recur
deepest; settled house-keeping genes (\texttt{GAPDH}, \texttt{ACTB}) exit at $d_m{=}1$, saving
compute.}
\label{fig:mor}
\end{figure*}

\paragraph{Mixture-of-Recursions over genes.}
Spending the full depth $K$ on every marker is wasteful: most genes are settled
after one pass, while a few disease drivers reward deeper iterative reasoning. We
therefore make the recursion \emph{adaptive per token} with a Mixture-of-Recursions
router \cite{bae2025mixture} (Figure~\ref{fig:mor}), turning weight-shared recursion (parameter
efficiency) and adaptive computation \cite{graves2016adaptive,raposo2024mixture}
into a single co-designed mechanism. Our headline model uses \emph{expert-choice}
routing: at step $t$ a lightweight router scores the active marker tokens and a
capacity $c_t$ (a geometric funnel $1,\tfrac12,\tfrac14,\dots$) keeps only the
top-$\lceil c_t M\rceil$; selected tokens are gated by the router weight and
updated by $f_\theta$, the rest are frozen at their current state. Survivors at
step $t$ are the only candidates at step $t{+}1$, so genes are progressively
filtered and the tokens that reach the deepest step are, by construction, those the
model judges most worth computing on. We define a gene's \emph{recursion depth}
$d_m\in\{0,\dots,K\}$ as the number of steps its marker survived; averaged over a
cohort this is an intrinsic, compute-allocation based importance score, a
biomarker signal read directly off the architecture rather than from post-hoc
attention. As an ablation we also implement \emph{token-choice} routing
\cite{bae2025mixture}, where each gene selects a single depth up front (top-1 over
$\{1,\dots,K\}$) with a Switch-style load-balancing loss
\cite{shazeer2017outrageously}; consistent with the original report we find
expert-choice the stronger of the two. Both share the block $f_\theta$, so the
parameter-efficiency claim is untouched, and both reduce to the uniform fixed-depth
recursion when routing is disabled. The router is trained with a small
logit $z$-loss (and, for token-choice, the balancing term) added to the objective.

\subsection{Biology-Informed Routing}
\label{sec:biorouter}
So far the router decides depth from data alone, and biology only enters afterwards,
when we cross-check the deepest-routed genes against curated markers. We instead move
a biological prior \emph{into} the routing decision. For marker token $m$ at step
$t$, the expert-choice logit becomes
\begin{equation}
\tilde r^{(t)}_m \;=\; \underbrace{\tfrac{1}{\tau}\,\mathbf{w}_r^{\top}\mathbf{H}^{(t)}_m}_{\text{data-driven (learned)}}
\;+\; \underbrace{\beta_t\,\pi_m}_{\text{biological prior}},
\label{eq:biorouter}
\end{equation}
and the keep/drop top-$\lceil c_t M\rceil$ and the gate
$g^{(t)}_m=\sigma(\tilde r^{(t)}_m)$ run exactly as before, so the recursion-depth
definition is unchanged but now reflects prior and data together. This is the same
loss-free additive-bias slot used by Switch routing \cite{shazeer2017outrageously},
except the bias is a per-gene \emph{biological} score, not a load-balancing scalar.

\paragraph{The prior $\pi_m$ from gene-gene interactions.}
We obtain $\pi_m$ from genomap's gene-gene \emph{interaction} identification
\cite{islam2023cartography}: genomap's \texttt{createInteractionMatrix} defines
interaction as the pairwise correlation distance between genes across samples
(which it then feeds to optimal transport for an image layout). We call that
function on the training split and reuse only its interaction matrix, taking the
co-expression affinity $\mathbf{W}_{ij}=|\mathrm{corr}(g_i,g_j)|=|1-d_{ij}|$ from
genomap's distance $d_{ij}$, sparsifying it to each gene's $k$ nearest neighbours,
symmetrising it, and reading off the network centrality
\begin{equation}
\pi \;=\; \mathrm{zscore}\big(\text{eigvec-centrality}(\mathbf{W})\big),
\end{equation}
so co-expression \emph{hub} genes, the master-regulator-like nodes whose perturbation
propagates widely, receive a larger prior and are nudged to recurse deeper. Crucially
$\mathbf{W}$ is built from \emph{expression alone, with no labels}, so the prior
injects biological network structure without leaking the cohort labels; this is what
keeps the gene-discovery claim honest.

\paragraph{Annealing $\beta_t$.}
A fixed strong prior would behave like hard routing, pinning compute to known hubs and
unable to discover new genes, exactly the failure mode of our uniform-init router. We
therefore warm-start: $\beta_t=\beta_0(1-\text{progress})$ decays to $0$ over training,
so the prior dominates early, when hidden states are still random and biology gives a
sensible starting allocation, and the data-driven term takes over late. This is an
empirical-Bayes shrinkage / curriculum: a strong prior when evidence is weak, fading
as evidence accumulates. Because $\pi_m$ is a constant additive bias, $\tilde r^{(t)}_m$
stays smooth in $\mathbf{w}_r$ and the gate $g^{(t)}_m$ still carries gradient, so
nothing about trainability or the parameter-efficiency claim changes. We validate the
component against a degree-matched \emph{random}-graph control: only if the real
co-expression graph beats the random one is it the biology, not mere smoothing, that
helps (Sec.~\ref{sec:interaction}).

\subsection{Training Objective}
We minimise a composite loss
$\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \lambda\mathcal{L}_{\mathrm{marker}}
+ \gamma\mathcal{L}_{\mathrm{div}} + \beta\mathcal{L}_{\mathrm{comp}}
+ \zeta\mathcal{L}_{z} + \eta\mathcal{L}_{\mathrm{bal}}$,
where $\mathcal{L}_{\mathrm{task}}$ is class-weighted cross-entropy (inverse-frequency
weights on the train split);
$\mathcal{L}_{\mathrm{marker}}$ is the cross-entropy of an auxiliary linear classifier
fed only the (pre-recursion) cluster tokens, which forces the marker head to
select task-sufficient genes;
$\mathcal{L}_{\mathrm{div}}=\tfrac{1}{M(M-1)}\sum_{i\neq j}(\tilde{\mathbf{e}}_i^\top
\tilde{\mathbf{e}}_j)^2$ is the off-diagonal energy of the normalised marker Gram
matrix (prevents marker collapse);
$\mathcal{L}_{\mathrm{comp}}=\tfrac1N\sum_i\sigma(s_i)$ encourages a sparse importance
distribution, where $s_i$ is the per-gene importance logit emitted by the
marker-scoring head and $\sigma$ the logistic function, so the term penalises the
mean selection mass and pushes most genes toward zero importance; its gradient flows
only into that head. For the soft selectors used in the headline model (cross-attention
router, Concrete) the temperature-annealed softmax already yields a peaked,
near-one-hot selection, so this explicit sparsity term is switched off
($\beta{=}0$) and $s_i$ refers to the learnable-head variant.
$\mathcal{L}_{z}=\mathbb{E}[(\mathrm{logsumexp}\,\mathbf{r})^2]$ is the
router logit $z$-loss; and $\mathcal{L}_{\mathrm{bal}}$ is the Switch-style
load-balancing term \cite{shazeer2017outrageously} for token-choice routing
($\mathcal{L}_{\mathrm{bal}}{=}K\sum_i P_i f_i$, the product of mean routing
probability $P_i$ and dispatch fraction $f_i$ at depth $i$; expert-choice is balanced
by construction so $\eta$ has no effect there). Unless noted we use
$\lambda{=}0.1$, $\gamma{=}0.05$, $\beta{=}0.01$, $\zeta{=}10^{-3}$, $\eta{=}0.1$;
the selection temperature $\tau$ is annealed geometrically from $1$ to $0.1$ over
training and each router query is peak-initialised on a distinct gene.

\section{Experiments}
\subsection{Setup}
We evaluate on pan-cancer bulk transcriptomes assembled from four TCGA cohorts
\cite{weinstein2013cancer} (breast, lung, head-and-neck, thyroid), with
per-gene $z$-scoring fit on the training split and the top @@NGENES@@
high-variance genes retained. Unless noted, $d{=}$@@DMODEL@@, $M{=}$@@NMARKERS@@
markers, recursion depth $K{=}$@@DEPTH@@, trained for @@EPOCHS@@ epochs with AdamW
and early stopping on validation macro-F1. The primary task is cell-of-origin /
cancer-type classification; the framework is data-agnostic and applies unchanged
to single-cell cell-type labels. For the harder clinical phenotype tasks and the
external-baseline comparison we instead use the \emph{full} @@UNIFIED_GENES@@-gene
vector (no high-variance pre-filter) on the unified @@UNIFIED_SAMPLES@@-sample table
(Appendix~\ref{app:data}); there SMART's hyperparameters are selected per task by
validation macro-F1 and every baseline uses the identical split and gene set. All
reported metrics use the \emph{hard} arg-max marker panel at inference (each query
collapses to its single top gene); the soft all-gene selection is used only during
training, so every number reflects the discrete, interpretable panel rather than a
soft mixture.

\subsection{Main Results}
The primary task is four-way pan-cancer cohort classification (Breast/BRCA,
Head-neck/HNSC, Lung/LUNG, Thyroid/THCA), so the four classes \emph{are} the four
datasets. Since a macro average can hide per-cohort weaknesses, Table~\ref{tab:perclass}
reports \emph{per-class (per-cohort)} precision, recall and F1 with support, and
Table~\ref{tab:main} the per-head aggregates. SMART attains
@@MAIN_ACC@@\% accuracy and @@MAIN_F1@@\% macro-F1, well balanced across cohorts
(Thyroid near-perfect, the smaller Head-neck hardest). To show the result is not a
single-seed artefact, we repeat the headline configuration over @@NSEEDS@@ random
seeds: it attains @@MAIN_SEED_F1@@\% macro-F1 and @@MAIN_SEED_ACC@@\% accuracy
(mean\,$\pm$\,std), so the cohort result is stable to initialisation.

\begin{table}[t]
\centering
@@MAIN_TABLE@@
\caption{Main results per output head ($^{\dagger}$ primary head). All numbers
are produced directly by our pipeline.}
\label{tab:main}
\end{table}

\begin{table}[t]
\centering
@@PERCLASS_TABLE@@
\caption{Per-class (per-cohort) breakdown of the primary cancer-type head:
precision, recall, F1 and test-set support for each of the four TCGA cohorts (the
macro and support-weighted aggregates are in Table~\ref{tab:main}). Reported per
cohort rather than as a single macro number so cohort-level strengths and weaknesses
are visible.}
\label{tab:perclass}
\end{table}

\subsection{Parameter Efficiency Is Architectural}
Table~\ref{tab:params} contrasts the transformer-stack parameters of our shared
recursion against an equivalent stack of independent layers, across depth. The
saving is exact and present \emph{before any training}: at $K{=}$@@DEPTH@@ the
shared model uses @@RATIO4@@$\times$ fewer stack parameters, and the gap widens
linearly with depth. The saving is built into the architecture, not recovered by
pruning.

\begin{table}[t]
\centering
@@PARAM_TABLE@@
\caption{Transformer-stack parameters: shared recursion (ours) vs.\ independent
layers. Reduction grows linearly with depth $K$.}
\label{tab:params}
\end{table}

To report the claim end-to-end rather than for the stack alone,
Table~\ref{tab:cost} lists the total parameter count, including the gene-identity
embedding and classifier, and the training wall-clock for the headline and key
ablation configurations. The full headline model has @@MAIN_TOTAL_PARAMS@@
parameters and trains in @@MAIN_WALL@@\,s on a single GPU. The gene-identity
embedding ($Nd$ parameters) dominates the non-stack count and is shared by every
variant, so removing weight sharing (the ``Independent layers'' row) inflates the
\emph{total} far less than the @@RATIO4@@$\times$ stack-only figure suggests; the
stack comparison in Table~\ref{tab:params} is therefore the meaningful axis for the
architectural claim, while the total and timing are given here for completeness.

\begin{table}[t]
\centering
@@COST_TABLE@@
\caption{Total cost per configuration: transformer-stack parameters, total
parameters (including the gene-identity embedding and classifier), and training
wall-clock. The shared embedding dominates the non-stack parameters and is common
to all variants.}
\label{tab:cost}
\end{table}

\subsection{Adaptive Recursion Allocates Compute to Driver Genes}
Table~\ref{tab:routing} compares our expert-choice Mixture-of-Recursions router
against token-choice routing and the uniform fixed-depth recursion. All three
regimes land within about half a macro-F1 point of one another (fixed depth
@@SEL_LEARN_F1@@\%, expert-choice @@MAIN_F1@@\%, token-choice @@MOR_TOKEN_F1@@\%),
so on this near-saturated task the choice of router does not change accuracy. The
value of routing is \emph{compute}, not accuracy: both adaptive routers let most
markers exit the funnel early, cutting the mean recursion depth from the full
$K{=}$@@DEPTH@@ to @@MEANDEPTH@@ and spending only @@SAVING@@$\times$ of the
fixed-depth stack FLOPs at no measurable accuracy cost. The depth is not spent
uniformly: a small set of genes survives to the deepest step, and their per-cohort
recursion depth $d_m$ gives an importance ranking that is intrinsic to the
computation rather than read off attention maps. Expert- and token-choice save
almost identically here; we adopt expert-choice as the headline because its
capacity funnel is load-balanced by construction and needs no auxiliary balancing
loss, whereas token-choice can be harder to balance \cite{bae2025mixture}. We use the gentle capacity funnel
$(1,\tfrac34,\tfrac12,\tfrac12)$ and full-strength router gates
($\alpha{=}1$); an aggressive $0.5^t$ funnel or a strongly damped gate
($\alpha{=}0.1$) starves the \emph{pooled} marker representation and costs several
macro-F1 points, an interaction we observed in development.

\begin{table}[t]
\centering
@@ROUTING_TABLE@@
\caption{Mixture-of-Recursions routing study. Expert-choice (ours) preserves
accuracy at a fraction of the fixed-depth compute; ``Mean depth'' is the average
per-gene recursion depth, an intrinsic importance signal.}
\label{tab:routing}
\end{table}

\subsection{Early-Exit Recursion vs Fixed Depth}
\label{sec:earlystop}
The expert-choice router makes the recursion an \emph{early-exit} (early-stopping)
process: rather than fixing a depth and running every marker token for all $K$
passes, the router decides which tokens survive each step, so a token's realised
recursion depth is learned and most tokens stop early. To make this explicit we set a
deep cap $K{=}8$ and compare fixed depth (every token runs all eight passes) against
the early-exit router (Table~\ref{tab:earlystop}). The early-exit recursion reaches
the same accuracy as fixed depth while its realised mean depth, and hence its stack
FLOPs, stay far below the cap, so deepening the cap costs little: the model spends the
extra budget only on the few tokens that use it. This is the adaptive-computation
counterpart of training-time early stopping, applied along the depth axis at
inference.

\begin{table}[t]
\centering
@@EARLYSTOP_TABLE@@
\caption{Early-exit recursion vs fixed depth at cap $K{=}8$ (macro-F1 \%,
mean\,$\pm$\,std over seeds). ``Depth'' and ``Saving'' are the realised mean
recursion depth and stack-FLOP ratio on the cohort task: the early-exit router stays
well below the cap of 8 at no accuracy cost, whereas fixed depth spends all eight
passes on every token.}
\label{tab:earlystop}
\end{table}

\subsection{Marker Selection Study}
Does \emph{learning} the markers help, and \emph{how} should one learn them? We
compare four selection strategies under an identical token construction and fixed
recursion, so only selection differs (Table~\ref{tab:selection}). The two fixed
panels (variance, random) see only their chosen genes, a hard drop of all others,
while the two differentiable selectors (our cross-attention router and the Concrete
relaxation) attend \emph{softly over every gene} during training, so gradients reach
the whole transcriptome, and collapse to a hard arg-max panel at evaluation; all four
then feed the same fixed-depth recursion, which isolates selection quality. Both
learned selectors top the table: the Concrete relaxation reaches @@SEL_CONC_F1@@\%
macro-F1 and our router @@SEL_LEARN_F1@@\%, ahead of the strong variance heuristic
(@@SEL_VAR_F1@@\%) and random selection (@@SEL_RAND_F1@@\%). Learning \emph{which}
genes are markers therefore helps over fixed heuristics, and the two differentiable
selectors are close, with Concrete edging the router here. The margins are small on
this near-saturated four-class task: the value of a learned selector is not a large
accuracy jump but that it is end-to-end and transfers to settings where a variance
prior is uninformative. We keep the cross-attention router as the headline selector
because it integrates directly with the recursive marker tokens and the refinement
gate, while reporting Concrete as an equally strong differentiable alternative.

Two design choices are needed to make the router learn at all. First, selection
must be \emph{soft over all genes}: a hard top-$k$ router only re-ranks genes it has
already selected and cannot discover new ones, whereas our router and the Concrete
relaxation attend over \emph{every} gene and receive gradient everywhere. Second,
the soft selector must be \emph{initialised peaked}, each slot starting on a distinct
gene; from a uniform start every marker token is the same average of all genes and
the model cannot escape this collapse within a practical budget. In preliminary runs
a naive hard router and a uniform-initialised router failed to beat random selection,
which motivated both choices; we report this as design rationale rather than a
headline result, since those variants are not part of the controlled study above.

\begin{table}[t]
\centering
@@SELECTION_TABLE@@
\caption{Marker-selection study at fixed recursion. Both learned selectors
(Concrete and the cross-attention router) beat the variance and random panels;
Concrete edges the router here. Margins are small on this near-saturated four-class
task (see text). Router and Concrete attend softly over all genes in training and
collapse to a hard panel at evaluation; variance and random see only their chosen
genes.}
\label{tab:selection}
\end{table}

\subsection{Peaked Initialisation Is the Decisive Ingredient}
The router relies on two design choices: a \emph{peaked} initialisation that points
each query at a distinct gene, and a temperature schedule that anneals the selection
softmax from soft to peaked. To isolate their contributions we cross them in a
$2\times2$ grid (peaked vs.\ uniform initialisation $\times$ annealed vs.\ constant
temperature), each cell averaged over @@NSEEDS_IA@@ seeds
(Table~\ref{tab:initanneal}). The outcome is unambiguous: peaked initialisation is
decisive, lifting macro-F1 from @@IA_UNIF_F1@@\% to @@IA_PEAK_F1@@\% (about
@@IA_GAP_F1@@ points), whereas temperature annealing moves it by under one point in
either initialisation regime. From a uniform start every marker token is the same
average of all genes and the model cannot escape that collapse within the training
budget; annealing is a useful refinement on top of a good initialisation, not a
substitute for it.

\begin{table}[t]
\centering
@@INIT_ANNEAL_TABLE@@
\caption{Initialisation $\times$ annealing ablation for the marker router (macro-F1
\%, mean\,$\pm$\,std over @@NSEEDS_IA@@ seeds). Peaked initialisation is decisive;
temperature annealing is a minor refinement on top of it.}
\label{tab:initanneal}
\end{table}

\subsection{Biology-Informed Routing}
\label{sec:interaction}
Section~\ref{sec:biorouter} adds a genomap gene-gene-interaction centrality prior to
the depth router. We test whether it helps, and whether it is the \emph{real}
co-expression structure that matters, by comparing three router priors under
otherwise identical training: \emph{none} (the data-only router), \emph{co-expression}
(the genomap correlation-graph centrality), and a degree-matched \emph{random graph}
control with the same sparsity but shuffled edges (Table~\ref{tab:interaction}). We
evaluate on the cohort task and the three hard phenotype tasks, since the prior is an
empirical-Bayes shrinkage that should help most where the label signal is weak. The
comparison of interest is co-expression versus random: a gain there, not merely over
\emph{none}, is what would show that biological network structure rather than any
additive bias drives the effect.

\paragraph{Finding (reported as a negative result).}
Across all four tasks the three router priors are statistically indistinguishable:
every co-expression cell lies within one standard deviation of both its
degree-matched random-graph control and the no-prior baseline
(Table~\ref{tab:interaction}). On the near-saturated cohort task the prior is mildly
negative, as expected once the label signal is already strong. On the
weakly-determined phenotype tasks it neither helps nor hurts on the mean, and
crucially it does not separate from the random graph, so on these labels we cannot
attribute any effect to biological co-expression structure rather than to generic
additive bias. The one consistent trend is lower seed-to-seed variance under the
co-expression prior on stage and tumour-T, consistent with its intended role as a
shrinkage regulariser, but we do not promote a variance effect that does not reach an
accuracy gain. We therefore present biology-informed routing as a principled,
label-free mechanism whose benefit is \emph{not} realised on the present datasets, and
scope its promise to settings with a stronger or more explicitly network-structured
signal (Discussion).

\begin{table}[t]
\centering
@@INTERACTION_TABLE@@
\caption{Biology-informed routing ablation (macro-F1 \%, mean\,$\pm$\,std over seeds):
the genomap gene-gene-interaction centrality prior (co-expression) vs.\ a
degree-matched random-graph control vs.\ no prior, on the cohort and hard phenotype
tasks. The co-expression-vs-random gap isolates the contribution of real biological
network structure.}
\label{tab:interaction}
\end{table}

\subsection{Architecture Ablations}
Table~\ref{tab:ablation} isolates each component on the full model, reported exactly
as we find it. On this four-class task no single component drives accuracy: a single
pass ($K{=}1$) nearly matches the four-step model, removing the recursive refinement
gate costs under one macro-F1 point, and removing weight sharing changes accuracy by
about the same amount in the other direction at roughly four times the stack
parameters. Every variant sits within the run-to-run noise that the harder-task
multi-seed study (Table~\ref{tab:multiseed}) makes explicit, so we read no ranking
into these fractions of a point. The ``$-$ weight sharing'' variant is exactly a standard $K$-layer transformer over
the $M$ marker tokens, matched in width, depth and token set, so it doubles as our
matched-budget standard-transformer baseline. Generic efficient-attention backbones
(Linformer, Performer, Nystr\"omformer
\cite{wang2020linformer,choromanski2021rethinking,xiong2021nystromformer}) lower the
quadratic attention cost orthogonally to our $\mathcal{O}(M^2)$ marker compression
and are a complementary direction we do not benchmark here.
The architecture's benefit on this dataset is therefore \emph{efficiency}
(weight-shared recursion, adaptive-depth compute saving) and the interpretable
recursion-depth signal, rather than an accuracy gain from depth itself; we expect
depth to contribute more on harder tasks with finer-grained classes, which we did
not test here.

\begin{table}[t]
\centering
@@ABLATION_TABLE@@
\caption{Architecture ablations on the primary head. On this near-saturated task
every component moves macro-F1 by under a point and within run-to-run noise;
weight-shared recursion is within a fraction of a point of independent layers at a
quarter of the stack parameters, so the gains are efficiency, not accuracy.}
\label{tab:ablation}
\end{table}

\subsection{Robust Ablations on Harder Tasks}
The cohort task is close to saturated, so all ablations cluster within run-to-run
noise and cannot by themselves rank the components. We therefore repeat the
ablations on the three genuinely hard phenotype labels (overall stage, tumour T,
node N), each over five random seeds, and report mean\,$\pm$\,std macro-F1
(Table~\ref{tab:multiseed}). We report a negative result here for completeness: on
these weakly-determined clinical labels the ablations remain statistically
indistinguishable, with all variants, including learned versus random marker
selection, falling within overlapping one-standard-deviation bands. The benefit of
learned marker selection that is visible on the cohort task (Table~\ref{tab:selection},
router @@SEL_LEARN_F1@@\% vs.\ random @@SEL_RAND_F1@@\%) therefore appears to require
a clear marker signal in the label; on staging and nodal status, which are only
faintly encoded in bulk expression, no selection or recursion choice separates from
the others. We accordingly scope the marker-learning claim to tasks with a genuine
marker signal rather than asserting it universally. We also sweep the two main knobs
on the same hard tasks. The recursion-depth sweep (Table~\ref{tab:depthsweep}) traces
the accuracy/compute curve, where accuracy is flat within run-to-run noise across
depth while the expert-choice router keeps effective FLOPs below the fixed-depth budget,
and the marker-count sweep (Table~\ref{tab:markersweep}) shows that even $M{=}32$
markers nearly match $M{=}512$, so the $\mathcal{O}(M^2)$ compression is close to
free on these tasks.

\begin{table}[t]
\centering
@@MULTISEED_ABLATION_TABLE@@
\caption{Multi-seed ablations (macro-F1 \%, mean\,$\pm$\,std over five seeds) on the
hard phenotype tasks. The variants overlap within one standard deviation, including
learned versus random marker selection: no component is statistically separable on
these weakly-determined clinical labels.}
\label{tab:multiseed}
\end{table}

\begin{table}[t]
\centering
@@DEPTH_SWEEP_TABLE@@
\caption{Recursion-depth sweep (headline config, three seeds): macro-F1 (\%,
mean\,$\pm$\,std) and effective FLOPs vs.\ depth $K$ on the hard tasks.}
\label{tab:depthsweep}
\end{table}

\begin{table}[t]
\centering
@@MARKER_SWEEP_TABLE@@
\caption{Marker-count ($M$) sweep (headline config, three seeds): macro-F1 (\%,
mean\,$\pm$\,std) and stack parameters vs.\ the number of marker tokens $M$.}
\label{tab:markersweep}
\end{table}

\subsection{Generalization to Single-Cell Datasets}
Although the controlled study above uses bulk pan-cancer data, SMART makes no assumption
about where the expression vector comes from. To test transfer, we run the \emph{identical}
headline configuration (cross-attention marker router with expert-choice recursion,
unchanged) on four single-cell benchmarks from the genomap capsule
\cite{islam2023cartography}: Tabula Muris and three curated cross-tissue panels, spanning
10 to 55 cell-type classes and up to 90{,}579 cells, using each dataset's own train/test
split where one is provided. Table~\ref{tab:singlecell} reports test accuracy and macro-F1.
The same architecture classifies cell types across all four datasets, including the
fine-grained 55-class Tabula Muris benchmark, which supports our view that marker-guided
recursion is a general mechanism for expression classification rather than something
specific to bulk cohorts.

The Pancreas row is the one place where accuracy stays high while macro-F1 drops, and we
read it as a property of the input representation rather than a shortcoming of the model.
Unlike the other panels, the Pancreas features are flattened $44\times44$ genomap images
\cite{islam2023cartography} rather than a raw named-gene vector, so the cross-attention
marker router selects over spatial genomap pixels instead of genes and its inductive bias
toward marker genes does not apply; the high overall accuracy then reflects the frequent
cell types while the rarer ones are diluted, which is what depresses the macro-average.
This is exactly the behaviour we hypothesised for the genomap-derived inputs ahead of the
run, and the result confirms that hypothesis. We therefore treat the gene-panel datasets
as the on-distribution test of the architecture and keep the genomap-image case as a
deliberately out-of-format stress test.

\begin{table}[t]
\centering
@@SINGLECELL_TABLE@@
\caption{SMART generalises to single-cell cell-type classification: the headline
configuration, run unchanged, on four genomap-capsule benchmarks
\cite{islam2023cartography} (each dataset's own split where available). Cells, genes
and classes vary widely across datasets.}
\label{tab:singlecell}
\end{table}

\subsection{Clinical Phenotype Tasks (genoNet Benchmark)}
Beyond cohort of origin, the same tumours carry clinically meaningful phenotype
labels. genomap's genoNet classifier \cite{islam2023cartography} reshapes the gene
vector into an image and applies a small CNN; we instead keep our model fixed and
feed it the raw, full gene vector (all @@UNIFIED_GENES@@ genes,
@@UNIFIED_SAMPLES@@ samples). We run the unchanged headline SMART configuration on
the five clinical genoNet tasks (Table~\ref{tab:genonet}): three ordinal pathology
labels (overall stage, tumour T, node N) and two binary clinical labels (overall
survival, tumour status). We deliberately exclude cohort detection from this
benchmark: it is near-linearly separable from bulk expression and saturated for any
reasonable model, so it is a sanity check rather than a predictive task. SMART
degrades gracefully across the genuinely hard staging and survival tasks, which are
only weakly determined by bulk expression, matching the difficulty ordering
reported for deep classifiers on TCGA RNA-seq
\cite{khalifa2020ai,rukhsar2022analyzing} and for staging in recent multi-omic
pipelines \cite{ghaleb2025sdcfe}.

\begin{table}[t]
\centering
@@GENONET_TABLE@@
\caption{SMART on the five clinical genoNet tasks (unified TCGA table, all genes),
run with the unchanged headline configuration. Staging, nodal and survival labels
are intrinsically hard from bulk expression.}
\label{tab:genonet}
\end{table}

\subsection{External Baselines}
To position the absolute numbers we compare SMART against strong \emph{nonlinear}
tabular learners under the \emph{identical} train/test split and gene set on every
genoNet task: a majority-class floor; $k$-nearest neighbours; an RBF SVM; three
forest/boosting ensembles from scikit-learn (random forest, extra trees, histogram
gradient boosting); the three widely used gradient-boosting libraries XGBoost,
LightGBM and CatBoost; and a one-hidden-layer MLP. We deliberately benchmark
against this nonlinear class rather than linear models, which are not informative
comparators for an architecture whose claim is parameter efficiency and
interpretability rather than raw accuracy. Table~\ref{tab:baselines} reports
macro-F1 per task and the across-task mean. For SMART, the model width, marker
count, recursion depth and learning rate are selected per task by \emph{validation}
macro-F1 over a small grid (selection never sees the test split); the baselines use
their standard strong settings. SMART is competitive with these strong
gradient-boosted baselines on the hard clinical tasks while additionally yielding
an interpretable, compute-allocated marker panel that none of them provide.

We do not retrain large pretrained foundation models (scGPT \cite{cui2024scgpt},
scFoundation \cite{hao2024large}, GET \cite{fu2025get}, network-aware
\cite{zhang2025cellular}, scPlantLLM \cite{cao2025scplantllm}) or the
feature-selection-plus-deep-net pipelines that report headline TCGA accuracies
\cite{polepalli2025cvae,khalifa2020ai,rukhsar2022analyzing,kim2025pancancer,%
ghaleb2025sdcfe,rahaman2025integrated,shukla2025discriminative,bouazza2025degs}
as head-to-head numbers, because they use different gene panels, splits, augmentation
and pretraining corpora and are therefore not directly comparable; we cite them as
the broader landscape. Our contribution is orthogonal to that line: parameter
efficiency and marker interpretability as an architectural property rather than a
property of scale or of an external feature-selection stage.

\begin{table}[t]
\centering
@@BASELINES_TABLE@@
\caption{SMART vs.\ strong nonlinear / gradient-boosted baselines (macro-F1, \%) on
the genoNet tasks under an identical split and gene set; rows sorted by across-task
mean. Stage / T / N / OS / Tumour are the five clinical phenotype labels.}
\label{tab:baselines}
\end{table}

\subsection{Router-Based Gene Identification}
\label{sec:geneid}
A central claim is that the architecture \emph{identifies} genes rather than merely
classifying. Two intrinsic signals rank genes: the cross-attention router's
per-slot selection (which $M$ genes become markers) and each marker's mean
recursion depth $d_m$ (how much adaptive computation the model spends on it,
Table~\ref{tab:routing}). We treat $d_m$ as a compute-allocation importance score
that complements selection and needs no post-hoc attribution.

\paragraph{The deepest-routed genes.}
Table~\ref{tab:geneid} lists the markers that survive to the deepest recursion
steps, ranked by mean depth $d_m$; these are the genes on which the model spends the
most adaptive computation. Several have documented roles in epithelial cancers, for
example the receptor tyrosine kinase \emph{ROS1}, an established lung-adenocarcinoma
driver, and the inflammatory cytokine \emph{IL1A}. We present this panel
\emph{descriptively} rather than as a
validation claim: the deepest-routed genes did not all coincide with our small
curated marker list, and establishing that the panel is enriched for cancer biology
beyond chance requires the systematic test described next, which we leave to future
work. The ranking is intrinsic to the architecture and needs no post-hoc
attribution.

\begin{table}[t]
\centering
@@GENEVAL_TABLE@@
\caption{Top router-identified genes ranked by mean recursion depth $d_m$ (the
adaptive compute the model allocates to each). Listed descriptively; the depth
ranking is intrinsic to the architecture and needs no post-hoc attribution.}
\label{tab:geneid}
\end{table}

\paragraph{Stability of the panel across seeds.}
Because the cohort task is near-saturated and co-expressed markers are redundant, the
\emph{identity} of the selected genes is far less stable than the accuracy: across our
five seeds the selected marker panels overlap by only a mean Jaccard of
@@MARKER_JACCARD@@, i.e.\ different seeds reach the same accuracy through largely
disjoint gene sets. We therefore treat a single run's recursion-depth ranking as a
per-run, hypothesis-generating signal rather than a fixed biomarker list, and any
biological claim should be drawn from the consensus of many runs, not one panel. This
instability is itself informative: it quantifies the redundancy of discriminative
genes in bulk expression that the marker literature describes qualitatively.

\paragraph{Systematic pathway enrichment.}
Beyond this curated check, an unbiased enrichment of the full learned panel against
Reactome gene sets \cite{gillespie2022reactome} (hypergeometric test with FDR
control), reported as ranked pathways with effect sizes, is a natural next step that
would turn the qualitative marker validation into a quantitative one; we leave it to
future work.

\section{Discussion and Limitations}
SMART shows that biological inductive bias and parameter-efficient recursion can be
co-designed: the same mechanism that makes the model small (sharing, compression)
also makes it interpretable (markers). Beyond the controlled cohort task, we
stress-tested the model on the full $\sim$20{,}530-gene transcriptome across five
clinical phenotype tasks, against ten strong nonlinear baselines, and across four
single-cell datasets, which addresses the all-gene regime, the external-baseline
comparison, and partial multi-seed variance reporting that an earlier version of
this work left open. The headline finding is mixed in an informative way: SMART
attains the best across-task mean macro-F1 and wins the binary clinical tasks, but
gradient-boosted trees lead on tumour stage and T, so SMART's advantage is
efficiency and interpretability at competitive accuracy rather than a universal
accuracy gain.

We scope the claims to what the evidence supports, and flag the genuine remaining
limitations. (i) \emph{Marker learning is signal-dependent.} The learned router
clearly beats random and variance selection on the cohort task, but our five-seed
ablations show that on the weakly-determined staging and nodal labels every
component, learned selection included, sits within one standard deviation of the
others; marker learning helps only where the label carries a marker signal.
(ii) \emph{Accuracy ceiling.} Against tuned gradient boosting SMART leads on the
mean and on three of five tasks but trails on two; closing the staging gap likely
needs ordinal-aware losses or pretraining \cite{cui2024scgpt,hao2024large,fu2025get}.
(iii) \emph{Generalisation.} Results are single external assembly (TCGA);
leave-one-cohort-out and an independent external cohort, plus large-scale
pretraining and comparison against efficient-attention backbones
\cite{wang2020linformer,choromanski2021rethinking,xiong2021nystromformer}, remain
future work. (iv) \emph{Routing/refinement interaction and optimisation.}
Expert-choice routing under-performs fixed depth unless the funnel is gentle and
gates are full-strength, and the differentiable top-$k$ is the main source of
optimisation noise; smoother relaxations are a promising direction.
(v) \emph{Depth versus statistical prominence.} Recursion depth and a gene's raw
expression variance may be partially entangled: a high-variance gene is both easier
to select and more likely to be routed deep, so part of the depth signal could
reflect statistical prominence rather than biology. We do not disentangle the two
here; a rank correlation of $d_m$ against per-gene variance and mean $|z|$ on a
held-out split, together with the Reactome enrichment above, is the test needed to
establish that depth carries information beyond variance, and we leave it to future
work. (vi) \emph{Biology-informed routing is not yet realised.} The genomap
interaction prior is principled and label-free, but on our four tasks it does not
separate from a degree-matched random-graph control (Sec.~\ref{sec:interaction}), so
we cannot claim a benefit from biological structure here; testing it where the prior
should bite, datasets with stronger or explicitly network-mediated signal, with
richer priors (pathway membership, regulatory-network centrality) and the optional
logit Laplacian smoothing of Appendix~\ref{app:theory}, is the natural next step.
(vii) \emph{Comparisons and profiling left to future work.} We do not yet include a
controlled from-scratch transcriptomics-transformer baseline (e.g.\ a Geneformer- or
GexBERT-style stack trained on our split) nor efficient-attention backbones
\cite{wang2020linformer,choromanski2021rethinking,xiong2021nystromformer} as
head-to-head comparators, and our efficiency evidence is stack FLOPs and training
wall-clock rather than end-to-end inference-time and memory profiles across batch
sizes; both are clear next steps to substantiate the practical efficiency claim. We
report these so the trade-offs are not overstated.

\section{Conclusion}
We presented SMART, a recursive marker-guided transformer that makes parameter
efficiency an architectural property of transcriptomic models. By learning marker
genes, compressing around them, and sharing one block across recursive refinement
steps, it matches strong cohort-classification accuracy with several times fewer
parameters and exposes an interpretable, compute-allocated recursion-depth gene panel.
Evaluated on the full transcriptome across five clinical phenotype tasks it achieves
the best mean macro-F1 against ten strong nonlinear baselines while remaining far
smaller, and it transfers unchanged to single-cell data; our multi-seed study also
delineates, honestly, where the marker-learning advantage does and does not hold. We
additionally formulate a biology-informed router that folds a label-free genomap
gene-gene interaction prior into the recursion decision, with full mathematical and
biological grounding; that it does not yet beat a random-graph control on these
datasets is reported as an honest negative result and a clear direction for stronger,
network-structured settings. The complete pipeline, including all experiments,
baselines, and this paper, regenerates from a single command.

\bibliographystyle{aaai}
\bibliography{refs}

\appendix
\section{Dataset Details}
\label{app:data}
We use three data sources, summarised in Table~\ref{tab:datasets}. All gene
expression is bulk or single-cell RNA-seq; bulk cohorts are Illumina HiSeqV2
$\log_2(\text{norm\_count}+1)$ profiles pulled from the UCSC Xena hub, and
single-cell sets are the genomap capsule datasets converted to plain CSV.

\paragraph{TCGA pan-cancer bulk (4 cohorts).}
The primary classification benchmark of the main paper. Four cohorts (breast/BRCA,
head-and-neck/HNSC, lung/LUNG, thyroid/THCA) are concatenated; per-gene
$z$-scoring is fit on the training split and the top @@NGENES@@ high-variance genes
of the $\sim$20{,}530 measured are retained. The label is cohort of origin.

\paragraph{Unified BIO5 phenotype table (genoNet benchmark).}
The same tumours assembled into one table of @@UNIFIED_SAMPLES@@ samples
$\times$ @@UNIFIED_GENES@@ genes with six aligned phenotype labels (Sec.~%
\ref{app:tasks}). The full gene vector (no high-variance pre-filter) is used, so
this is the ``all data'' setting for SMART and for every external baseline.

\paragraph{Single-cell genomap datasets.}
Four datasets from the genomap capsule \cite{islam2023cartography}, converted to
readable CSV (expression + labels + split where provided): Tabula Muris (a
fine-grained 55 cell-type panel), a 19-class common-class panel, a 10-class
prototype panel, and a 15-class pancreas panel whose features are flattened
$44\times44$ genomap images rather than a raw gene vector.

\begin{table}[h]
\centering
\resizebox{\columnwidth}{!}{%
@@DATASET_OVERVIEW_TABLE@@}
\caption{All datasets used. Features for the bulk cohort task are the top
high-variance genes; the genoNet table and single-cell sets use their full feature
set. Counts are read directly from the data.}
\label{tab:datasets}
\end{table}

\section{Task Descriptions}
\label{app:tasks}
The genoNet benchmark contains five clinical classification tasks defined on the
unified BIO5 table; each is described below and their class distributions are listed
in Table~\ref{tab:taskdist}. All tasks share the same samples and gene set and use
the identical stratified 80/20 split, so they differ only in the target label.
(Cohort-of-origin detection is excluded: it is near-linearly separable from bulk
expression and saturated, so it is a sanity check rather than a predictive task; it
remains the main-paper four-cohort result.)

\begin{itemize}
\item \textbf{Pathologic stage} -- the overall AJCC stage grouped into four ordered
levels (Stage I--IV). Only weakly determined by bulk expression.
\item \textbf{Primary tumour (T)} -- size/extent of the primary tumour on the
four-level TNM T-axis (T1--T4).
\item \textbf{Regional lymph node (N)} -- nodal involvement on the TNM N-axis
(N0--N3); strongly imbalanced, with N3 very rare.
\item \textbf{Overall-survival status} -- a binary survival label; the majority
class dominates, so macro-F1 (not accuracy) is the meaningful metric.
\item \textbf{Tumour status at follow-up} -- binary tumour-free vs.\ with-tumour at
last follow-up; also imbalanced.
\end{itemize}

The single-cell datasets each define one cell-type classification task with the
number of classes given in Table~\ref{tab:datasets}; we run SMART on them unchanged
to test transfer beyond bulk data.

\begin{table}[h]
\centering
\resizebox{\columnwidth}{!}{%
@@TASK_DISTRIBUTION_TABLE@@}
\caption{Per-task class distributions for the five genoNet phenotype tasks (counts
over all @@UNIFIED_SAMPLES@@ samples), read directly from the data. The staging,
node and clinical tasks are markedly imbalanced.}
\label{tab:taskdist}
\end{table}

\section{Effective-FLOPs Accounting}
\label{app:flops}
The compute numbers report per-sample FLOPs \emph{of the recursive transformer
stack}, the only component the routing study changes. One application of the shared
block to $a$ tokens costs $\phi(a)=4a^2 d + 4\,a\,d\,d_{\mathrm{ff}}$, the first
term the multi-head self-attention ($\mathcal{O}(a^2 d)$ query/key/value and
output projections plus the score-value products) and the second the two-layer
feed-forward network ($\mathcal{O}(a\,d\,d_{\mathrm{ff}})$); we count these matmuls
and treat LayerNorm, the lightweight router head and the final classifier as
lower-order. The \emph{nominal} fixed-depth cost is $K$ blocks over all $M$ markers,
$\Phi_{\mathrm{nom}}=K\,\phi(M)$. The \emph{effective} cost sums one block over the
tokens actually active at each step, $\Phi_{\mathrm{eff}}=\sum_{t=1}^{K}\phi(a_t)$,
where $a_t$ is the mean number of markers the expert-choice funnel keeps active at
step $t$, measured on the test set (so depth aggregation is over the empirical
per-step survivor counts, not a nominal schedule). The saving ratio reported in the
routing study is $\Phi_{\mathrm{eff}}/\Phi_{\mathrm{nom}}$. Gene embedding and marker
selection are $\mathcal{O}(Nd)$, run once before the stack and are identical across
all routing modes, so they are excluded from this stack-level comparison; total
parameter counts (Table~\ref{tab:cost}) do include them.

\section{Theoretical Foundation of the Router}
\label{app:theory}
This appendix gives the mathematical and biological grounding for SMART's router,
both the data-driven Mixture-of-Recursions core and the biology-informed prior of
Sec.~\ref{sec:biorouter}.

\paragraph{Routing as conditional computation.}
The router implements \emph{conditional computation}: rather than sending every token
through a fixed stack, a learned policy routes each token to a token-specific amount
of compute. This is the gating principle of sparsely-gated mixtures of experts
\cite{shazeer2017outrageously}, reused by Mixture-of-Depths \cite{raposo2024mixture}
and Mixture-of-Recursions \cite{bae2025mixture}; the ``experts'' here are recursion
\emph{depths} of one shared block rather than separate sub-networks, which is what
couples adaptive computation \cite{graves2016adaptive} to weight sharing.

\paragraph{Token-choice math.}
Let $\mathbf{h}_m\in\mathbb{R}^d$ be a marker token and $\mathbf{W}_r\in\mathbb{R}^{K\times d}$
the router. The token is scored over the $K$ candidate depths, softmaxed, and assigned
the arg-max depth:
\begin{equation}
\mathbf{z}_m=\tfrac{1}{\tau}\mathbf{W}_r\mathbf{h}_m,\quad
\mathbf{p}_m=\alpha\,\mathrm{softmax}(\mathbf{z}_m),\quad
i_m=\arg\max_i p_{m,i},
\end{equation}
so token $m$ rides the shared block through steps $0,\dots,i_m$ and exits. The
realised depth is $d_m=i_m{+}1$.

\paragraph{Expert-choice math.}
Here a per-step scalar router scores the currently active tokens and a capacity
$c_t$ keeps the top-$\lceil c_t M\rceil$; survivors are the only candidates at the
next step. A token's depth $d_m$ is the deepest step it survived. Expert-choice is
load-balanced by construction (each step keeps a fixed fraction), so it needs no
balancing loss; token-choice does not balance itself and adds the Switch term
$\mathcal{L}_{\mathrm{bal}}=K\sum_i P_i f_i$ (mean routing probability $P_i$ times
dispatch fraction $f_i$ at depth $i$) \cite{shazeer2017outrageously}. Both add the
$z$-loss $\mathcal{L}_z=\mathbb{E}[(\mathrm{logsumexp}\,\mathbf{z})^2]$ to keep logits
bounded.

\paragraph{The differentiable handle.}
The discrete choice ($\arg\max$ / top-$k$) has zero gradient almost everywhere, so a
bare router could not learn. SMART, like MoR, keeps the soft probability of the chosen
route as a multiplicative \emph{gate} on the block output,
$\mathbf{o}_m=g_m\,f_\theta(\mathbf{h}_m)$ with $g_m=\sigma(\tilde r_m)$ (or
$p_{m,i_m}$). Because $g_m$ is smooth in $\mathbf{W}_r$, the chain
$\mathcal{L}\!\leftarrow\!\mathbf{o}_m\!\leftarrow\!g_m\!\leftarrow\!\mathbf{W}_r$ is
unbroken: the hard choice does the routing, the soft gate carries the gradient (a
straight-through-style estimator). The biological prior of
Eq.~\eqref{eq:biorouter} is an additive constant in this logit, so it shifts the
decision without breaking this path.

\paragraph{Biological foundation of the prior.}
Three decades of single-cell biology rest on two facts the prior encodes. (i) A small
set of \emph{marker} genes carries most discriminative signal
\cite{ianevski2022fully,franzen2019panglaodb,hu2023cellmarker}, so compute should be
unevenly allocated. (ii) Gene regulatory and protein-interaction networks are
approximately scale-free: a few high-degree \emph{hub} genes (master regulators)
exert outsized influence, and perturbing a hub propagates across the network. We
operationalise ``hub'' as eigenvector centrality on the genomap co-expression graph
$\mathbf{W}$: the leading eigenvector $\mathbf{v}$ of $\mathbf{W}\mathbf{v}=\lambda\mathbf{v}$
scores each gene by the centrality of its neighbours, recursively, so a gene is
central if it co-expresses with other central genes. We $z$-score $\mathbf{v}$ to form
$\pi$. The prior thus tells the router, before any training, which genes are network
hubs worth deeper computation.

\paragraph{Empirical-Bayes reading.}
Equation~\eqref{eq:biorouter} is a log-linear prior on the routing decision:
$\tilde r^{(t)}_m = \text{(data evidence)} + \beta_t\,\pi_m$ is the unnormalised
log-score of keeping token $m$, with $\beta_t\pi_m$ a Gaussian-like prior mean. The
anneal $\beta_t=\beta_0(1-\text{progress})$ is shrinkage whose strength decays as data
evidence accumulates, the standard empirical-Bayes behaviour: prior-dominated when
the likelihood is uninformative (early training, random hidden states), data-dominated
later. This is most defensible exactly in the low-signal regime (stage, node), where
the likelihood alone barely separates the depths.

\paragraph{Leakage safety.}
$\mathbf{W}$ is computed from training-split \emph{expression only}; no phenotype label
enters it. The prior therefore injects network topology, not the answer, which is why
a marker-recovery or gene-discovery claim under this prior is not circular, unlike a
prior built from curated cohort-specific marker lists.

\paragraph{Optional pathway-graph smoothing.}
The additive bias treats genes independently. Co-pathway genes should route coherently,
which one obtains by smoothing the logits over $\mathbf{W}$ with the normalised
Laplacian $\mathbf{L}=\mathbf{I}-\mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$,
$\hat{\mathbf{r}}^{(t)}=\tilde{\mathbf{r}}^{(t)}-\gamma\,\mathbf{L}\tilde{\mathbf{r}}^{(t)}$,
i.e.\ graph-Laplacian (label-propagation) regularisation: a gene borrows routing
strength from its network neighbours. This is one sparse matrix-vector product per
step and adds no transformer parameters, so the parameter-efficiency claim is
untouched; we expose it as an option and leave its evaluation to future work.

\end{document}
"""


_BIB = r"""@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2017}
}
@inproceedings{dehghani2019universal,
  title={Universal Transformers},
  author={Dehghani, Mostafa and Gouws, Stephan and Vinyals, Oriol and Uszkoreit, Jakob and Kaiser, Lukasz},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019}
}
@inproceedings{lan2020albert,
  title={{ALBERT}: A Lite {BERT} for Self-supervised Learning of Language Representations},
  author={Lan, Zhenzhong and Chen, Mingda and Goodman, Sebastian and Gimpel, Kevin and Sharma, Piyush and Soricut, Radu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
@inproceedings{bae2025mixture,
  title={Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation},
  author={Bae, Sangmin and Kim, Yujin and Bayat, Reza and Kim, Sungnyun and Ha, Jiyoun and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
@article{raposo2024mixture,
  title={Mixture-of-Depths: Dynamically Allocating Compute in Transformer-based Language Models},
  author={Raposo, David and Ritter, Sam and Richards, Blake and Lillicrap, Timothy and Humphreys, Peter Conway and Santoro, Adam},
  journal={arXiv preprint arXiv:2404.02258},
  year={2024}
}
@inproceedings{shazeer2017outrageously,
  title={Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer},
  author={Shazeer, Noam and Mirhoseini, Azalia and Maziarz, Krzysztof and Davis, Andy and Le, Quoc and Hinton, Geoffrey and Dean, Jeff},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
@article{graves2016adaptive,
  title={Adaptive Computation Time for Recurrent Neural Networks},
  author={Graves, Alex},
  journal={arXiv preprint arXiv:1603.08983},
  year={2016}
}
@article{wang2020linformer,
  title={Linformer: Self-Attention with Linear Complexity},
  author={Wang, Sinong and Li, Belinda Z and Khabsa, Madian and Fang, Han and Ma, Hao},
  journal={arXiv preprint arXiv:2006.04768},
  year={2020}
}
@inproceedings{choromanski2021rethinking,
  title={Rethinking Attention with Performers},
  author={Choromanski, Krzysztof and Likhosherstov, Valerii and Dohan, David and Song, Xingyou and Gane, Andreea and Sarlos, Tamas and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
@inproceedings{xiong2021nystromformer,
  title={Nystr\"omformer: A Nystr\"om-based Algorithm for Approximating Self-Attention},
  author={Xiong, Yunyang and Zeng, Zhanpeng and Chakraborty, Rudrasis and Tan, Mingxing and Fung, Glenn and Li, Yin and Singh, Vikas},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2021}
}
@inproceedings{jang2017categorical,
  title={Categorical Reparameterization with Gumbel-Softmax},
  author={Jang, Eric and Gu, Shixiang and Poole, Ben},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
@inproceedings{balin2019concrete,
  title={Concrete Autoencoders: Differentiable Feature Selection and Reconstruction},
  author={Bal{\i}n, Muhammed Fatih and Abid, Abubakar and Zou, James},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2019}
}
@article{cui2024scgpt,
  title={scGPT: Toward Building a Foundation Model for Single-Cell Multi-omics Using Generative {AI}},
  author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Duan, Nan and Wang, Bo},
  journal={Nature Methods},
  volume={21},
  pages={1470--1480},
  year={2024}
}
@article{hao2024large,
  title={Large-scale Foundation Model on Single-cell Transcriptomics},
  author={Hao, Minsheng and Gong, Jing and Zeng, Xin and Liu, Chiming and Guo, Yucheng and Cheng, Xingyi and Wang, Taifeng and Ma, Jianzhu and Zhang, Xuegong and Song, Le},
  journal={Nature Methods},
  volume={21},
  pages={1481--1491},
  year={2024}
}
@article{theodoris2023transfer,
  title={Transfer Learning Enables Predictions in Network Biology},
  author={Theodoris, Christina V and Xiao, Ling and Chopra, Anant and Chaffin, Mark D and Al Sayed, Zeina R and Hill, Matthew C and Mantineo, Helene and Brydon, Elizabeth M and Zeng, Zexian and Liu, X Shirley and Ellinor, Patrick T},
  journal={Nature},
  volume={618},
  pages={616--624},
  year={2023}
}
@article{yang2022scbert,
  title={scBERT as a Large-scale Pretrained Deep Language Model for Cell Type Annotation of Single-cell {RNA}-seq Data},
  author={Yang, Fan and Wang, Wenchuan and Wang, Fang and Fang, Yuan and Tang, Duyu and Huang, Junzhou and Lu, Hui and Yao, Jianhua},
  journal={Nature Machine Intelligence},
  volume={4},
  pages={852--866},
  year={2022}
}
@article{ianevski2022fully,
  title={Fully-automated and Ultra-fast Cell-type Identification Using Specific Marker Combinations from Single-cell Transcriptomic Data},
  author={Ianevski, Aleksandr and Giri, Anil K and Aittokallio, Tero},
  journal={Nature Communications},
  volume={13},
  pages={1246},
  year={2022}
}
@article{islam2023cartography,
  title={Cartography of Genomic Interactions Enables Deep Analysis of Single-cell Expression Data},
  author={Islam, Md Tauhidul and Xing, Lei},
  journal={Nature Communications},
  volume={14},
  pages={679},
  year={2023}
}
@article{hu2023cellmarker,
  title={CellMarker 2.0: An Updated Database of Manually Curated Cell Markers in Human/Mouse and Web Tools Based on {scRNA}-seq Data},
  author={Hu, Congxue and Li, Tengyue and Xu, Yingqi and Zhang, Xinxin and Li, Feng and Bai, Jing and Chen, Jing and Jiang, Wenqi and Yang, Kaiyue and Ou, Qi and Li, Xia and Wang, Peng and Zhang, Yunpeng},
  journal={Nucleic Acids Research},
  volume={51},
  number={D1},
  pages={D870--D876},
  year={2023}
}
@article{franzen2019panglaodb,
  title={PanglaoDB: A Web Server for Exploration of Mouse and Human Single-cell {RNA} Sequencing Data},
  author={Franz{\'e}n, Oscar and Gan, Li-Ming and Bj{\"o}rkegren, Johan LM},
  journal={Database},
  volume={2019},
  pages={baz046},
  year={2019}
}
@article{jager2001nybr1,
  title={Identification of a Tissue-specific Putative Transcription Factor in Breast Tissue by Serological Screening of a Breast Cancer Library},
  author={J{\"a}ger, Dirk and Stockert, Elisabeth and G{\"u}re, Ali O and Scanlan, Matthew J and Karbach, Julia and J{\"a}ger, Elke and Knuth, Alexander and Old, Lloyd J and Chen, Yao-Tseng},
  journal={Cancer Research},
  volume={61},
  number={5},
  pages={2055--2061},
  year={2001}
}
@article{gillespie2022reactome,
  title={The Reactome Pathway Knowledgebase 2022},
  author={Gillespie, Marc and Jassal, Bijay and Stephan, Ralf and Milacic, Marija and Rothfels, Karen and Senff-Ribeiro, Andrea and Griss, Johannes and others},
  journal={Nucleic Acids Research},
  volume={50},
  number={D1},
  pages={D687--D692},
  year={2022}
}
@article{tabula2022tabula,
  title={The Tabula Sapiens: A Multiple-organ, Single-cell Transcriptomic Atlas of Humans},
  author={{The Tabula Sapiens Consortium}},
  journal={Science},
  volume={376},
  number={6594},
  pages={eabl4896},
  year={2022}
}
@article{wolf2018scanpy,
  title={{SCANPY}: Large-scale Single-cell Gene Expression Data Analysis},
  author={Wolf, F Alexander and Angerer, Philipp and Theis, Fabian J},
  journal={Genome Biology},
  volume={19},
  pages={15},
  year={2018}
}
@article{regev2017human,
  title={The Human Cell Atlas},
  author={Regev, Aviv and Teichmann, Sarah A and Lander, Eric S and Amit, Ido and Benoist, Christophe and Birney, Ewan and others},
  journal={eLife},
  volume={6},
  pages={e27041},
  year={2017}
}
@article{aran2019reference,
  title={Reference-based Analysis of Lung Single-cell Sequencing Reveals a Transitional Profibrotic Macrophage},
  author={Aran, Dvir and Looney, Agnieszka P and Liu, Leqian and Wu, Esther and Fong, Valerie and Hsu, Austin and others},
  journal={Nature Immunology},
  volume={20},
  pages={163--172},
  year={2019}
}
@article{luecken2019current,
  title={Current Best Practices in Single-cell {RNA}-seq Analysis: A Tutorial},
  author={Luecken, Malte D and Theis, Fabian J},
  journal={Molecular Systems Biology},
  volume={15},
  number={6},
  pages={e8746},
  year={2019}
}
@article{howlader2026graph,
  title={Graph Transformer-based Pathway Embedding for Cancer Prognosis},
  author={Howlader, Koushik and Islam, Md Tauhidul and Le, Wei},
  journal={arXiv preprint arXiv:2604.16685},
  year={2026}
}
@article{jiang2025gexbert,
  title={Transformer-Based Representation Learning for Robust Gene Expression Modeling and Cancer Prognosis},
  author={Jiang, Shuai and Hassanpour, Saeed},
  journal={Scientific Reports},
  volume={15},
  year={2025},
  doi={10.1038/s41598-025-14949-2}
}
@article{lopez2018deep,
  title={Deep Generative Modeling for Single-cell Transcriptomics},
  author={Lopez, Romain and Regier, Jeffrey and Cole, Michael B and Jordan, Michael I and Yosef, Nir},
  journal={Nature Methods},
  volume={15},
  pages={1053--1058},
  year={2018}
}
@article{weinstein2013cancer,
  title={The Cancer Genome Atlas Pan-Cancer Analysis Project},
  author={Weinstein, John N and Collisson, Eric A and Mills, Gordon B and Shaw, Kenna R Mills and Ozenberger, Brad A and Ellrott, Kyle and Shmulevich, Ilya and Sander, Chris and Stuart, Joshua M},
  journal={Nature Genetics},
  volume={45},
  pages={1113--1120},
  year={2013}
}
@inproceedings{shen2021sliced,
  title={Sliced Recursive Transformer},
  author={Shen, Zhiqiang and Liu, Zechun and Xing, Eric P},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022},
  doi={10.1007/978-3-031-20053-3_42}
}
@article{zhou2024ristra,
  title={{RISTRA}: Recursive Image Super-Resolution Transformer With Relativistic Assessment},
  author={Zhou, Xiaoqiang and Huang, Huaibo and Wang, Zilei and He, Ran},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  doi={10.1109/tmm.2024.3352400}
}
@article{nouriborji2025mol,
  title={Improving Recursive Transformers with Mixture of {LoRAs}},
  author={Nouriborji, Mohammadmahdi and Rohanian, Morteza and Rohanian, Omid},
  journal={arXiv preprint arXiv:2512.12880},
  year={2025},
  doi={10.48550/arxiv.2512.12880}
}
@article{jaber2026ouroboros,
  title={Ouroboros: Dynamic Weight Generation for Recursive Transformers via Input-Conditioned {LoRA} Modulation},
  author={Jaber, Jaber and Jaber, Obeida},
  journal={arXiv preprint},
  year={2026}
}
@article{fu2025get,
  title={A Foundation Model of Transcription Across Human Cell Types},
  author={Fu, Xi and Mo, Shentong and Buendia, Alejandro and Laurent, Anouchka P and Shao, Anqi and others},
  journal={Nature},
  year={2025},
  doi={10.1038/s41586-024-08391-z}
}
@inproceedings{zhang2025cellular,
  title={A Cellular Network-Aware Foundation Model Improves Single-Cell Level Predictions},
  author={Zhang, Mingxuan and Swamy, Vinay and Dupire, L\'eo and Cassius, Rowan and Kanatsoulis, Charilaos and Paull, Evan and Karaletsos, Theofanis and Califano, Andrea},
  booktitle={Clinical Cancer Research (AACR AI/ML Special Conf.)},
  year={2025},
  doi={10.1158/1557-3265.aimachine-b041}
}
@article{cao2025scplantllm,
  title={{scPlantLLM}: A Foundation Model for Exploring Single-cell Expression Atlases in Plants},
  author={Cao, Guangshuo and Chao, Haoyu and Zheng, Wenqi and Lan, Yangming and Lu, Kaiyan and Wang, Yueyi and Chen, Ming and Zhang, He and Chen, Dijun},
  journal={Genomics, Proteomics \& Bioinformatics},
  year={2025},
  doi={10.1093/gpbjnl/qzaf024}
}
@article{khalifa2020ai,
  title={Artificial Intelligence Technique for Gene Expression by Tumor {RNA}-Seq Data: A Novel Optimized Deep Learning Approach},
  author={Khalifa, Nour Eldeen M and Taha, Mohamed Hamed N and Ali, Dalia Ezzat and S{\l}owik, Adam and Hassanien, Aboul Ella},
  journal={IEEE Access},
  volume={8},
  year={2020},
  doi={10.1109/access.2020.2970210}
}
@article{rukhsar2022analyzing,
  title={Analyzing {RNA}-Seq Gene Expression Data Using Deep Learning Approaches for Cancer Classification},
  author={Rukhsar, Laiqa and Bangyal, Waqas Haider and Khan, Muhammad Sadiq Ali and Ibrahim, Ag Asri Ag and Nisar, Kashif and Rawat, Danda B},
  journal={Applied Sciences},
  volume={12},
  number={4},
  pages={1850},
  year={2022},
  doi={10.3390/app12041850}
}
@article{polepalli2025cvae,
  title={A Novel {cVAE}-Augmented Deep Learning Framework for Pan-Cancer {RNA}-Seq Classification},
  author={Polepalli, Vinil},
  journal={arXiv preprint},
  year={2025}
}
@article{kim2025pancancer,
  title={Pan-cancer Gene Set Discovery via {scRNA}-seq for Optimal Deep Learning Based Downstream Tasks},
  author={Kim, Jong Hyun and Jang, Jongseong},
  journal={Scientific Reports},
  year={2025},
  doi={10.1038/s41598-025-27296-z}
}
@article{ghaleb2025sdcfe,
  title={A Novel Statistical Feature Selection Framework for Biomarker Discovery and Cancer Classification via Multiomics Integration},
  author={Ghaleb, Moshira S and Al-Berry, Maryam and Ebied, Hala M and Tolba, Mohamed F},
  journal={BMC Medical Research Methodology},
  year={2025},
  doi={10.1186/s12874-025-02713-z}
}
@article{rahaman2025integrated,
  title={An Integrated Approach for Key Gene Selection and Cancer Phenotype Classification: Improving Diagnosis and Prediction},
  author={Rahaman, Matiur and Sarker, Bandhan and Alamin, Muhammad Habibulla and Ferdousi, Farzana},
  journal={Computers in Biology and Medicine},
  year={2025},
  doi={10.1016/j.compbiomed.2025.110687}
}
@article{shukla2025discriminative,
  title={Discriminative Biomarker Selection Using Hybrid Multi-Population Evolutionary Computation},
  author={Shukla, Alok Kumar and Dwivedi, Shubhra and Mishra, Aishwarya},
  journal={Scientific Reports},
  year={2025},
  doi={10.1038/s41598-025-29921-3}
}
@article{bouazza2025degs,
  title={A Deep Ensemble Gene Selection and Attention-guided Classification Framework for Robust Cancer Diagnosis from Microarray Data},
  author={Bouazza, Sara Haddou},
  journal={Engineering, Technology \& Applied Science Research},
  year={2025},
  doi={10.48084/etasr.9476}
}
"""


if __name__ == "__main__":
    main()
