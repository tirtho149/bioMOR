# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
# Reproduction of ALL 14 Mixture-of-Recursions (Bae et al. 2025) tables with SMART
# on the genomap dataset suite (Islam & Xing 2023). NO TCGA.
#
# Each table is rendered from the result directories produced by the run_*.sbatch
# jobs; cells with no result yet show "--" (never fabricated). Run any time to get
# the current state of the 14-table reproduction:
#     python -m recursive_marker_transformer.mor_tables
# writes paper/mor_tables.md and prints a coverage summary.
# ============================================================================
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENOMAP = ["tabula_muris", "pancreas", "common_class", "prototype", "baron", "segerstolpe",
           "lung", "oesophagus", "spleen", "tcell"]


def _metric(path: Path, key: str = "macro_f1"):
    """Read a singlecell/pancan result JSON -> (metric%, transformer_params, total_params)."""
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    heads = d.get("heads")
    if heads:                                            # singlecell schema
        h = heads.get("cell_type") or next(iter(heads.values()))
        m = h.get(key)
    else:                                                # pancan/pathway schema
        m = d.get(key)
    if m is None:
        return None
    return (100 * m, d.get("transformer_params"), d.get("total_params"))


def _seed_mean(variant_dir: str, ds: str, key="macro_f1"):
    """Mean+/-std over seed subdirs results_singlecell_arch/<variant>/s*/<ds>.json."""
    vals = []
    tp = None
    for f in glob.glob(str(ROOT / variant_dir / "s*" / f"{ds}.json")):
        r = _metric(Path(f), key)
        if r:
            vals.append(r[0]); tp = r[1]
    if not vals:
        return None
    return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0, tp, len(vals))


def _cell(v):
    return "--" if v is None else (f"{v[0]:.1f}" if isinstance(v, tuple) else f"{v:.1f}")


FMT = "md"   # "md" | "tex"; main() renders both


import re as _re


def _tex_escape(s):
    s = str(s)
    for a, b in [("±", r"$\pm$"), ("_", r"\_"), ("%", r"\%"), ("#", r"\#")]:
        s = s.replace(a, b)
    s = _re.sub(r"(\d)x\b", r"\1$\\times$", s)            # "6x" -> 6x; leave words alone
    return s


def _md_table(header, rows):
    if FMT == "tex":
        cols = "l" + "r" * (len(header) - 1)
        out = [r"\begin{tabular}{" + cols + "}", r"\toprule",
               " & ".join(_tex_escape(h) for h in header) + r" \\", r"\midrule"]
        for r in rows:
            out.append(" & ".join(_tex_escape(c) for c in r) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(out)
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _present(ds):
    return [d for d in ds if (ROOT / "data" / "singlecell" / d).exists()]


# GitHub pathway/P-NET cohorts reported alongside the genomap datasets in every table.
PATHWAY_COH = ["prostate", "blca", "stad", "panmeta_subtype"]


def _pw(subdir: str, variant: str, task: str, key: str = "macro_f1"):
    """Mean macro-F1 (%) for a pathway cohort from results_pathway_<subdir>/<variant>/<task>__*.json."""
    vals = []
    for f in glob.glob(str(ROOT / f"results_pathway_{subdir}" / variant / f"{task}__*.json")):
        d = json.loads(Path(f).read_text())
        if d.get(key) is not None:
            vals.append(100 * d[key])
    return st.mean(vals) if vals else None


def _cols_all():
    """Column order used by every dataset-as-column table: genomap then pathway."""
    return _present(GENOMAP) + PATHWAY_COH


def _pw_warm(task: str, key: str):
    """Warm-start field for a pathway cohort: results_pathway_warmstart/<task>.json."""
    p = ROOT / "results_pathway_warmstart" / f"{task}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(key)


def _pw_depth(task: str):
    """Adaptive-depth stats for a cohort from its expert-choice (shared) arch run:
    (mean_recursion_depth, K, compute_saving_ratio)."""
    hits = glob.glob(str(ROOT / "results_pathway_arch" / "shared" / f"{task}__*.json"))
    if not hits:
        return None
    d = json.loads(Path(hits[0]).read_text())
    K = d.get("config", {}).get("recursion_depth", 4)
    return (d.get("mean_recursion_depth"), K, d.get("compute_saving_ratio"))


# ----------------------------------------------------------------------------- tables
def t_adaptive_depth():
    """ADAPTIVE-DEPTH table (MoR's core claim): fixed-depth recursion vs adaptive
    per-token routed depth (expert-choice MoR). Accuracy from results_singlecell_arch,
    mean per-token depth + active-fraction-per-step (=> compute saved) from results_depth.
    """
    ds = _present(GENOMAP)
    rows = []
    for d in ds:
        fx = _seed_mean("results_singlecell_arch/fixed", d)
        ad = _seed_mean("results_singlecell_arch/shared", d)     # expert-choice = adaptive
        k1 = _seed_mean("results_singlecell_arch/depth1", d)     # K=1, no recursion
        depth = ROOT / "results_depth" / f"{d}.json"
        mdepth, saved = "--", "--"
        if depth.exists():
            r = json.loads(depth.read_text())
            K = r.get("recursion_depth", 4)
            mdepth = f"{r.get('mean_token_depth', 0):.2f}/{K}"
            af = r.get("active_fraction_per_step", [])
            if af:
                saved = f"{100*(1 - sum(af)/len(af)):.0f}%"        # avg idle fraction
        rows.append([d,
                     "--" if k1 is None else f"{k1[0]:.1f}",
                     "--" if fx is None else f"{fx[0]:.1f}",
                     "--" if ad is None else f"{ad[0]:.1f}",
                     mdepth, saved])
    for t in PATHWAY_COH:                 # pathway/P-NET cohorts (arch ablation + depth run)
        k1 = _pw("arch", "depth1", t)
        fx = _pw("arch", "fixed", t)
        ad = _pw("arch", "shared", t)
        dp = _pw_depth(t)
        mdepth = saved = "--"
        if dp and dp[0] is not None:
            mdepth = f"{dp[0]:.2f}/{dp[1]}"
            if dp[2] is not None:
                saved = f"{100*(1 - dp[2]):.0f}%"
        rows.append([t,
                     "--" if k1 is None else f"{k1:.1f}",
                     "--" if fx is None else f"{fx:.1f}",
                     "--" if ad is None else f"{ad:.1f}",
                     mdepth, saved])
    return _md_table(["Dataset", "K=1 (no recursion)", "fixed depth K",
                      "adaptive depth (MoR)", "mean token depth", "compute saved"], rows)


def t4_router_ablation():
    """T4: expert/token-choice router + selection ablation (results_singlecell_arch)."""
    variants = [("shared", "expert-choice MoR (shared)"), ("token", "token-choice MoR"),
                ("fixed", "fixed-depth (no routing)"), ("depth1", "K=1 (no recursion)"),
                ("independent", "independent blocks"),
                ("marker_random", "random markers"), ("marker_var", "variance markers")]
    ds = _present(GENOMAP)
    rows = []
    for v, label in variants:
        cells = []
        for d in ds:
            r = _seed_mean(f"results_singlecell_arch/{v}", d)
            cells.append("--" if r is None else f"{r[0]:.1f}±{r[1]:.1f}")
        rows.append([label] + cells)
    return _md_table(["Variant (macro-F1)"] + ds, rows)


def t3_arch_scaling(key="macro_f1"):
    """T3/T7: MoR vs Recursive vs Vanilla across model sizes (results_scaling)."""
    archs = [("vanilla", "Vanilla (independent)"), ("recursive", "Recursive (shared)"),
             ("mor", "MoR (SMART)")]
    # shared size rungs: single-cell d in {48,96,192,384}, the larger cohorts in
    # {64,128,256} (no cohort xlarge run). One row per (arch, rung); each column reads
    # its own modality's d for that rung.
    rungs = [("small", 48, 64), ("medium", 96, 128), ("large", 192, 256), ("xlarge", 384, None)]
    ds = _present(GENOMAP)
    rows = []
    for a, label in archs:
        for rlab, Dsc, Dco in rungs:
            cells = [_cell(_metric(ROOT / f"results_scaling/{a}_d{Dsc}" / f"{d}.json", key))
                     for d in ds]
            for t in PATHWAY_COH:
                v = _pw("scaling", f"{a}_d{Dco}", t, key) if Dco else None
                cells.append(f"{v:.1f}" if v is not None else "TODO")
            rows.append([f"{label} {rlab}"] + cells)
    return _md_table([f"Arch x size ({key})"] + ds + PATHWAY_COH, rows)


def t6_config():
    """T6: parameter counts of the size variants (transformer params, from results_scaling)."""
    sizes = [48, 96, 192, 384]
    rows = []
    for D in sizes:
        r = _metric(ROOT / f"results_scaling/mor_d{D}" / "tabula_muris.json")
        van = _metric(ROOT / f"results_scaling/vanilla_d{D}" / "tabula_muris.json")
        tp = "--" if not r else f"{r[1]:,}"
        vtp = "--" if not van else f"{van[1]:,}"
        ratio = "--" if (not r or not van or not r[1]) else f"{van[1]/r[1]:.1f}x"
        rows.append([f"d_model={D}", "K=4", tp, vtp, ratio])
    return _md_table(["Size", "Recursions", "MoR params", "Vanilla params", "param reduction"], rows)


def t158_sharing():
    """T1/T5/T8: parameter-sharing schemes (results_sharing)."""
    tags = [("shared_nu1", "shared (1 block)"), ("cycle_nu3", "Cycle (3)"),
            ("sequence_nu3", "Sequence (3)"), ("middle_cycle_nu3", "Middle-Cycle (3)"),
            ("middle_sequence_nu3", "Middle-Sequence (3)"),
            ("independent_nu6", "independent (6)")]
    gds = _present(GENOMAP)
    rows = []
    for tag, label in tags:
        cells = [_cell(_metric(ROOT / f"results_sharing/{tag}" / f"{d}.json")) for d in gds]
        for t in PATHWAY_COH:
            v = _pw("sharing", tag, t)
            cells.append(f"{v:.1f}" if v is not None else "--")
        rows.append([label] + cells)
    return _md_table(["Sharing scheme (K=6)"] + gds + PATHWAY_COH, rows)


def t2_12_13_stepcache():
    """T2/T12/T13: step-cache (reuse vs recompute step-1 K/V) (results_stepcache)."""
    gds = _present(GENOMAP)
    rows = []
    for tag, label in [("off", "recompute K/V (no cache)"), ("on", "reuse step-1 K/V (step-cache)")]:
        cells = [_cell(_metric(ROOT / f"results_stepcache/{tag}" / f"{d}.json")) for d in gds]
        cells += [(lambda v: f"{v:.1f}" if v is not None else "--")(_pw("stepcache", tag, t))
                  for t in PATHWAY_COH]
        rows.append([label] + cells)
    return _md_table(["KV strategy (macro-F1)"] + gds + PATHWAY_COH, rows)


def t_pathway_arch():
    """GitHub pathway/P-NET cohorts through the architecture ablation
    (results_pathway_arch/<variant>/<task>__*.json) -- the 3 claims on the new data."""
    tasks = ["prostate", "blca", "stad", "brca", "panmeta_response", "panmeta_subtype"]
    variants = [("shared", "adaptive MoR (shared)"), ("independent", "independent"),
                ("token", "token-choice"), ("fixed", "fixed-depth"), ("depth1", "K=1")]
    rows = []
    for v, label in variants:
        cells = []
        for tk in tasks:
            hits = glob.glob(str(ROOT / "results_pathway_arch" / v / f"{tk}__*.json"))
            r = _metric(Path(hits[0])) if hits else None
            cells.append(_cell(r))
        rows.append([label] + cells)
    return _md_table(["Pathway cohort (macro-F1)"] + tasks, rows)


def t_biorouter():
    """Biology-informed routing: co-expression centrality prior on the recursion
    router vs no prior and a degree-matched random-graph control, all datasets."""
    ds = _present(GENOMAP)
    rows = []
    for m, lab in [("none", "no prior"), ("coexpr", "co-expression prior"),
                   ("random", "random-graph control")]:
        cells = []
        for d in ds:
            xs = [json.loads(Path(f).read_text())["macro_f1"] * 100
                  for f in glob.glob(str(ROOT / "results_sc_interaction" / f"{d}__{m}__seed*.json"))]
            cells.append(f"{st.mean(xs):.1f}±{st.pstdev(xs):.1f}" if xs else "--")
        rows.append([lab] + cells)
    return _md_table(["Biology prior (macro-F1)"] + ds, rows)


def t_token_reduction():
    """Token reduction: macro-F1 as a function of the number of marker tokens M
    (results_msweep/M<n>/<ds>.json) -- how few interpretable tokens suffice."""
    Ms = [16, 32, 64, 128, 256]
    ds = _present(GENOMAP)
    rows = []
    for d in ds:
        cells = [_cell(_metric(ROOT / f"results_msweep/M{m}" / f"{d}.json")) for m in Ms]
        rows.append([d] + cells)
    for t in PATHWAY_COH:                 # pathway/P-NET cohorts (learned-token M-sweep)
        cells = [(lambda v: f"{v:.1f}" if v is not None else "TODO")(_pw("msweep", f"M{m}", t))
                 for m in Ms]
        rows.append([t] + cells)
    return _md_table(["Dataset / #tokens M"] + [f"M={m}" for m in Ms], rows)


def t9_warmstart():
    """T9: warm-start (uptraining analogue) -- results_warmstart/<ds>.json."""
    ds = _present(GENOMAP)
    rows = []
    for label, key in [("fixed-depth source", "fixed_source_macro_f1"),
                       ("MoR from scratch", "mor_from_scratch_macro_f1"),
                       ("MoR warm-started", "mor_warm_start_macro_f1"),
                       ("warm-start gain", "warm_start_gain")]:
        cells = []
        for d in ds:
            p = ROOT / "results_warmstart" / f"{d}.json"
            if p.exists():
                v = json.loads(p.read_text()).get(key)
                cells.append(f"{v:+.1f}" if key == "warm_start_gain" else f"{v:.1f}")
            else:
                cells.append("--")
        for t in PATHWAY_COH:             # pathway/P-NET cohorts (bulk warm-start)
            v = _pw_warm(t, key)
            if v is None:
                cells.append("TODO")
            else:
                cells.append(f"{v:+.1f}" if key == "warm_start_gain" else f"{v:.1f}")
        rows.append([label] + cells)
    return _md_table(["Warm-start (macro-F1)"] + ds + PATHWAY_COH, rows)


def t1011_routing():
    """T10/T11: expert + token router under different routing configs."""
    tags = [("exp_linear", "expert linear"), ("exp_mlp", "expert MLP"),
            ("exp_temp2", "expert temp=2"), ("tok_linear", "token linear"),
            ("tok_mlp", "token MLP"), ("tok_bal01", "token balance=0.01")]
    ds = _present(GENOMAP)
    rows = []
    for tag, label in tags:
        cells = [_cell(_metric(ROOT / f"results_routing/{tag}" / f"{d}.json")) for d in ds]
        for t in PATHWAY_COH:             # same router configs on the pathway cohorts
            v = _pw("routing", tag, t)
            cells.append(f"{v:.1f}" if v is not None else "TODO")
        rows.append([label] + cells)
    return _md_table(["Routing config (macro-F1)"] + ds + PATHWAY_COH, rows)


def t14_depth():
    """T14 / Fig5: per-marker recursion depth + active fraction per step."""
    ds = _present(GENOMAP)
    rows = []
    for d in ds:
        p = ROOT / "results_depth" / f"{d}.json"
        if not p.exists():
            rows.append([d, "--", "--"]); continue
        r = json.loads(p.read_text())
        af = ", ".join(f"{x:.2f}" for x in r.get("active_fraction_per_step", []))
        rows.append([d, f"{r.get('mean_token_depth', 0):.2f}/{r.get('recursion_depth')}", af])
    for t in PATHWAY_COH:                 # pathway/P-NET cohorts (depth from the adaptive run)
        r = _pw_depth(t)
        if r is None or r[0] is None:
            rows.append([t, "TODO", "TODO"]); continue
        md, K, saving = r
        rows.append([t, f"{md:.2f}/{K}",
                     f"compute {saving:.2f}x (saving)" if saving is not None else "--"])
    return _md_table(["Dataset", "mean token depth", "active fraction per step (1..K)"], rows)


PENDING = {}


# Story-named tables (title, one-line caption, builder, label). No "T1/T2/T3" -- each
# table is a result of the paper, ordered to tell the SMART story: token reduction ->
# adaptive recursion -> parameter sharing -> compute allocation -> transfer.
SECTIONS = [
    # Only design dimensions NOT already in the main-paper tables (which now report
    # all 10 genomap datasets + the pathway/P-NET cohorts). No redundancy.
    ("How few tokens suffice",
     "macro-F1 as the number of marker tokens $M$ is reduced (token reduction)",
     t_token_reduction, "tab:tokens"),
    ("Recursion versus independent layers across model sizes",
     "macro-F1 of independent layers (Vanilla), one weight-shared block (Recursive), and adaptive routing (SMART)",
     t3_arch_scaling, "tab:scaling"),
    ("Weight-sharing schemes",
     "Cycle, Sequence, Middle-Cycle and Middle-Sequence sharing between the fully-shared and fully-independent extremes",
     t158_sharing, "tab:sharing"),
    ("Where computation is spent",
     "mean recursion depth per marker token and the fraction of tokens still active at each step",
     t14_depth, "tab:depth"),
    ("Key/value reuse across recursions",
     "recomputing vs reusing the first-step attention keys/values across recursion steps",
     t2_12_13_stepcache, "tab:cache"),
    ("Warm-starting recursion from a fixed-depth model",
     "initialising the shared block from a trained fixed-depth model vs training from scratch",
     t9_warmstart, "tab:warmstart"),
    ("Routing configurations",
     "router head, temperature and load balancing for the expert- and token-choice routers",
     t1011_routing, "tab:routing"),
]

# Story description paragraph per table (keyed by title). Prose, not "MoR table N".
DESC = {
    "How few tokens suffice":
        "Token reduction. SMART keeps only $M$ interpretable tokens, so attention costs "
        "$O(M^2)$ rather than $O(N^2)$ in the number of genes. Accuracy is reported as $M$ "
        "shrinks; a few dozen to a few hundred tokens recover most of the full-gene signal.",
    "Routing and marker-token selection":
        "Two questions about the selective recursion. Expert- vs token-choice routing "
        "allocate depth differently; and learning which genes are markers (the cross-attention "
        "router) is compared against fixed random and variance-ranked panels.",
    "Biology-informed routing":
        "SMART's headline component: a label-free prior from the gene-gene co-expression "
        "graph (network centrality) is added to the recursion-depth router, nudging "
        "co-expression hubs to recurse deeper. We compare it against no prior and a "
        "degree-matched random-graph control across all datasets; gains are small and "
        "mostly within noise, so we report it as a stabiliser rather than an accuracy driver.",
    "Adaptive recursion depth":
        "The adaptive-loop result. A single pass ($K{=}1$), fixed-depth recursion, and "
        "adaptive per-token routed depth are compared; the mean token depth and compute saved "
        "quantify the funnel -- uninformative tokens exit early while accuracy tracks fixed depth.",
    "Recursion versus independent layers across model sizes":
        "Recursion vs independent depth across model sizes: independent layers (Vanilla), one "
        "weight-shared block applied $K$ times (Recursive), and adaptive routing (SMART). "
        "Recursive matches independent at a fraction of the parameters.",
    "Parameter reduction from weight sharing":
        "The parameter-reduction result. Applying one block recursively uses $1/K$ of the "
        "parameters of $K$ independent blocks of the same width (a $4\\times$ reduction at $K{=}4$).",
    "Weight-sharing schemes":
        "Between fully shared and fully independent lie graded sharing schemes -- Cycle, "
        "Sequence, Middle-Cycle and Middle-Sequence -- trading parameters for capacity.",
    "Where computation is spent":
        "An interpretability view of the adaptive loop: the mean recursion depth assigned to "
        "each marker token and the fraction of tokens still active at each recursion step.",
    "Key/value reuse across recursions":
        "Whether the first recursion's attention keys/values can be reused across later steps "
        "(a cache) instead of recomputed, trading a little accuracy for compute.",
    "Warm-starting recursion from a fixed-depth model":
        "Whether a recursive model benefits from initialising its shared block from a trained "
        "fixed-depth model and continuing training, versus training from scratch.",
    "Routing configurations":
        "How the routing behaves under different router heads (linear vs MLP), temperatures "
        "and load-balancing strengths, for both expert- and token-choice routing.",
    "Transfer to pathway-informed multi-omics":
        "Whether the same design decisions hold beyond single cells: the architecture ablation "
        "rerun on Reactome/P-NET mutation/CNV/expression cohorts.",
}

FIGURES = [
    ("fig_scaling.png", "Macro-F1 versus model size for independent (Vanilla), weight-shared "
                        "(Recursive) and adaptively-routed (SMART) stacks on each dataset.", "fig:scaling"),
    ("fig_depth.png", "Adaptive recursion depth across the genomap datasets: macro-F1 under "
                      "no recursion (K=1), fixed depth, and adaptive routed depth.", "fig:depth"),
    ("fig_param_efficiency.png", "Accuracy versus transformer parameters for weight-shared "
                                 "recursion against independent layers.", "fig:param"),
]


def main():
    global FMT
    out_dir = ROOT / "paper"
    out_dir.mkdir(exist_ok=True)

    def render(fmt):
        global FMT
        FMT = fmt
        return [(t, c, fn(), lab) for t, c, fn, lab in SECTIONS]

    # ---- markdown ----
    md = ["# SMART: experiments behind the three claims (genomap + pathway, no TCGA)\n",
          "macro-F1 (mean±std); `--` = job still running.\n"]
    for t, c, body, lab in render("md"):
        md.append(f"## {t} — {c}\n{body}\n")
    (out_dir / "mor_tables.md").write_text("\n".join(md) + "\n")

    # ---- latex (\input-ready Extended Results) ----
    # Narrow tables are single-column [!htbp] floats so they pack 2-3 per page next to
    # their describing paragraph; only genuinely-wide tables (>=10 cols) span both AAAI
    # columns as table*[t]. Every float is cited in its paragraph -> no orphan floats,
    # no float-only pages with stranded whitespace.
    WIDE = {"tab:sharing", "tab:cache"}  # >=10 data columns: must span both columns
    tex = [r"% Auto-generated by recursive_marker_transformer.mor_tables -- do not edit."]
    for t, c, body, lab in render("tex"):
        elab = lab.replace("tab:", "tab:exp-")  # avoid collision w/ main paper
        tex.append(r"\paragraph{" + t + ".} " + DESC.get(t, c) +
                   r" (Table~\ref{" + elab + "}).")
        if lab in WIDE:
            tex.append(r"\begin{table*}[t]\centering\footnotesize")
            tex.append(r"\setlength{\tabcolsep}{5pt}")
            tex.append(r"\caption{" + c[0].upper() + c[1:] + ".}")
            tex.append(r"\label{" + elab + "}")
            tex.append(r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%")
        else:
            tex.append(r"\begin{table}[H]\centering\footnotesize")
            tex.append(r"\setlength{\tabcolsep}{4pt}")
            tex.append(r"\caption{" + c[0].upper() + c[1:] + ".}")
            tex.append(r"\label{" + elab + "}")
            tex.append(r"\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{%")
        tex.append(body)
        tex.append(r"}")
        tex.append(r"\end{table*}" if lab in WIDE else r"\end{table}")
    # figures: one citing paragraph, then single-column floats that pack with the text
    figs = [(img, cap, lab) for img, cap, lab in FIGURES if (out_dir / "figs" / img).exists()]
    if figs:
        joined = (", ".join(r"Fig.~\ref{" + f[2] + "}" for f in figs[:-1])
                  + (" and " if len(figs) > 1 else "")
                  + r"Fig.~\ref{" + figs[-1][2] + "}")
        tex.append(r"\paragraph{Visual summary.} The same trends are shown graphically -- "
                   "scaling behaviour, adaptive recursion depth and parameter efficiency "
                   "across the genomap suite (" + joined + ").")
        for img, cap, lab in figs:
            tex.append(r"\begin{figure}[H]\centering")
            tex.append(r"\includegraphics[width=\columnwidth]{figs/" + img + "}")
            tex.append(r"\caption{" + cap + "}")
            tex.append(r"\label{" + lab + "}")
            tex.append(r"\end{figure}")
    (out_dir / "mor_tables.tex").write_text("\n".join(tex) + "\n")
    FMT = "md"

    print("\n".join(md))
    print(f"\n[mor_tables] wrote {out_dir/'mor_tables.md'} and {out_dir/'mor_tables.tex'}")


if __name__ == "__main__":
    main()
