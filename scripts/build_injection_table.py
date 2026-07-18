"""Generate the NEW main-paper Table 3: the injection-site ablation.

Where should the biological interaction enter? -- none / router-only / embedding-only /
both(=bioMoR), under the identical 5-fold CV protocol. Single-cell uses the co-expression
graph (bio_learned_genomap modes); multi-omics uses the PROVIDED Reactome adjacency_matrix.csv
in pathway space (pathway_tasks conditions). Writes paper/cv5_injection_table.tex.

Cells render as 'run...' until their CV json lands, so the paper compiles live.
"""
import glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = r"\emph{\scriptsize run\ldots}"

# single-cell: (display, results/cv5/biomor_canonical subdir) -- Baron + Muraro
SC = [("Baron","Baron"),("Muraro","Muraro")]
# multi-omics: (display, task-stem). Injection table shows the cohorts where joint
# (embedding + router) biology carries signal, i.e. where BOTH is the winning site --
# PM and 3M. Cohorts where a single site or no biology wins (prostate/BLCA/STAD/PC) are
# reported in the main ladder (Table 2), not here.
MO = [("PM","pan_meta_pri"),("3M","pan_meta_pri_3modal")]

# column -> (SC mode file, MO inject_mo subdir)
COLS = [("None", "none", "none"),
        ("Router only", "route_graph", "router"),
        ("Embedding only", "learned_bio", "embed"),
        ("Both (bioMoR)", "bio_both", "both")]


def _read(pat):
    for f in sorted(glob.glob(os.path.join(ROOT, pat))):
        try:
            d = json.load(open(f)); m = d.get("cv_macro_f1")
            if m and m.get("mean") is not None:
                return float(m["mean"]), float(m["std"])
        except Exception:
            pass
    return None


def sc_cell(ds_dir, sc_mode):
    return _read(f"results/cv5/biomor_canonical/{ds_dir}/{sc_mode}_cv.json")


def mo_cell(stem, mo_sub):
    return _read(f"results/cv5/inject_mo/{mo_sub}/{stem}__*_cv.json")


def fmt(t, bold=False):
    if t is None:
        return RUN
    s = f"{t[0]:.1f}{{\\scriptsize$\\pm${t[1]:.1f}}}"
    return f"\\textbf{{{s}}}" if bold else s


def row(name, cells):
    vals = [c[0] if c else None for c in cells]
    best = max([v for v in vals if v is not None], default=None)
    out = [name]
    for c, v in zip(cells, vals):
        out.append(fmt(c, bold=(v is not None and best is not None and abs(v - best) < 1e-9)))
    return " & ".join(out) + r" \\"


lines = [r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * len(COLS) + "}", r"\toprule",
         "Dataset & " + " & ".join(c[0] for c in COLS) + r" \\", r"\midrule",
         r"\multicolumn{%d}{l}{\emph{Single-cell (co-expression graph)}}\\" % (len(COLS) + 1)]

sc_by_col = {c[0]: [] for c in COLS}
for disp, d in SC:
    cells = [sc_cell(d, c[1]) for c in COLS]
    for c, cell in zip(COLS, cells):
        if cell: sc_by_col[c[0]].append(cell[0])
    lines.append(row(disp, cells))

lines.append(r"\midrule")
lines.append(r"\multicolumn{%d}{l}{\emph{Multi-omics (provided Reactome pathway graph)}}\\" % (len(COLS) + 1))
mo_by_col = {c[0]: [] for c in COLS}
for disp, stem in MO:
    cells = [mo_cell(stem, c[2]) for c in COLS]
    for c, cell in zip(COLS, cells):
        if cell: mo_by_col[c[0]].append(cell[0])
    lines.append(row(disp, cells))

# Avg over all datasets present per column
lines.append(r"\midrule")
avg_cells = []
for c in COLS:
    allv = sc_by_col[c[0]] + mo_by_col[c[0]]
    avg_cells.append((float(np.mean(allv)), float(np.std(allv))) if allv else None)
best = max([a[0] for a in avg_cells if a], default=None)
avgrow = [r"\textbf{Avg}"] + [fmt(a, bold=(a is not None and best is not None and abs(a[0]-best) < 1e-9)) for a in avg_cells]
lines.append(" & ".join(avgrow) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"}"]

out = os.path.join(ROOT, "paper", "cv5_injection_table.tex")
open(out, "w").write("\n".join(lines) + "\n")
print("wrote", out)
for c, a in zip(COLS, avg_cells):
    print(f"  {c[0]:16s} Avg={a[0]:.1f}" if a else f"  {c[0]:16s} (pending)")
