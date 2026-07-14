#!/usr/bin/env python3
"""
PaperVizAgent, re-backed by Claude headless (`claude -p`).

Same reference-driven multi-agent pipeline as google-research/papervizagent
(Retriever -> Planner -> Stylist -> Visualizer -> Critic loop), but:
  * every LLM agent runs on the local Claude Code CLI in headless mode
    (`claude -p`), so NO Google/OpenAI/Anthropic API key is needed; and
  * the Visualizer emits a COMPILABLE standalone TikZ document instead of
    calling a raster image model. We render it with pdflatex + PyMuPDF (fitz),
    which lets the Critic actually *see* the figure and iterate.

Output: teaser_smart.pdf / teaser_smart.png plus a full per-run trace.
"""
import subprocess, sys, re, shutil, time, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
RUN.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- backend
def claude(prompt: str, system: str | None = None, image: Path | None = None,
           timeout: int = 300, label: str = "") -> str:
    """One headless Claude call. Prompt goes on stdin; image (if any) is read
    by the Read tool from an absolute path embedded in the prompt."""
    cmd = ["claude", "-p"]
    if system:
        cmd += ["--append-system-prompt", system]
    if image is not None:
        prompt = f"First, use the Read tool to view the image at: {image}\n\n" + prompt
        cmd += ["--allowedTools", "Read"]
    t0 = time.time()
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    out = r.stdout.strip()
    print(f"    [{label}] {dt:4.0f}s  {len(out)} chars"
          + ("" if r.returncode == 0 else f"  (rc={r.returncode})"))
    if r.returncode != 0 and not out:
        raise RuntimeError(f"claude call failed ({label}): {r.stderr[:500]}")
    return out

# ----------------------------------------------------------------------------- helpers
def extract_tikz(text: str) -> str:
    """Pull the standalone LaTeX/TikZ document out of a model reply."""
    blocks = re.findall(r"```(?:latex|tex|tikz)?\s*\n(.*?)```", text, re.DOTALL)
    cands = [b for b in blocks if "\\documentclass" in b] or \
            [b for b in blocks if "tikzpicture" in b] or blocks
    code = (cands[0] if cands else text).strip()
    if "\\documentclass" not in code:  # wrap a bare picture just in case
        code = ("\\documentclass[tikz,border=8pt]{standalone}\n"
                "\\usepackage{amsmath,amssymb}\n"
                "\\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc,shapes.geometric}\n"
                "\\begin{document}\n" + code + "\n\\end{document}\n")
    return code

def compile_tikz(code: str, stem: str) -> tuple[bool, Path | None, str]:
    """pdflatex -> PDF; returns (ok, png_path, error_excerpt)."""
    d = RUN / stem
    d.mkdir(exist_ok=True)
    tex = d / f"{stem}.tex"
    tex.write_text(code)
    err = ""
    for _ in range(2):  # two passes for positioning/fit
        p = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            f"{stem}.tex"], cwd=d, capture_output=True, text=True, timeout=180)
    pdf = d / f"{stem}.pdf"
    if not pdf.exists():
        log = (d / f"{stem}.log")
        txt = log.read_text(errors="ignore") if log.exists() else p.stdout
        m = re.findall(r"^(!.*(?:\n.*){0,3})", txt, re.MULTILINE)
        err = "\n".join(m[:4])[:1500] or txt[-1200:]
        return False, None, err
    import fitz
    png = d / f"{stem}.png"
    pix = fitz.open(pdf)[0].get_pixmap(matrix=fitz.Matrix(4, 4))
    pix.save(png)
    return True, png, ""

# ----------------------------------------------------------------------------- agents
RETRIEVER_SYS = "You are the Retriever agent of an academic-figure pipeline. You summarize the visual identity a figure must match. Be concise and concrete."
PLANNER_SYS   = "You are the Planner agent. You turn a paper's method + a figure caption into a precise, unambiguous spatial description of ONE publication-quality diagram: every box, its label, position, every arrow and its meaning, and the single headline message. You never write code."
STYLIST_SYS   = "You are the Stylist agent. You refine a diagram description to meet top-tier ML venue (AAAI/NeurIPS) aesthetic standards: clean grid alignment, restrained palette, generous whitespace, legible type, no clutter or overlap. You keep it faithful to the plan and realizable as vector TikZ."
VIS_SYS       = "You are the Visualizer agent. You output ONE complete, COMPILABLE standalone LaTeX/TikZ document and nothing else. It must compile with pdflatex using only: standalone, amsmath, amssymb, tikz, xcolor, and tikzlibraries arrows.meta,positioning,fit,backgrounds,calc,shapes.geometric. No external files, no fontawesome, no includegraphics. Keep it landscape, publication-clean, no figure title inside the image."
CRITIC_SYS    = "You are the Critic agent. You look at the RENDERED figure and judge it against the plan: factual fidelity to the paper, clarity of the headline message, alignment/overlap/legibility, and aesthetic quality. You are specific and demanding."

def retriever(paper: str) -> str:
    return claude(
        f"From the method notes below, extract a 6-8 line STYLE REFERENCE for the teaser: "
        f"the color identity, panel scheme, box/arrow style, and which single element is the visual hero.\n\n{paper}",
        system=RETRIEVER_SYS, label="retriever", timeout=180)

def planner(paper: str, caption: str, style_ref: str) -> str:
    return claude(
        f"METHOD NOTES:\n{paper}\n\nTARGET CAPTION / INTENT:\n{caption}\n\n"
        f"STYLE REFERENCE:\n{style_ref}\n\n"
        f"Produce a detailed spatial PLAN for one landscape teaser diagram: list every node "
        f"(id, label text incl. math, panel, approx. row/col position), every arrow "
        f"(from->to, label, meaning), the left architecture flow, the right results bar chart "
        f"with the exact numbers (random ~57.5 vs LEARNED 70.8, +11.7; fixed = flat negative) "
        f"and the two efficiency badges. State the headline message in one sentence.",
        system=PLANNER_SYS, label="planner", timeout=300)

def stylist(plan: str, style_ref: str) -> str:
    return claude(
        f"PLAN:\n{plan}\n\nSTYLE REFERENCE:\n{style_ref}\n\n"
        f"Rewrite as a tightened, aesthetically-refined SPEC ready for a TikZ engineer: fix a "
        f"concrete grid layout and spacing, the exact green/blue palette (accentA=2E7D5B, "
        f"accentB=2F5BAA, panel fills EAF2EA / EAF0FA, edges 6B7280), font sizes, and callouts. "
        f"Guarantee no overlaps and that the learned-gene-graph 'smooth/denoise' arrow reads as the hero.",
        system=STYLIST_SYS, label="stylist", timeout=300)

def visualizer(spec: str, prior_code: str | None, feedback: str | None) -> str:
    if prior_code is None:
        p = (f"Build the teaser as a standalone TikZ document per this SPEC:\n\n{spec}\n\n"
             f"Output ONLY the LaTeX document in a single ```latex code block.")
    else:
        p = (f"Here is the current TikZ document:\n```latex\n{prior_code}\n```\n\n"
             f"Revise it to address this feedback, changing only what is needed and keeping it "
             f"compilable:\n{feedback}\n\nOutput ONLY the full revised LaTeX in one ```latex block.")
    return claude(p, system=VIS_SYS, label="visualizer", timeout=420)

def fix_compile(code: str, error: str) -> str:
    return claude(
        f"This standalone TikZ document FAILED to compile with pdflatex.\n\nERROR:\n{error}\n\n"
        f"DOCUMENT:\n```latex\n{code}\n```\n\n"
        f"Return the corrected full document in one ```latex block. Use only the allowed packages "
        f"(standalone, amsmath, amssymb, tikz, xcolor) and libraries "
        f"(arrows.meta,positioning,fit,backgrounds,calc,shapes.geometric). No fontawesome/includegraphics.",
        system=VIS_SYS, label="fix", timeout=300)

def critic(png: Path, plan: str, rnd: int) -> dict:
    txt = claude(
        f"This is round {rnd} of the teaser. Judge the RENDERED figure against the PLAN.\n\n"
        f"PLAN:\n{plan}\n\nReturn STRICT JSON only:\n"
        f'{{"score": <0-10 float>, "accept": <bool>, "issues": ["..."], '
        f'"revision_instructions": "concrete edits for the TikZ engineer, or empty if accept"}}\n'
        f"Accept only if score >= 8.5 and the headline (learned graph wins; markers; recursive "
        f"shared block; efficiency) is unmistakable and nothing overlaps or is cut off.",
        system=CRITIC_SYS, image=png, label=f"critic-r{rnd}", timeout=300)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {"score": 0, "accept": False,
                "issues": ["unparseable critic output"], "revision_instructions": txt[:800]}
    except Exception:
        return {"score": 0, "accept": False, "issues": ["bad json"],
                "revision_instructions": txt[:800]}

# ----------------------------------------------------------------------------- orchestration
def compile_with_repair(code: str, stem: str, max_fix: int = 3):
    for attempt in range(max_fix + 1):
        ok, png, err = compile_tikz(code, stem)
        if ok:
            print(f"    compiled {stem} on attempt {attempt+1}")
            return code, png
        print(f"    compile FAILED ({stem}) attempt {attempt+1}: {err.splitlines()[0][:80] if err else ''}")
        if attempt == max_fix:
            return code, None
        code = extract_tikz(fix_compile(code, err))
    return code, None

def main(max_rounds: int = 3):
    paper = (HERE / "paper_content.md").read_text()
    caption = (HERE / "caption.txt").read_text()

    print(">> Retriever"); style_ref = retriever(paper)
    (RUN / "01_style_ref.txt").write_text(style_ref)
    print(">> Planner");   plan = planner(paper, caption, style_ref)
    (RUN / "02_plan.txt").write_text(plan)
    print(">> Stylist");   spec = stylist(plan, style_ref)
    (RUN / "03_spec.txt").write_text(spec)

    print(">> Visualizer (initial)")
    code = extract_tikz(visualizer(spec, None, None))
    code, png = compile_with_repair(code, "round0")
    (RUN / "round0" / "round0.tex").write_text(code)

    best = {"score": -1, "png": png, "code": code, "round": 0}
    for rnd in range(1, max_rounds + 1):
        if png is None:
            print("    !! no render to critique; asking visualizer to rebuild from spec")
            code = extract_tikz(visualizer(spec, None, None))
            code, png = compile_with_repair(code, f"round{rnd}")
            if png is None:
                continue
        print(f">> Critic round {rnd}")
        verdict = critic(png, plan, rnd)
        print(f"    score={verdict.get('score')}  accept={verdict.get('accept')}")
        (RUN / f"critic_r{rnd}.json").write_text(json.dumps(verdict, indent=2))
        if float(verdict.get("score", 0)) >= best["score"]:
            best = {"score": float(verdict.get("score", 0)), "png": png, "code": code, "round": rnd}
        if verdict.get("accept"):
            print("    Critic ACCEPTED."); break
        if rnd == max_rounds:
            break
        print(f">> Visualizer (revision {rnd})")
        code = extract_tikz(visualizer(spec, code, verdict.get("revision_instructions", "")))
        code, png = compile_with_repair(code, f"round{rnd}")
        if png is not None:
            (RUN / f"round{rnd}" / f"round{rnd}.tex").write_text(code)

    if best["png"] is None:
        print("FAILED: no compilable teaser produced."); sys.exit(1)
    # copy final artifacts
    Path(HERE / "teaser_smart.tex").write_text(best["code"])
    pdf = best["png"].with_suffix(".pdf")
    shutil.copy(pdf, HERE / "teaser_smart.pdf")
    shutil.copy(best["png"], HERE / "teaser_smart.png")
    print(f"\nDONE. best round={best['round']} score={best['score']}")
    print(f"  {HERE/'teaser_smart.pdf'}\n  {HERE/'teaser_smart.png'}")

if __name__ == "__main__":
    main(max_rounds=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
