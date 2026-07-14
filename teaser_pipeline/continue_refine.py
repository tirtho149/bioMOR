#!/usr/bin/env python3
"""Continue the Critic<->Visualizer loop from the current best teaser_smart.tex."""
import sys, json, shutil
from pathlib import Path
import paperviz_claude as P

HERE = P.HERE
plan = (P.RUN / "02_plan.txt").read_text()
spec = (P.RUN / "03_spec.txt").read_text()
code = (HERE / "teaser_smart.tex").read_text()
best = {"score": 8.0, "code": code, "png": HERE / "teaser_smart.png", "round": 3}

# seed feedback = last critic's instructions + explicit border-clip guard
seed = json.loads((P.RUN / "critic_r3.json").read_text())["revision_instructions"]
seed += ("\nHARD CONSTRAINT: nothing may touch or cross the rounded panel borders. Give the "
         "right edge of Panel B >=6mm inner margin so no badge text is clipped. Shrink the mini "
         "bar chart and the badges if needed. Keep the whole figure landscape and uncrowded.")

start = 4
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
feedback = seed
for k in range(N):
    rnd = start + k
    print(f">> Visualizer (revision -> round{rnd})")
    code = P.extract_tikz(P.visualizer(spec, code, feedback))
    code, png = P.compile_with_repair(code, f"round{rnd}")
    if png is None:
        print("    compile failed; keeping previous best"); continue
    (P.RUN / f"round{rnd}" / f"round{rnd}.tex").write_text(code)
    print(f">> Critic round {rnd}")
    v = P.critic(png, plan, rnd)
    (P.RUN / f"critic_r{rnd}.json").write_text(json.dumps(v, indent=2))
    print(f"    score={v.get('score')} accept={v.get('accept')}")
    if float(v.get("score", 0)) >= best["score"]:
        best = {"score": float(v.get("score", 0)), "code": code, "png": png, "round": rnd}
    if v.get("accept"):
        print("    ACCEPTED"); break
    feedback = v.get("revision_instructions", "") + "\n" + \
        "HARD CONSTRAINT: keep >=6mm margin from all panel borders; no clipped text."

(HERE / "teaser_smart.tex").write_text(best["code"])
shutil.copy(best["png"].with_suffix(".pdf"), HERE / "teaser_smart.pdf")
shutil.copy(best["png"], HERE / "teaser_smart.png")
print(f"\nFINAL best round={best['round']} score={best['score']}")
