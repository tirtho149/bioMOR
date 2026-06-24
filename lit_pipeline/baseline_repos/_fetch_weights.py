#!/usr/bin/env python3
"""Fetch pretrained checkpoints for the reproducible baselines (scriptable hosts only)."""
import os, sys, traceback, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
def log(m): print(m, flush=True)

def geneformer():
    from huggingface_hub import snapshot_download
    dest = os.path.join(BASE, "Geneformer")
    # 104M base + cancer-tuned (most relevant for TCGA) + tokenizer dictionaries; skip the big 316M/V1
    snapshot_download(
        repo_id="ctheodoris/Geneformer", repo_type="model", local_dir=dest,
        allow_patterns=[
            "Geneformer-V2-104M/*", "Geneformer-V2-104M_CLcancer/*",
            "geneformer/*.pkl", "geneformer/*.pickle", "*.pkl", "*.pickle",
        ],
    )
    return "Geneformer-V2-104M + V2-104M_CLcancer + dictionaries"

def cell2sentence():
    from huggingface_hub import snapshot_download
    dest = os.path.join(BASE, "Cell2Sentence", "checkpoints", "C2S-Pythia-410m-cell-type-prediction")
    os.makedirs(dest, exist_ok=True)
    snapshot_download(repo_id="vandijklab/C2S-Pythia-410m-cell-type-prediction",
                      repo_type="model", local_dir=dest)
    return "C2S-Pythia-410M cell-type-prediction"

def scgpt():
    import gdown
    folders = {
        "whole-human": "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
        "pan-cancer":  "https://drive.google.com/drive/folders/13QzLHilYUd0v3HTwa_9n4G4yEF-hdkqa",
    }
    out=[]
    for name, url in folders.items():
        d = os.path.join(BASE, "scGPT", "checkpoints", name)
        os.makedirs(d, exist_ok=True)
        gdown.download_folder(url=url, output=d, quiet=False, use_cookies=False)
        out.append(name)
    return "scGPT checkpoints: " + ", ".join(out)

def cellplm():
    import gdown
    # Dropbox folder -> zip
    d = os.path.join(BASE, "CellPLM", "ckpt"); os.makedirs(d, exist_ok=True)
    zurl = "https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&dl=1"
    zpath = os.path.join(d, "cellplm_ckpt.zip")
    subprocess.run(["curl", "-L", "--fail", "-o", zpath, zurl], check=True)
    subprocess.run(["python", "-c", f"import zipfile;zipfile.ZipFile('{zpath}').extractall('{d}')"], check=True)
    os.remove(zpath)
    return "CellPLM 20230926_85M checkpoint"

STEPS = [("Geneformer", geneformer), ("Cell2Sentence", cell2sentence),
         ("scGPT", scgpt), ("CellPLM", cellplm)]

results={}
for name, fn in STEPS:
    log(f"\n===== FETCH {name} =====")
    try:
        msg = fn(); results[name]=("OK", msg); log(f"  OK: {msg}")
    except Exception as e:
        results[name]=("FAIL", str(e)); log(f"  FAIL: {e}"); traceback.print_exc()

log("\n===== SUMMARY =====")
for n,(s,m) in results.items(): log(f"  {s:4s}  {n}: {m}")
log("DONE_FETCH")
