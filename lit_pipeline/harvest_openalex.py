#!/usr/bin/env python3
"""
Step 1 (OpenAlex variant): HARVEST gene-classification papers from ICLR / ICML / AAAI,
years 2024-2026, using the free OpenAlex API -- which has *reliable* conference-venue
metadata (unlike Semantic Scholar's often-blank `venue` field for these venues).

Strategy (transparent + reproducible):
  * Pull EVERY work published 2024-2026 at the exact OpenAlex source IDs for
    ICML / ICLR / AAAI (cursor pagination, 200/page).
  * Keep only works whose title+abstract contain a genuine gene/genomic signal.
  * Drop image-/vision-based papers (user wants sequence/expression, not imaging).

Emits papers.jsonl in the SAME schema get_pdfs.py expects, so the downloader runs
unchanged. OpenAlex even hands us pdf_url + DOI + PMID directly.
"""
import argparse
import json
import os
import re

import requests

from gene_filter import keep

WORKS = "https://api.openalex.org/works"
MAILTO = os.environ.get("UNPAYWALL_EMAIL", "tracclaude3@gmail.com")

# Exact OpenAlex source IDs (verified via /sources?search=).
# NOTE: OpenAlex only reliably indexes AAAI under its source. ICML/ICLR live on
# OpenReview -> use harvest_openreview.py for those.
SOURCES = {
    "ICML": "s4306419644",
    "ICLR": "s4306419637",
    "AAAI": "s4210191458",   # Proceedings of the AAAI Conference on Artificial Intelligence
}


def reconstruct_abstract(inv):
    if not inv:
        return ""
    order = {}
    for word, positions in inv.items():
        for pos in positions:
            order[pos] = word
    return " ".join(order[i] for i in sorted(order))


def to_record(w):
    """Map an OpenAlex work into the get_pdfs.py schema."""
    ids = w.get("ids") or {}
    doi = (ids.get("doi") or "").replace("https://doi.org/", "") or None
    pmid = (ids.get("pmid") or "").replace("https://pubmed.ncbi.nlm.nih.gov/", "") or None

    arxiv = None
    pdf_url = None
    locations = w.get("locations") or []
    boa = w.get("best_oa_location") or {}
    if boa.get("pdf_url"):
        pdf_url = boa["pdf_url"]
    for loc in locations:
        lp = (loc.get("landing_page_url") or "")
        m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})", lp)
        if m and not arxiv:
            arxiv = m.group(1)
        if not pdf_url and loc.get("pdf_url"):
            pdf_url = loc["pdf_url"]

    ext = {}
    if doi:
        ext["DOI"] = doi
    if arxiv:
        ext["ArXiv"] = arxiv
    if pmid:
        ext["PubMed"] = pmid

    src = ((w.get("primary_location") or {}).get("source") or {})
    return {
        "paperId": w.get("id"),
        "title": w.get("title") or w.get("display_name") or "",
        "year": w.get("publication_year"),
        "venue": src.get("display_name") or "",
        "authors": [{"name": a.get("author", {}).get("display_name", "")}
                    for a in (w.get("authorships") or [])],
        "url": ids.get("doi") or w.get("id"),
        "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
        "externalIds": ext,
        "openAccessPdf": {"url": pdf_url} if pdf_url else {},
        "citationCount": w.get("cited_by_count"),
    }


def harvest_source(session, source_id, years, keep_image):
    """Cursor through every work at a venue+years; yield matching records."""
    flt = f"publication_year:{years},primary_location.source.id:{source_id}"
    cursor = "*"
    seen = kept = 0
    while cursor:
        params = {
            "filter": flt, "per-page": 200, "cursor": cursor,
            "mailto": MAILTO,
            "select": "id,ids,title,display_name,publication_year,primary_location,"
                      "locations,best_oa_location,authorships,abstract_inverted_index,cited_by_count",
        }
        r = session.get(WORKS, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        for w in data.get("results", []):
            seen += 1
            rec = to_record(w)
            if not keep(rec["title"], rec.get("abstract"), drop_image=not keep_image):
                continue
            kept += 1
            yield rec
        cursor = data.get("meta", {}).get("next_cursor")
        print(f"    scanned {seen}, kept {kept}...", end="\r", flush=True)
    print(f"    scanned {seen}, kept {kept}            ")


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser(description="Harvest ICLR/ICML/AAAI gene-classification papers (OpenAlex).")
    ap.add_argument("--years", default="2024-2026")
    ap.add_argument("--out", default=os.path.join(here, "papers.jsonl"))
    ap.add_argument("--venues", default="ICML,ICLR,AAAI",
                    help="Subset of ICML,ICLR,AAAI to harvest.")
    ap.add_argument("--keep-image", action="store_true", help="Do not drop image-based papers.")
    args = ap.parse_args()

    sess = requests.Session()
    sess.headers.update({"User-Agent": f"RecursiveQFormer-LitPipeline/1.0 (mailto:{MAILTO})"})

    want = [v.strip().upper() for v in args.venues.split(",") if v.strip()]
    total = 0
    with open(args.out, "w") as fh:
        for name in want:
            sid = SOURCES.get(name)
            if not sid:
                print(f"  unknown venue {name}, skipping")
                continue
            print(f"[{name}] {sid}  years={args.years}")
            for rec in harvest_source(sess, sid, args.years, args.keep_image):
                fh.write(json.dumps(rec) + "\n")
                total += 1
    print(f"\n[harvest_openalex] wrote {total} gene-classification records -> {args.out}")


if __name__ == "__main__":
    main()
