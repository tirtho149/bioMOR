#!/usr/bin/env python3
"""
Step 1 of the literature pipeline: HARVEST metadata for gene-classification papers.

Uses the free Semantic Scholar Graph API (bulk search, no key required) to pull
every paper matching the query, keeping the fields we need to *resolve a PDF*
later: openAccessPdf (direct link), externalIds (ArXiv / DOI / PMID / PMCID).

Output: papers.jsonl  (one JSON record per paper)

This step does NOT download PDFs -- it only builds the worklist. Run get_pdfs.py next.
"""
import argparse
import json
import os
import re
import sys
import time

import requests

S2_BULK = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# Everything we need downstream to find a free PDF.
FIELDS = ",".join([
    "title", "year", "venue", "authors", "url", "abstract",
    "externalIds",       # ArXiv, DOI, PubMed (PMID), PubMedCentral (PMCID), ...
    "openAccessPdf",     # {"url": ...} when S2 already knows a free PDF
    "citationCount",
    "publicationTypes",
])

DEFAULT_QUERY = (
    "gene classification | genomic sequence classification | "
    "gene expression classification | DNA sequence classification | "
    "single-cell classification | cell type annotation | "
    "scRNA-seq classification | gene regulatory network classification | "
    "genomics deep learning classification"
)

# EXACT venue match (case-insensitive, punctuation-stripped). ICLR/ICML/AAAI only.
# Exact matching avoids look-alikes like "ICML and AI Applications" / "ICMLAS".
DEFAULT_VENUES = [
    "iclr",
    "international conference on learning representations",
    "icml",
    "international conference on machine learning",
    "aaai",
    "aaai conference on artificial intelligence",
    "proceedings of the aaai conference on artificial intelligence",
]

# Drop image-/vision-based gene classification papers (genomap-style etc.).
IMAGE_KEYWORDS = [
    "image", "imaging", "vision", "convolutional image", "microscop",
    "histopath", "histolog", "radiolog", "mri", "ct scan", "x-ray",
    "photograph", "pixel", "gene2image", "genomap", "spatial transcriptomic image",
]


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower()).strip()


def venue_ok(paper, venues):
    if not venues:
        return True
    v = _norm(paper.get("venue"))
    return v in venues


def is_image_based(paper):
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    return any(k in text for k in IMAGE_KEYWORDS)


def harvest(query, year, out_path, max_papers, session, venues, exclude_image):
    """Page through the bulk endpoint via continuation token; stream to JSONL."""
    params = {"query": query, "fields": FIELDS}
    if year:
        params["year"] = year

    token = None
    n = 0
    seen = 0
    total = None
    with open(out_path, "w") as fh:
        while True:
            if token:
                params["token"] = token
            for attempt in range(6):
                r = session.get(S2_BULK, params=params, timeout=60)
                if r.status_code == 429:          # rate limited -> back off
                    time.sleep(3 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            else:
                print("  giving up after repeated 429s", file=sys.stderr)
                break

            data = r.json()
            if total is None:
                total = data.get("total")
                print(f"  query matches ~{total} papers total")

            for p in (data.get("data") or []):
                seen += 1
                if not venue_ok(p, venues):
                    continue
                if exclude_image and is_image_based(p):
                    continue
                fh.write(json.dumps(p) + "\n")
                n += 1
                if max_papers and n >= max_papers:
                    print(f"  hit --max {max_papers}, stopping")
                    return n

            token = data.get("token")
            print(f"  scanned {seen}, kept {n}...", end="\r", flush=True)
            if not token:
                break
            time.sleep(1)          # be polite to the free endpoint
    print()
    return n


def main():
    ap = argparse.ArgumentParser(description="Harvest gene-classification paper metadata (free S2 API).")
    ap.add_argument("--query", default=DEFAULT_QUERY,
                    help="Semantic Scholar bulk query (supports | & + operators).")
    ap.add_argument("--year", default="2020-2026",
                    help="Year range, e.g. 2024-2026. Empty string = all years.")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "papers.jsonl"))
    ap.add_argument("--max", type=int, default=0, help="Cap number of papers (0 = unlimited).")
    ap.add_argument("--venues", default=",".join(DEFAULT_VENUES),
                    help="Comma-separated venue substrings to keep. Empty = all venues.")
    ap.add_argument("--no-venue-filter", action="store_true", help="Keep all venues.")
    ap.add_argument("--keep-image", action="store_true",
                    help="Do NOT drop image-/vision-based papers (default is to drop them).")
    args = ap.parse_args()

    venues = [] if args.no_venue_filter else [_norm(v) for v in args.venues.split(",") if v.strip()]
    exclude_image = not args.keep_image

    sess = requests.Session()
    sess.headers.update({"User-Agent": "RecursiveQFormer-LitPipeline/1.0 (research; mailto:tracclaude3@gmail.com)"})
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        sess.headers["x-api-key"] = api_key
        print("  using S2_API_KEY from env (higher rate limit)")

    print(f"[harvest] query = {args.query!r}  year = {args.year or 'ALL'}")
    print(f"[harvest] venues = {venues or 'ALL'}  exclude_image = {exclude_image}")
    n = harvest(args.query, args.year, args.out, args.max, sess, venues, exclude_image)
    print(f"\n[harvest] wrote {n} records -> {args.out}")


if __name__ == "__main__":
    main()
