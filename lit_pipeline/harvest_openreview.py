#!/usr/bin/env python3
"""
Step 1 (OpenReview variant): HARVEST gene-classification papers from ICLR & ICML,
years 2024-2026, via the free OpenReview API -- the authoritative source for these
venues (OpenAlex barely indexes them). OpenReview hands us the PDF link directly.

Accepted papers have content.venueid == "<Venue>.cc/<year>/Conference" and a
content.venue like "ICLR 2024 poster/oral/spotlight". We pull all accepted notes,
keep genuine gene/genomic work (shared gene_filter), drop image-based.

Emits papers.jsonl rows in the get_pdfs.py schema, with
  openAccessPdf.url = https://openreview.net/pdf?id=<noteId>
so the downloader fetches straight from OpenReview.
"""
import argparse
import json
import os
import time

import requests

from gene_filter import keep

API = "https://api2.openreview.net/notes"


def get_with_retry(session, params, tries=6):
    for attempt in range(tries):
        r = session.get(API, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def val(content, key, default=""):
    v = (content or {}).get(key)
    if isinstance(v, dict):
        return v.get("value", default)
    return v if v is not None else default


def harvest_venue(session, venueid, drop_image):
    """Page through all accepted notes for a venueid; yield matching records."""
    offset, seen, kept = 0, 0, 0
    while True:
        r = get_with_retry(session, {"content.venueid": venueid, "limit": 1000,
                                      "offset": offset, "details": "replyCount"})
        notes = r.json().get("notes") or []
        if not notes:
            break
        for note in notes:
            seen += 1
            c = note.get("content", {})
            title = val(c, "title")
            abstract = val(c, "abstract")
            if not keep(title, abstract, drop_image=drop_image):
                continue
            nid = note.get("id")
            authors = val(c, "authors") or []
            venue = val(c, "venue")
            pdf = f"https://openreview.net/pdf?id={nid}"
            rec = {
                "paperId": nid,
                "title": title,
                "year": int(venueid.split("/")[1]) if venueid.split("/")[1].isdigit() else None,
                "venue": venue or venueid,
                "authors": [{"name": a} for a in authors],
                "url": f"https://openreview.net/forum?id={nid}",
                "abstract": abstract,
                "externalIds": {},               # OpenReview rarely carries DOIs
                "openAccessPdf": {"url": pdf},   # <-- direct PDF
                "citationCount": None,
            }
            kept += 1
            yield rec
        offset += len(notes)
        print(f"    {venueid}: scanned {seen}, kept {kept}...", end="\r", flush=True)
        if len(notes) < 1000:
            break
    print(f"    {venueid}: scanned {seen}, kept {kept}            ")


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser(description="Harvest ICLR/ICML gene-classification papers (OpenReview).")
    ap.add_argument("--years", default="2024,2025,2026")
    ap.add_argument("--venues", default="ICLR,ICML")
    ap.add_argument("--out", default=os.path.join(here, "papers.jsonl"))
    ap.add_argument("--append", action="store_true", help="Append instead of overwrite (merge with AAAI).")
    ap.add_argument("--keep-image", action="store_true")
    args = ap.parse_args()

    sess = requests.Session()
    sess.headers.update({"User-Agent": "RecursiveQFormer-LitPipeline/1.0"})

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    venues = [v.strip().upper() for v in args.venues.split(",") if v.strip()]
    total = 0
    mode = "a" if args.append else "w"
    with open(args.out, mode) as fh:
        for venue in venues:
            for year in years:
                venueid = f"{venue}.cc/{year}/Conference"
                print(f"[{venue} {year}] {venueid}")
                try:
                    for rec in harvest_venue(sess, venueid, drop_image=not args.keep_image):
                        fh.write(json.dumps(rec) + "\n")
                        total += 1
                except requests.HTTPError as e:
                    print(f"    (no data for {venueid}: {e})")
    print(f"\n[harvest_openreview] wrote {total} records -> {args.out} (append={args.append})")


if __name__ == "__main__":
    main()
