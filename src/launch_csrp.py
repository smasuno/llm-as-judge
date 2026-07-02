"""
launch_csrp.py — end-to-end CSRP pipeline entry point (build-order step 10).

Wires the whole pipeline for one Japanese securities report (有価証券報告書):
  ingest → extract (Layer 2 NLP + Layer 1 structured) → merge → score → writeup.

Usage (from the repo root):
  python -m src.launch_csrp data/SAAF.pdf --company "SAAF Co., Ltd."
  python -m src.launch_csrp data/SAAF.pdf --no-markdown      # JSON + Excel only

Outputs land in outputs/<slug>/ (JSON, DCF-ready Excel, and — unless
--no-markdown — the LLM-authored EN/JA markdown). Agent reasoning (section
selection + scoring THOUGHT) is logged to outputs/<slug>/run_log.jsonl.
"""

import argparse
import datetime
import json
import os
from typing import Any

from .llm import create_client
from .perform_ingestion import ingest
from .perform_extraction import Extractor, StructuredExtractor
from .perform_review import merge_records, score, DEFAULT_MODEL
from .perform_writeup import perform_writeup, recommended_bps, _slug


def load_taxonomy() -> dict[str, Any]:
    here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(os.path.join(here, "taxonomy.json"), encoding="utf-8") as f:
        return json.load(f)["csrp_framework"]


def run(filename: str, company: str | None = None, model: str = DEFAULT_MODEL,
        out_dir: str = "outputs", make_markdown: bool = True,
        num_refinements: int = 0, include_jpn: bool = True,
        filing_date: str = "N/A") -> dict[str, Any]:
    """Run the full pipeline on one filing. Returns {"result", "paths"}."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Filing not found: {filename}")

    taxonomy = load_taxonomy()
    company = company or os.path.splitext(os.path.basename(filename))[0]
    folder = os.path.join(out_dir, _slug(company))
    os.makedirs(folder, exist_ok=True)
    run_log = os.path.join(folder, "run_log.jsonl")
    open(run_log, "w").close()  # truncate any prior run

    client, model = create_client(model)

    print(f"[1/5] ingesting {os.path.basename(filename)} …")
    doc = ingest(filename, client=client, model=model)
    print(f"      {doc.page_count} pages, {len(doc.sections)} sections ({doc.toc_source} TOC)")

    print("[2/5] extracting — Layer 2 (NLP, per category) …")
    ext = Extractor(doc, taxonomy, client=client, model=model,
                    company_name=company, log_path=run_log)
    nlp = ext.extract_all(include_jpn=include_jpn)

    print("[3/5] extracting — Layer 1 (structured / quantitative) …")
    structured = StructuredExtractor(doc, client=client, model=model,
                                     company_name=company).run()
    merged = merge_records(nlp, structured)

    print("[4/5] scoring …")
    result = score(taxonomy, merged, client=client, model=model,
                   company_name=company, log_path=run_log)

    metadata = {
        "company": company,
        "filing_date": filing_date,
        "run_date": datetime.date.today().isoformat(),
        "model": model,
        "comparables_mode": result.get("comparables_mode"),
    }

    print(f"[5/5] writing outputs{'' if make_markdown else ' (JSON + Excel only)'} …")
    paths = perform_writeup(result, taxonomy, metadata,
                            client=(client if make_markdown else None),
                            model=model, out_dir=out_dir, num_refinements=num_refinements)

    _print_summary(company, result, paths, run_log)
    return {"result": result, "paths": paths}


def _print_summary(company: str, result: dict[str, Any], paths: dict[str, str], run_log: str) -> None:
    scored = sum(1 for r in result["signals"].values() if r["status"] == "scored")
    total = len(result["signals"])
    print("\n" + "=" * 60)
    print(f"CSRP result — {company}")
    print(f"  composite score : {result['composite_score']}")
    print(f"  CSRP premium    : {result['csrp_bps_low']}–{result['csrp_bps_high']} bps "
          f"(recommended {recommended_bps(result)} bps to add to WACC)")
    print(f"  signals scored  : {scored}/{total}  ({total - scored} unscored)")
    print("  outputs:")
    for key, path in paths.items():
        print(f"    {key:8} {path}")
    print(f"    run_log  {run_log}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CSRP pipeline on a Japanese securities report (有価証券報告書).")
    parser.add_argument("filename", help="Path to the yūhō PDF (e.g. data/SAAF.pdf)")
    parser.add_argument("--company", default=None,
                        help="Company name for the report (default: PDF filename stem)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model id")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--filing-date", default="N/A", help="Filing date for metadata")
    parser.add_argument("--no-markdown", action="store_true",
                        help="Skip the LLM-authored markdown (produce JSON + Excel only)")
    parser.add_argument("--no-jpn", action="store_true",
                        help="Skip the JPN supplemental signals")
    parser.add_argument("--refine", type=int, default=0,
                        help="Markdown refinement passes per section (default 0)")
    args = parser.parse_args()

    run(args.filename, company=args.company, model=args.model, out_dir=args.out_dir,
        make_markdown=not args.no_markdown, num_refinements=args.refine,
        include_jpn=not args.no_jpn, filing_date=args.filing_date)


if __name__ == "__main__":
    main()
