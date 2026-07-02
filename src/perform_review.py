"""
perform_review.py — CSRP scoring engine (build-order step 8).

Adapted in place from the AI-Scientist paper-reviewer, mirroring its structure:
module-level system prompt + instruction form, a THOUGHT/JSON response contract
parsed with extract_json_between_markers, a per-unit LLM scoring call, and a
deterministic aggregation step (the analogue of get_meta_review).

Pipeline role: take the merged extraction records {signal_id: record} from
perform_extraction (Layer 1 structured records supersede Layer 2 NLP records for
the same signal), assign each signal a 1-5 score, then aggregate to a composite
and a bps premium.

Scoring routes (spec):
  - threshold : hard-threshold signals carry a deterministic Layer-1 flag, so we
                score them in pure Python (no LLM).
  - llm       : relative signals are scored 1-5 by the LLM, anchored at 3 = sector
                peer-average (comparables_mode OFF -> LLM prior is the benchmark).
  - unscored  : extraction had no citation -> never fabricated, carried through.
"""

import json
from typing import Any

from .llm import get_response_from_llm, extract_json_between_markers

DEFAULT_MODEL: str = "claude-sonnet-4-6"

# Peer comparables are out of scope for the POC. When True, peer financials would
# be loaded from data/peers/ (not implemented).
CONFIG: dict[str, bool] = {"comparables_mode": False}

# Deterministic scores for hard-threshold signals (1-5 scale, 3 = peer-average).
THRESHOLD_BREACH_SCORE: int = 4   # flag True  -> above-average risk
THRESHOLD_CLEAR_SCORE: int = 2    # flag False -> below-average risk

Record = dict[str, Any]

# --- Prompts (f-string constants at top, max 5 per file; 2 used) -------------

SCORER_SYSTEM_PROMPT: str = (
    "You are a valuation analyst assigning company-specific risk scores from a "
    "Japanese securities report (有価証券報告書). Be critical and cautious. Score "
    "only on the cited evidence provided; if evidence is weak, score near the "
    "peer-average anchor rather than inventing risk."
)

SCORING_INSTRUCTIONS: str = """Score each risk signal from 1 to 5 RELATIVE to sector peers:
  1 = minimal risk (better than peer average)
  2 = below-average risk
  3 = peer-average (the anchor — use this when evidence is neutral or thin)
  4 = above-average risk, warrants attention
  5 = significant, value-impairing risk

Comparables are unavailable, so use your sector prior as the "3" anchor.

Respond in this format:

THOUGHT:
<brief, signal-specific reasoning — not generic boilerplate>

SCORE JSON:
```json
{{"SIGNAL_ID": {{"score": <1-5 int>, "rationale_en": "<one sentence>", "rationale_ja": "<同じ趣旨を日本語で一文>"}}}}
```

Score every signal id listed below. Give both an English and a Japanese
rationale. This JSON is parsed automatically."""

CATEGORY_SCORE_PROMPT: str = """Risk category: {category_name} for {company_name}.

Score the following signals using the evidence extracted from the filing:
{signals_block}

{instructions}"""


# --- merge -------------------------------------------------------------------


def merge_records(nlp: dict[str, Record], structured: dict[str, Record]) -> dict[str, Record]:
    """Combine Layer-2 (NLP) and Layer-1 (structured) records. A structured
    record supersedes the NLP one for the same signal when it is scored;
    otherwise the scored record (if any) is kept."""
    merged: dict[str, Record] = dict(nlp)
    for sid, srec in structured.items():
        prev = merged.get(sid)
        if srec.get("status") == "scored" or prev is None or prev.get("status") != "scored":
            merged[sid] = srec
    return merged


# --- per-signal scoring ------------------------------------------------------


def _signal_result(rec: Record, score: int | None, scored_by: str,
                   rationale_en: str | None, rationale_ja: str | None) -> dict[str, Any]:
    return {
        "score": score,
        "scored_by": scored_by,           # "threshold" | "llm" | "unscored"
        "rationale_en": rationale_en,
        "rationale_ja": rationale_ja,
        "value": rec.get("value"),
        "flag": rec.get("flag"),
        "citation": rec.get("citation"),
        "status": "scored" if score is not None else "unscored",
    }


def _append_log(log_path: str, entry: dict[str, Any]) -> None:
    """Append one JSONL line (mirrors perform_extraction._log)."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _score_category_llm(category: dict[str, Any], signals: list[dict[str, Any]],
                        records: dict[str, Record], client: Any, model: str,
                        company_name: str, log_path: str | None = None
                        ) -> dict[str, dict[str, Any]]:
    """One LLM call scoring the relative signals of a category. Returns
    {signal_id: {"score": int, "rationale_en": str, "rationale_ja": str}}
    (clamped to 1-5). Persists the full THOUGHT to log_path if given."""
    lines = []
    for s in signals:
        rec = records[s["id"]]
        lines.append(
            f"- {s['id']} | {s['name_en']}: evidence = {rec.get('evidence_en') or rec.get('value')}"
        )
    prompt = CATEGORY_SCORE_PROMPT.format(
        category_name=category["name_en"], company_name=company_name,
        signals_block="\n".join(lines), instructions=SCORING_INSTRUCTIONS,
    )
    resp, _ = get_response_from_llm(
        prompt, client=client, model=model,
        system_message=SCORER_SYSTEM_PROMPT, temperature=0.0,
    )
    parsed = extract_json_between_markers(resp) or {}
    out: dict[str, dict[str, Any]] = {}
    for s in signals:
        entry = parsed.get(s["id"]) if isinstance(parsed, dict) else None
        if entry and entry.get("score") is not None:
            score = max(1, min(5, int(entry["score"])))
            out[s["id"]] = {"score": score,
                            "rationale_en": entry.get("rationale_en") or entry.get("rationale"),
                            "rationale_ja": entry.get("rationale_ja")}
    if log_path:
        thought = resp.split("```")[0].split("SCORE JSON")[0].strip()
        _append_log(log_path, {"stage": "scoring", "category": category["id"],
                               "thought": thought, "scores": out})
    return out


def score_signals(taxonomy: dict[str, Any], records: dict[str, Record],
                  client: Any = None, model: str = DEFAULT_MODEL,
                  company_name: str = "the company",
                  log_path: str | None = None) -> dict[str, dict[str, Any]]:
    """Score every signal in the taxonomy. Threshold signals are scored in
    Python; relative signals go to the LLM (one call per category)."""
    results: dict[str, dict[str, Any]] = {}
    for category in taxonomy["categories"]:
        relative: list[dict[str, Any]] = []
        for s in category["signals"]:
            rec = records.get(s["id"])
            if not rec or rec.get("status") != "scored":
                reason = (rec or {}).get("reason", "not extracted")
                results[s["id"]] = _signal_result(rec or {}, None, "unscored", reason, reason)
            elif rec.get("layer") == "structured" and rec.get("flag") is not None:
                score = THRESHOLD_BREACH_SCORE if rec["flag"] else THRESHOLD_CLEAR_SCORE
                results[s["id"]] = _signal_result(rec, score, "threshold",
                                                  rec.get("evidence_en"), rec.get("evidence_ja"))
            else:
                relative.append(s)

        if relative and client is not None:
            llm_scores = _score_category_llm(category, relative, records, client, model,
                                             company_name, log_path)
            for s in relative:
                got = llm_scores.get(s["id"])
                if got:
                    results[s["id"]] = _signal_result(records[s["id"]], got["score"], "llm",
                                                      got.get("rationale_en"), got.get("rationale_ja"))
                else:
                    results[s["id"]] = _signal_result(records[s["id"]], None, "unscored",
                                                      "LLM returned no score", "LLMがスコアを返しませんでした")
        else:
            for s in relative:  # no client: cannot score relative signals
                results[s["id"]] = _signal_result(records[s["id"]], None, "unscored",
                                                  "LLM scoring skipped (no client)", "LLMスコアリングをスキップ（クライアント無し）")
    return results


# --- aggregation (deterministic; the get_meta_review analogue) ---------------


def composite_to_bps(taxonomy: dict[str, Any], composite: float) -> tuple[int, int]:
    """Map a composite score to a bps premium range via taxonomy.score_to_premium_bps."""
    ranges = []
    for score_key, bps_val in taxonomy["score_to_premium_bps"].items():
        lo_s, hi_s = (float(x) for x in score_key.split("-"))
        lo_b, hi_b = (int(x) for x in bps_val.split("-"))
        ranges.append((lo_s, hi_s, lo_b, hi_b))
    ranges.sort()
    for lo_s, hi_s, lo_b, hi_b in ranges:
        if composite < hi_s:
            return lo_b, hi_b
    return ranges[-1][2], ranges[-1][3]


def aggregate(taxonomy: dict[str, Any], scored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Category means -> weighted composite -> bps. Supplemental (JPN) scores are
    reported separately and excluded from the composite."""
    category_scores: dict[str, float | None] = {}
    japan_supplemental: dict[str, Any] = {}
    unscored: list[dict[str, str]] = []

    for category in taxonomy["categories"]:
        vals = []
        for s in category["signals"]:
            r = scored.get(s["id"], {})
            if r.get("score") is not None:
                vals.append(r["score"])
            else:
                unscored.append({"id": s["id"], "reason": r.get("rationale_en") or "unscored"})
        mean = round(sum(vals) / len(vals), 2) if vals else None
        if category.get("supplemental"):
            japan_supplemental[category["id"]] = {
                "score": mean,
                "signals": {s["id"]: scored.get(s["id"], {}) for s in category["signals"]},
            }
        else:
            category_scores[category["id"]] = mean

    num = den = 0.0
    for category in taxonomy["categories"]:
        if category.get("supplemental"):
            continue
        cs = category_scores.get(category["id"])
        if cs is not None:
            num += cs * category["default_weight"]
            den += category["default_weight"]
    composite = round(num / den, 2) if den else None

    bps_low, bps_high = (composite_to_bps(taxonomy, composite) if composite is not None else (None, None))
    return {
        "category_scores": category_scores,
        "composite_score": composite,
        "csrp_bps_low": bps_low,
        "csrp_bps_high": bps_high,
        "japan_supplemental": japan_supplemental,
        "unscored_signals": unscored,
    }


def score(taxonomy: dict[str, Any], records: dict[str, Record], client: Any = None,
          model: str = DEFAULT_MODEL, company_name: str = "the company",
          log_path: str | None = None) -> dict[str, Any]:
    """Top-level entry: score all signals then aggregate. Mirrors perform_review's
    role of returning the full result dict."""
    signals = score_signals(taxonomy, records, client, model, company_name, log_path)
    result = aggregate(taxonomy, signals)
    result["signals"] = signals
    result["comparables_mode"] = CONFIG["comparables_mode"]
    return result


# --- Smoke test (no API by default) ------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    taxonomy = json.load(open(os.path.join(here, "taxonomy.json")))["csrp_framework"]

    # Synthetic structured records exercise the deterministic path with no API.
    def struct(flag: bool | None, value: Any) -> Record:
        return {"value": value, "flag": flag, "evidence_en": "synthetic",
                "evidence_ja": "（合成データ）", "citation": {"citation_type": "structured"},
                "layer": "structured", "status": "scored"}

    records: dict[str, Record] = {
        "FIN_01": struct(True, 6.9),    # breach -> 4
        "FIN_02": struct(False, 6.73),  # clear  -> 2
        "FIN_03": struct(True, "57% ≤24m"),
    }
    scored = score_signals(taxonomy, records, client=None)
    print("threshold scoring:")
    for sid in ("FIN_01", "FIN_02", "FIN_03"):
        r = scored[sid]
        print(f"  {sid}: score={r['score']} scored_by={r['scored_by']}")

    agg = aggregate(taxonomy, scored)
    scored_cats = {k: v for k, v in agg["category_scores"].items() if v is not None}
    print(f"\ncategory_scores (scored only): {scored_cats}")
    print(f"composite={agg['composite_score']}  -> "
          f"{agg['csrp_bps_low']}-{agg['csrp_bps_high']} bps")
    print(f"unscored signals: {len(agg['unscored_signals'])}")

    print("\ncomposite_to_bps checks:")
    for c in (1.2, 2.4, 3.2, 4.5):
        print(f"  {c} -> {composite_to_bps(taxonomy, c)} bps")

    # Live end-to-end scoring (costs tokens): python -m src.perform_review --llm
    if "--llm" in sys.argv:
        from .perform_ingestion import ingest
        from .perform_extraction import Extractor, StructuredExtractor
        from .llm import create_client

        client, model = create_client(DEFAULT_MODEL)
        doc = ingest(os.path.join(here, "data", "SAAF.pdf"))
        ext = Extractor(doc, taxonomy, client=client, model=model, company_name="SAAF Co., Ltd.")
        by_id = {c["id"]: c for c in taxonomy["categories"]}
        nlp = ext.extract_category(by_id["MGMT"])  # one NLP category to keep it cheap
        structured = StructuredExtractor(doc, client=client, model=model, company_name="SAAF Co., Ltd.").run()
        merged = merge_records(nlp, structured)
        log_path = os.path.join(here, "outputs", "_smoke_run_log.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        result = score(taxonomy, merged, client=client, model=model,
                       company_name="SAAF Co., Ltd.", log_path=log_path)
        print("\n### live scoring (MGMT + structured) ###")
        for sid, r in result["signals"].items():
            if r["score"] is not None:
                print(f"  {sid}: {r['score']} ({r['scored_by']}) — {r['rationale_en']}")
        print(f"composite={result['composite_score']} -> {result['csrp_bps_low']}-{result['csrp_bps_high']} bps")
        print(f"THOUGHT log -> {log_path}")
