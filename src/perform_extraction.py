"""
perform_extraction.py — Layer 2 (NLP) extraction for the CSRP pipeline.

Turns an IngestedDoc into per-signal evidence the scoring engine can consume,
following the TOC-first strategy:
  1. _get_relevant_sections() decides which TOC sections matter for a category —
     deterministic match (japan_utils) first, LLM fallback only for the leftovers.
  2. extract_category() reads ONLY those sections and asks the LLM for grounded,
     cited evidence per signal. No citation -> the signal is UNSCORED, never
     fabricated.

Layer 1 (quantitative ratios / hard-threshold flags) is intentionally NOT here
yet — it is the next focused increment. Every record this module emits is
tagged `"layer": "nlp"` so Layer 1 can later supersede the quantitative signals.
"""

import json
import os
from typing import Any

from .japan_utils import select_sections, match_section
from .llm import get_response_from_llm, extract_json_between_markers
from .perform_ingestion import IngestedDoc

DEFAULT_MODEL: str = "claude-sonnet-4-6"

# Cap on how much section text we feed per category, to bound token cost — the
# 注記事項 node alone can run to tens of thousands of characters.
MAX_SECTION_CHARS: int = 16000
MAX_PASSAGE_CHARS: int = 150

# Layer 1 hard-threshold rules (spec). Flag = elevated risk.
NET_DEBT_EBITDA_FLAG: float = 4.0    # FIN_01: flag if Net Debt/EBITDA > 4.0x
EBITDA_INTEREST_FLAG: float = 3.0    # FIN_02: flag if EBITDA/Interest < 3.0x
MATURITY_WALL_FLAG: float = 0.5      # FIN_03: flag if >50% of debt due within 24 months
DSO_GROWTH_FLAG: float = 0.15        # EQ_08: flag if DSO grew >15% ...
REVENUE_FLAT_BAND: float = 0.05      # EQ_08: ... while revenue moved <5%

# Financial-statement section concepts to read for Layer 1 (resolved against the
# filing's TOC via japan_utils).
FIN_SECTION_CONCEPTS: list[str] = [
    "連結損益計算書", "連結貸借対照表", "連結キャッシュ・フロー計算書",
    "借入金等明細表", "主要な経営指標等の推移",
]

# A taxonomy category/signal object, and a per-signal extraction record.
Category = dict[str, Any]
Record = dict[str, Any]

SYSTEM_MESSAGE: str = (
    "You are a financial analyst extracting company-specific risk signals from a "
    "Japanese securities report (有価証券報告書). Ground every claim in the provided "
    "text and quote the source verbatim. If the text does not support a signal, "
    "say so — never invent facts."
)

# --- Prompts (f-string constants at top, max 5 per file; 2 used this pass) ---

SECTION_SELECT_PROMPT = """You are mapping risk-analysis topics to the sections of a filing for {company_name}.

Risk category: {category_name}

These topics were NOT matched to a section by exact name, so pick the closest sections from the filing's table of contents below (Japanese 有価証券報告書). Only choose titles that genuinely cover the topic; if none fit, return an empty list.

Unmatched topics:
{unmatched_topics}

Table of contents titles (choose from these EXACT strings only):
{toc_titles}

Return JSON, titles copied exactly as written above:
```json
{{"sections": ["...", "..."]}}
```
"""

NLP_EXTRACT_PROMPT = """You are extracting risk signals for {company_name} from a Japanese securities report (filing category: {category_name}).

For EACH signal below, search the provided section text for relevant evidence.

Signals (id | name: what to extract):
{signals_block}

Section text (each block prefixed with "### SECTION: <title>"):
---
{section_text}
---

Rules:
- Report only figures and facts STATED in the text. Do NOT calculate ratios,
  growth rates, or estimates — report the raw disclosed numbers.
- The "passage" must be a CONTIGUOUS span copied character-for-character from a
  single section. Do not merge separate lines/rows, do not insert separators
  like "/", and do not add years or commentary that are not at that exact spot.

Return JSON mapping every signal id to an object. When evidence is found:
  {{"found": true, "value": <short string or number, the key figure/finding as disclosed>,
    "flag": <true if this indicates elevated risk, else false>,
    "evidence_en": <one-sentence English summary>,
    "section": <the exact "### SECTION:" title the evidence came from>,
    "passage": <contiguous verbatim Japanese quote, <= {max_passage} characters>}}
When the text does not support the signal:
  {{"found": false}}

Output only the JSON, copying section titles exactly:
```json
{{"{first_signal_id}": {{"found": false}}}}
```
"""

FINANCIALS_EXTRACT_PROMPT = """You are transcribing financial line items for {company_name} from the statement sections of a Japanese securities report (有価証券報告書). Do NOT calculate anything — only copy figures that are explicitly printed.

Section text (each block prefixed with "### SECTION: <title>"):
---
{section_text}
---

Report every monetary figure in ONE consistent unit (the statements' 単位, e.g. 千円). For each item give the current-year and prior-year value when both are shown (use null when an item is absent). Include the section title and a short verbatim Japanese passage so each figure can be cited.

Return JSON exactly in this shape (null for anything not disclosed):
```json
{{
  "currency_unit": "千円",
  "items": {{
    "revenue":                 {{"current": null, "prior": null, "section": null, "passage": null}},
    "cogs":                    {{"current": null, "prior": null, "section": null, "passage": null}},
    "operating_income":        {{"current": null, "prior": null, "section": null, "passage": null}},
    "depreciation":            {{"current": null, "prior": null, "section": null, "passage": null}},
    "capex":                   {{"current": null, "section": null, "passage": null}},
    "interest_expense":        {{"current": null, "section": null, "passage": null}},
    "cash":                    {{"current": null, "section": null, "passage": null}},
    "short_term_debt":         {{"current": null, "section": null, "passage": null}},
    "long_term_debt":          {{"current": null, "section": null, "passage": null}},
    "current_portion_lt_debt": {{"current": null, "section": null, "passage": null}},
    "bonds":                   {{"current": null, "section": null, "passage": null}},
    "accounts_receivable":     {{"current": null, "prior": null, "section": null, "passage": null}},
    "inventory":               {{"current": null, "prior": null, "section": null, "passage": null}},
    "accounts_payable":        {{"current": null, "prior": null, "section": null, "passage": null}},
    "debt_due_within_1y":      {{"current": null, "section": null, "passage": null}},
    "debt_due_1_to_2y":        {{"current": null, "section": null, "passage": null}}
  }},
  "revenue_history": [{{"year": "YYYY/M", "value": null}}]
}}
```
"""


def _norm_ws(s: str | None) -> str:
    """Drop all whitespace (incl. fullwidth space) for substring grounding checks."""
    return "".join((s or "").split()).replace("　", "")


class Extractor:
    """Layer-2 NLP extractor over a single IngestedDoc."""

    def __init__(self, doc: IngestedDoc, taxonomy: dict[str, Any], client: Any,
                 model: str = DEFAULT_MODEL, company_name: str | None = None,
                 log_path: str | None = None) -> None:
        self.doc: IngestedDoc = doc
        self.taxonomy: dict[str, Any] = taxonomy   # the "csrp_framework" dict
        self.client: Any = client                  # None => deterministic selection only
        self.model: str = model
        self.company_name: str = company_name or os.path.basename(doc.path)
        self.log_path: str | None = log_path
        self._last_selection: dict[str, Any] = {}

    # --- selection ----------------------------------------------------------

    def _category_terms(self, category: Category) -> list[str]:
        """Unique source_sections_ja concepts across a category's signals."""
        terms: list[str] = []
        for sig in category["signals"]:
            for t in sig.get("source_sections_ja", []):
                if t not in terms:
                    terms.append(t)
        return terms

    def _llm_select_sections(self, category: Category, unmatched: list[str],
                             toc_titles: list[str]) -> list[str]:
        """LLM fallback: pick TOC titles for concepts the matcher missed."""
        prompt = SECTION_SELECT_PROMPT.format(
            company_name=self.company_name,
            category_name=category["name_en"],
            unmatched_topics="\n".join(f"- {t}" for t in unmatched),
            toc_titles="\n".join(f"- {t}" for t in toc_titles),
        )
        resp, _ = get_response_from_llm(
            prompt, client=self.client, model=self.model,
            system_message=SYSTEM_MESSAGE, temperature=0.0,
        )
        parsed = extract_json_between_markers(resp) or {}
        picked = parsed.get("sections", []) if isinstance(parsed, dict) else []
        valid = set(toc_titles)
        return [t for t in picked if t in valid]

    def _get_relevant_sections(self, category: Category) -> list[str]:
        """Return the exact TOC titles relevant to a category (deterministic
        matches + LLM fallback for the rest). Records selection meta for logs."""
        toc_titles = self.doc.section_titles()
        terms = self._category_terms(category)
        matched, unmatched = select_sections(terms, toc_titles)

        titles = list(dict.fromkeys(matched.values()))  # dedup, keep order
        fallback: list[str] = []
        if unmatched and self.client is not None:
            fallback = self._llm_select_sections(category, unmatched, toc_titles)
        for t in fallback:
            if t not in titles:
                titles.append(t)

        self._last_selection = {
            "category": category["id"],
            "deterministic": matched,
            "unmatched": unmatched,
            "llm_fallback": fallback,
            "selected": titles,
        }
        return titles

    # --- extraction ---------------------------------------------------------

    def _gather_text(self, titles: list[str]) -> str:
        """Concatenate selected sections (with title headers), capped."""
        return gather_section_text(self.doc, titles)

    def _unscored(self, reason: str, sections: list[str]) -> Record:
        return {
            "value": None, "flag": None, "evidence_en": None, "citation": None,
            "sections_used": sections, "layer": "nlp",
            "status": "unscored", "reason": reason,
        }

    def _build_record(self, raw: dict[str, Any] | None, titles: list[str],
                      section_text: str) -> Record:
        """Turn one signal's raw LLM object into a grounded record."""
        if not raw or not raw.get("found"):
            return self._unscored("no supporting evidence in selected sections", titles)
        passage = (raw.get("passage") or "")[:MAX_PASSAGE_CHARS]
        if not passage:
            return self._unscored("evidence claimed but no citation passage returned", titles)
        section = raw.get("section") if raw.get("section") in titles else (titles[0] if titles else None)
        verified = _norm_ws(passage) in _norm_ws(section_text)
        return {
            "value": raw.get("value"),
            "flag": raw.get("flag"),
            "evidence_en": raw.get("evidence_en"),
            "citation": {
                "citation_type": "nlp",
                "doc": os.path.basename(self.doc.path),
                "section": section,
                "passage": passage,
                "passage_verified": verified,
            },
            "sections_used": titles,
            "layer": "nlp",
            "status": "scored",
        }

    def extract_category(self, category: Category) -> dict[str, Record]:
        """Extract every signal in one category. Returns {signal_id: record}."""
        titles = self._get_relevant_sections(category)
        signals = category["signals"]
        section_text = self._gather_text(titles)

        if not section_text.strip():
            records = {s["id"]: self._unscored("no relevant sections found", titles)
                       for s in signals}
            self._log(category, titles, llm_called=False)
            return records

        signals_block = "\n".join(
            f"- {s['id']} | {s['name_en']}: {s['extraction_target_en']}" for s in signals
        )
        prompt = NLP_EXTRACT_PROMPT.format(
            company_name=self.company_name,
            category_name=category["name_en"],
            signals_block=signals_block,
            section_text=section_text,
            max_passage=MAX_PASSAGE_CHARS,
            first_signal_id=signals[0]["id"],
        )
        resp, _ = get_response_from_llm(
            prompt, client=self.client, model=self.model,
            system_message=SYSTEM_MESSAGE, temperature=0.0,
        )
        parsed = extract_json_between_markers(resp) or {}

        records: dict[str, Record] = {}
        for s in signals:
            raw = parsed.get(s["id"]) if isinstance(parsed, dict) else None
            records[s["id"]] = self._build_record(raw, titles, section_text)
        self._log(category, titles, llm_called=True)
        return records

    def extract_all(self, include_jpn: bool = True) -> dict[str, Record]:
        """Run every category. Returns {signal_id: record}."""
        out: dict[str, Record] = {}
        for category in self.taxonomy["categories"]:
            if category.get("supplemental") and not include_jpn:
                continue
            out.update(self.extract_category(category))
        return out

    # --- logging ------------------------------------------------------------

    def _log(self, category: Category, titles: list[str], llm_called: bool) -> None:
        if not self.log_path:
            return
        sel = self._last_selection
        entry = {
            "stage": "extraction",
            "category": category["id"],
            "selected_sections": titles,
            "deterministic_hits": list(sel.get("deterministic", {}).keys()),
            "llm_fallback_sections": sel.get("llm_fallback", []),
            "llm_called": llm_called,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# Layer 1 — structured / quantitative extraction
# =============================================================================
#
# The LLM only transcribes printed line items (with citations); ALL arithmetic
# and threshold logic below is pure Python — so the math is deterministic and
# auditable. Records are tagged `"layer": "structured"` and supersede the Layer-2
# NLP records for the same quantitative signal ids during the later merge.


def _val(item: dict[str, Any] | None, which: str = "current") -> float | None:
    """Read a numeric value from a financials item dict, or None if absent."""
    if not item:
        return None
    v = item.get(which)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sum_present(values: list[float | None]) -> tuple[float, bool]:
    """Sum the non-None values; also report whether any value was present."""
    present = [v for v in values if v is not None]
    return (sum(present), len(present) > 0)


class StructuredExtractor:
    """Layer-1 extractor: pull cited line items via one LLM call, then compute
    ratios and hard-threshold flags in pure Python."""

    def __init__(self, doc: IngestedDoc, client: Any, model: str = DEFAULT_MODEL,
                 company_name: str | None = None) -> None:
        self.doc: IngestedDoc = doc
        self.client: Any = client
        self.model: str = model
        self.company_name: str = company_name or os.path.basename(doc.path)

    # --- line-item extraction ----------------------------------------------

    def _fin_section_titles(self) -> list[str]:
        """Resolve the financial-statement section concepts to this filing's TOC."""
        toc_titles = self.doc.section_titles()
        titles: list[str] = []
        for concept in FIN_SECTION_CONCEPTS:
            hit = match_section(concept, toc_titles)
            if hit and hit not in titles:
                titles.append(hit)
        return titles

    def extract_financials(self) -> dict[str, Any]:
        """One LLM call that transcribes the line items (no math). Returns the
        parsed financials dict (currency_unit / items / revenue_history)."""
        titles = self._fin_section_titles()
        section_text = gather_section_text(self.doc, titles)
        prompt = FINANCIALS_EXTRACT_PROMPT.format(
            company_name=self.company_name, section_text=section_text,
        )
        resp, _ = get_response_from_llm(
            prompt, client=self.client, model=self.model,
            system_message=SYSTEM_MESSAGE, temperature=0.0,
        )
        parsed = extract_json_between_markers(resp) or {}
        parsed.setdefault("items", {})
        parsed.setdefault("revenue_history", [])
        parsed["_sections_used"] = titles
        return parsed

    # --- record helpers -----------------------------------------------------

    def _cite(self, items: dict[str, Any], names: list[str]) -> dict[str, Any]:
        """Build a structured citation from the line items a metric was built on."""
        inputs = []
        for name in names:
            it = items.get(name)
            if it:
                inputs.append({
                    "line_item": name,
                    "value": it.get("current"),
                    "section": it.get("section"),
                    "passage": (it.get("passage") or "")[:MAX_PASSAGE_CHARS],
                })
        return {"citation_type": "structured", "doc": os.path.basename(self.doc.path),
                "inputs": inputs}

    def _record(self, value: Any, flag: bool | None, evidence_en: str,
                evidence_ja: str, items: dict[str, Any], inputs: list[str]) -> Record:
        return {
            "value": value, "flag": flag,
            "evidence_en": evidence_en, "evidence_ja": evidence_ja,
            "citation": self._cite(items, inputs),
            "sections_used": None, "layer": "structured", "status": "scored",
        }

    def _unscored(self, reason: str) -> Record:
        return {
            "value": None, "flag": None, "evidence_en": None, "citation": None,
            "sections_used": None, "layer": "structured",
            "status": "unscored", "reason": reason,
        }

    # --- the math (pure Python, deterministic) ------------------------------

    def compute_signals(self, fin: dict[str, Any]) -> dict[str, Record]:
        """Compute Layer-1 signal records from transcribed line items."""
        items: dict[str, Any] = fin.get("items", {})
        out: dict[str, Record] = {}

        rev_c, rev_p = _val(items.get("revenue")), _val(items.get("revenue"), "prior")
        cogs_c, cogs_p = _val(items.get("cogs")), _val(items.get("cogs"), "prior")
        opinc_c = _val(items.get("operating_income"))
        dep_c = _val(items.get("depreciation"))
        interest_c = _val(items.get("interest_expense"))
        cash_c = _val(items.get("cash"))
        ar_c, ar_p = _val(items.get("accounts_receivable")), _val(items.get("accounts_receivable"), "prior")
        inv_c = _val(items.get("inventory"))
        ap_c = _val(items.get("accounts_payable"))
        capex_c = _val(items.get("capex"))

        ebitda_c = (opinc_c + dep_c) if (opinc_c is not None and dep_c is not None) else None
        total_debt, has_debt = _sum_present([
            _val(items.get("short_term_debt")), _val(items.get("long_term_debt")),
            _val(items.get("current_portion_lt_debt")), _val(items.get("bonds")),
        ])

        # FIN_01 — Net Debt / EBITDA (flag > 4.0x)
        if ebitda_c and ebitda_c != 0 and has_debt and cash_c is not None:
            ratio = (total_debt - cash_c) / ebitda_c
            out["FIN_01"] = self._record(
                round(ratio, 2), ratio > NET_DEBT_EBITDA_FLAG,
                f"Net Debt/EBITDA ≈ {ratio:.2f}x (threshold {NET_DEBT_EBITDA_FLAG}x).",
                f"ネット有利子負債/EBITDA ≈ {ratio:.2f}倍（閾値 {NET_DEBT_EBITDA_FLAG}倍）。",
                items, ["short_term_debt", "long_term_debt", "current_portion_lt_debt", "bonds", "cash", "operating_income", "depreciation"])
        else:
            out["FIN_01"] = self._unscored("missing debt/cash/EBITDA inputs")

        # FIN_02 — EBITDA / Interest (flag < 3.0x)
        if ebitda_c is not None and interest_c and interest_c != 0:
            cov = ebitda_c / interest_c
            out["FIN_02"] = self._record(
                round(cov, 2), cov < EBITDA_INTEREST_FLAG,
                f"EBITDA/Interest ≈ {cov:.2f}x (threshold {EBITDA_INTEREST_FLAG}x).",
                f"EBITDA/支払利息 ≈ {cov:.2f}倍（閾値 {EBITDA_INTEREST_FLAG}倍）。",
                items, ["operating_income", "depreciation", "interest_expense"])
        else:
            out["FIN_02"] = self._unscored("missing EBITDA or interest expense")

        # FIN_03 — maturity wall within 24 months (flag if >50% of debt)
        within24, has_24 = _sum_present([
            _val(items.get("debt_due_within_1y")), _val(items.get("debt_due_1_to_2y"))])
        if not has_24:  # fallback to short-term + current portion of LT
            within24, has_24 = _sum_present([
                _val(items.get("short_term_debt")), _val(items.get("current_portion_lt_debt"))])
        if has_24 and has_debt and total_debt != 0:
            share = within24 / total_debt
            out["FIN_03"] = self._record(
                f"{within24:.0f} of {total_debt:.0f} ({share:.0%}) due ≤24m",
                share > MATURITY_WALL_FLAG,
                f"{share:.0%} of debt matures within 24 months (threshold {MATURITY_WALL_FLAG:.0%}).",
                f"有利子負債の{share:.0%}が24か月以内に返済期限（閾値 {MATURITY_WALL_FLAG:.0%}）。",
                items, ["debt_due_within_1y", "debt_due_1_to_2y", "short_term_debt", "current_portion_lt_debt"])
        else:
            out["FIN_03"] = self._unscored("missing debt maturity inputs")

        # FIN_06 — working-capital days (DSO/DIO/DPO), informational (no flag)
        dso = (ar_c / rev_c * 365) if (ar_c is not None and rev_c) else None
        dio = (inv_c / cogs_c * 365) if (inv_c is not None and cogs_c) else None
        dpo = (ap_c / cogs_c * 365) if (ap_c is not None and cogs_c) else None
        if any(x is not None for x in (dso, dio, dpo)):
            parts = {"DSO": dso, "DIO": dio, "DPO": dpo}
            out["FIN_06"] = self._record(
                {k: round(v, 1) for k, v in parts.items() if v is not None}, None,
                "Working-capital days computed from balance sheet and income statement.",
                "貸借対照表・損益計算書から算出した運転資本回転日数（DSO/DIO/DPO）。",
                items, ["accounts_receivable", "inventory", "accounts_payable", "revenue", "cogs"])
        else:
            out["FIN_06"] = self._unscored("missing working-capital inputs")

        # EQ_08 — DSO rising while revenue flat (2-year approximation)
        if all(x is not None and x != 0 for x in (ar_p, rev_p)) and ar_c is not None and rev_c:
            dso_c, dso_p = ar_c / rev_c, ar_p / rev_p
            dso_growth = (dso_c - dso_p) / dso_p if dso_p else 0.0
            rev_growth = (rev_c - rev_p) / rev_p if rev_p else 0.0
            flag = dso_growth > DSO_GROWTH_FLAG and abs(rev_growth) < REVENUE_FLAT_BAND
            out["EQ_08"] = self._record(
                f"DSO {dso_growth:+.0%}, revenue {rev_growth:+.0%} YoY (2yr approx.)", flag,
                f"DSO moved {dso_growth:+.0%} vs revenue {rev_growth:+.0%} year-over-year "
                "(3yr trend not available from a single filing).",
                f"DSOは前年比{dso_growth:+.0%}、売上高は{rev_growth:+.0%}"
                "（単一の有報では3年趨勢は取得不可、2年近似）。",
                items, ["accounts_receivable", "revenue"])
        else:
            out["EQ_08"] = self._unscored("need current+prior receivables and revenue")

        # REV_06 — revenue volatility (σ of YoY growth) from the 5-year history
        history = [h.get("value") for h in fin.get("revenue_history", []) if h.get("value") is not None]
        if len(history) >= 3:
            growths = [(history[i] - history[i - 1]) / history[i - 1]
                       for i in range(1, len(history)) if history[i - 1]]
            if growths:
                mean = sum(growths) / len(growths)
                sigma = (sum((g - mean) ** 2 for g in growths) / len(growths)) ** 0.5
                out["REV_06"] = self._record(
                    f"σ(YoY growth) ≈ {sigma:.1%} over {len(history)} years", None,
                    f"Revenue YoY-growth volatility ≈ {sigma:.1%}.",
                    f"売上高前年比成長率のボラティリティ（σ）≈ {sigma:.1%}（{len(history)}年）。",
                    items, ["revenue"])
        out.setdefault("REV_06", self._unscored("need ≥3 years of revenue history"))

        # MKT_04 — gross-margin trend (current vs prior), informational
        if all(x is not None and x != 0 for x in (rev_c, rev_p)) and cogs_c is not None and cogs_p is not None:
            gm_c, gm_p = (rev_c - cogs_c) / rev_c, (rev_p - cogs_p) / rev_p
            out["MKT_04"] = self._record(
                f"gross margin {gm_p:.1%} → {gm_c:.1%} ({(gm_c - gm_p) * 100:+.1f} pts)", None,
                "Gross-margin change year-over-year.",
                f"売上総利益率 {gm_p:.1%} → {gm_c:.1%}（前年比 {(gm_c - gm_p) * 100:+.1f}pt）。",
                items, ["revenue", "cogs"])
        else:
            out["MKT_04"] = self._unscored("missing revenue/COGS for both years")

        # OPS_03 — capex intensity (Capex / Revenue), informational
        if capex_c is not None and rev_c:
            out["OPS_03"] = self._record(
                f"Capex/Revenue ≈ {capex_c / rev_c:.1%}", None,
                "Capital-expenditure intensity from the cash-flow statement.",
                f"設備投資/売上高 ≈ {capex_c / rev_c:.1%}（キャッシュ・フロー計算書）。",
                items, ["capex", "revenue"])
        else:
            out["OPS_03"] = self._unscored("missing capex or revenue")

        return out

    def run(self) -> dict[str, Record]:
        """Extract line items then compute all Layer-1 signal records."""
        fin = self.extract_financials()
        return self.compute_signals(fin)


def gather_section_text(doc: IngestedDoc, titles: list[str],
                        cap: int = MAX_SECTION_CHARS) -> str:
    """Concatenate the given sections' text (with title headers), capped."""
    parts: list[str] = []
    total = 0
    for title in titles:
        body = doc.section_text(title) or ""
        chunk = f"\n### SECTION: {title}\n{body}\n"
        if total + len(chunk) > cap:
            chunk = chunk[: max(0, cap - total)]
        parts.append(chunk)
        total += len(chunk)
        if total >= cap:
            break
    return "".join(parts)


# --- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from .perform_ingestion import ingest
    from .llm import create_client

    here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    taxonomy = json.load(open(os.path.join(here, "taxonomy.json")))["csrp_framework"]
    pdfs = ["SAAF.pdf", "kddi.pdf", "yamaura.pdf"]

    # Part A — selection only (free, no LLM): confirm >=1 section per category.
    print("### Part A — deterministic section selection across all filings\n")
    for name in pdfs:
        doc = ingest(os.path.join(here, "data", name))
        ext = Extractor(doc, taxonomy, client=None, company_name=name)
        print(f"== {name} ({doc.page_count}p) ==")
        for cat in taxonomy["categories"]:
            titles = ext._get_relevant_sections(cat)
            mark = "" if titles else "  <-- NONE"
            print(f"  {cat['id']:5} {len(titles)} sections{mark}: {titles[:3]}{' …' if len(titles) > 3 else ''}")
        print()

    # Part B — real Layer-2 extraction (costs tokens). Gate behind --llm.
    if "--llm" in sys.argv:
        print("### Part B — Layer 2 NLP extraction on SAAF.pdf (MGMT, FIN)\n")
        doc = ingest(os.path.join(here, "data", "SAAF.pdf"))
        client, model = create_client(DEFAULT_MODEL)
        ext = Extractor(doc, taxonomy, client=client, model=model,
                        company_name="SAAF Co., Ltd.")
        by_id = {c["id"]: c for c in taxonomy["categories"]}
        for cid in ["MGMT", "FIN"]:
            print(f"== category {cid} ==")
            recs = ext.extract_category(by_id[cid])
            for sid, r in recs.items():
                if r["status"] == "scored":
                    c = r["citation"]
                    print(f"  {sid} [SCORED flag={r['flag']} verified={c['passage_verified']}]")
                    print(f"     value: {r['value']}")
                    print(f"     cite : {c['section']} | {c['passage'][:60]}…")
                else:
                    print(f"  {sid} [UNSCORED] {r['reason']}")
            print()

        # Part C — Layer 1 structured/quantitative extraction on SAAF.
        print("### Part C — Layer 1 structured extraction on SAAF.pdf\n")
        structured = StructuredExtractor(doc, client=client, model=model,
                                         company_name="SAAF Co., Ltd.")
        fin = structured.extract_financials()
        print(f"currency_unit: {fin.get('currency_unit')}; "
              f"sections: {fin.get('_sections_used')}")
        for sid, r in structured.compute_signals(fin).items():
            if r["status"] == "scored":
                n = len(r["citation"]["inputs"])
                print(f"  {sid} [SCORED flag={r['flag']}] {r['value']}  ({n} cited inputs)")
            else:
                print(f"  {sid} [UNSCORED] {r['reason']}")
    else:
        print("(Part B/C skipped — pass --llm to run the live extraction on SAAF.)")
