# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A proof-of-concept agentic pipeline that ingests Japanese corporate financial
filings (有価証券報告書), extracts Company-Specific Risk Premium (CSRP /
会社固有リスクプレミアム) signals across 8 risk categories, scores each signal
1–5, and produces a grounded CSRP report (a bps premium added to WACC) in both
English and Japanese. It adapts Sakana's AI-Scientist automated-reviewer
pipeline to financial-risk extraction for M&A valuation.

This is interview/POC code, not a product. The governing constraints (from the
parent `../../CLAUDE.md`) override default behavior:
- **Keep it simple and readable.** Prefer clarity over abstraction; no premature
  generalization. Avoid a sprawling multi-file `src/` tree.
- **Small changes at a time.** Expect many iterations. Prefer starting a fresh
  approach over large rewrites of existing code.
- **Everything must be explainable** — the author has to defend every line.

## Current state (important — read before assuming structure)

The repo is a **skeleton**, far less built-out than the design spec implies:

```
taxonomy.json          # CSRP signal schema (the real one; see divergences below)
data/                  # 3 sample filings: kddi.pdf, SAAF.pdf, yamaura.pdf
2023-international-valuation-guide-to-cost-of-capital.pdf   # reference doc
src/
  crsp.ipynb           # the working surface — currently ~2 cells, just started
  llm.py               # AI-Scientist multi-provider client (+ openai-0.28 import fix)
  perform_ingestion.py # BUILT — loader + TOC-first section map (step 5)
  japan_utils.py       # BUILT — 有報 section matcher + seed glossary (step 1)
  perform_extraction.py# BUILT — Layer 2 NLP + Layer 1 structured extraction (step 7)
  perform_review.py    # BUILT — CSRP scoring engine (step 8), adapted from reviewer
  perform_writeup.py   # BUILT — md (LLM) + JSON + DCF-ready Excel output (step 9)
  launch_csrp.py       # BUILT — end-to-end CLI entry point (step 10)
  __init__.py          # empty
```

**Built so far** (bottom-up per the build order): `perform_ingestion.py` →
`japan_utils.py` → `perform_extraction.py` (both layers). Each has a `__main__`
smoke test (run with `python -m src.<module>`). In `perform_extraction.py`:
- `Extractor` = **Layer 2** NLP — per-category cited evidence, `layer:"nlp"`.
- `StructuredExtractor` = **Layer 1** — one LLM call transcribes cited line
  items, then pure-Python `compute_signals()` produces ratios + hard-threshold
  flags (FIN_01/02/03/06, EQ_08, REV_06, MKT_04, OPS_03), `layer:"structured"`.
  Layer 1 records **supersede** the Layer-2 record for the same signal id — the
  caller (launch/scoring) must apply that override during merge. (Demonstrated:
  Layer 2 leaves Net Debt/EBITDA UNSCORED; Layer 1 computes it.)

`perform_review.py` is the **scoring engine** (adapted in place from the
reviewer, mirroring its prompt/THOUGHT-JSON/aggregate structure): `merge_records`
(Layer 1 supersedes Layer 2) → `score_signals` (threshold signals scored in
Python, relative signals via one LLM call per category, anchored 3=peer-avg) →
`aggregate` (category means → weighted composite → `composite_to_bps`, JPN
excluded). Top-level `score()` returns the full result dict. `comparables_mode`
is OFF (`CONFIG`). Run `python -m src.perform_review` for the free deterministic
test; add `--llm` for live end-to-end on SAAF.

Scoring also persists the LLM scorer's full **THOUGHT** to `run_log.jsonl` (pass
`log_path` to `score()`), and emits **bilingual rationale** (`rationale_en`/
`rationale_ja`) at no extra LLM cost — threshold signals reuse the Layer-1
record's `evidence_en`/`evidence_ja`.

`perform_writeup.py` is the **output stage** (step 9), producing per company in
`outputs/<slug>/`:
- `csrp_<slug>_en.json` / `_ja.json` — deterministic machine-readable result.
- `csrp_<slug>.xlsx` — **deterministic, DCF-ready Excel** (`openpyxl`, installed)
  with **"English" + "日本語" tabs**: recommended CSRP = bps-range midpoint (in
  bps and %) + per-category component scores at the top, detailed
  rationale+citations at the bottom. This is the headline analyst deliverable.
- `csrp_<slug>_en.md` / `_ja.md` — **LLM-authored** narrative, section-by-section
  mirroring `../AI-Scientist/ai_scientist/perform_writeup.py` (only produced when
  a `client` is passed; gated behind `--llm`).
Run free (JSON+Excel) with `python -m src.perform_writeup`; add `--llm` for md.

`launch_csrp.py` is the **end-to-end entry point** (step 10). It takes the yūhō
PDF path as a positional arg and runs ingest → extract (L2+L1) → merge → score →
writeup, logging agent reasoning to `outputs/<slug>/run_log.jsonl`:
`python -m src.launch_csrp data/SAAF.pdf --company "SAAF Co., Ltd."`
(`--no-markdown` for JSON+Excel only; `--no-jpn`, `--refine N`, `--filing-date`).
A full run makes many API calls (≈8 NLP categories + 1 structured + per-category
scoring + EN/JA markdown); use `--no-markdown` to cut the largest cost.

Known follow-ups: the 16k-char section cap can truncate the last gathered
section (e.g. REV_06's 5-year history); `filing_date` is passed in (not yet
extracted); `perform_search.py` and the `ALLOWED_MODELS` guard remain stubs/TODO.

`src/perform_review.py` is still an **unmodified copy** from the parent
AI-Scientist repo — a template to adapt, not finished code:
- `perform_review.py` still imports `from ai_scientist.llm import ...`. That
  package does not exist here — the import is **broken** and must be rewired to
  the local `src.llm` (or the file restructured) before it runs.
- It still reviews ML papers (`perform_review`, `get_meta_review`,
  `neurips_form`). The reuse plan: keep `get_response_from_llm` /
  `get_batch_responses_from_llm` / `extract_json_between_markers` /
  `load_paper` as-is; adapt the `perform_review` + `get_meta_review`
  ensemble-and-aggregate pattern in place for 1–5 signal scoring and composite
  aggregation. **The scoring engine stays in `perform_review.py`** — do not
  create a separate `perform_scoring.py`.

Development happens primarily in `src/crsp.ipynb`. Build the pipeline there
incrementally; only promote stable pieces into `.py` files.

## Running

No build system, no `requirements.txt`, no test runner is set up yet. The code
is plain Python driven from the notebook.

- **Use the `ai_scientist` conda env** — the default `python3` (Homebrew) has
  none of the dependencies. The env has pymupdf, pymupdf4llm, pypdf, anthropic,
  openai, etc. installed. Interpreter:
  `/opt/homebrew/Caskroom/miniforge/base/envs/ai_scientist/bin/python`
  (or `conda activate ai_scientist`). It is also the Jupyter kernel for
  `crsp.ipynb`.
- `src/` is a package using **relative imports** (`from .llm import ...`), so
  `import src` works from the repo root. Run a module's smoke test with `-m`
  from the repo root — NOT `cd src && python x.py` (relative imports need the
  package parent):
  `python -m src.perform_ingestion`, `python -m src.japan_utils`,
  `python -m src.perform_extraction` (free section-selection check; add `--llm`
  for real Layer-2 extraction on SAAF, which makes Anthropic calls).
- Code style: all functions/methods are **type-hinted** (args + return).
- `llm.py` had one fix applied: its `@backoff` decorators referenced
  openai≥1.0 exception classes that don't exist in this env's openai 0.28, which
  broke *import*. Now resolved version-safely (`_OPENAI_RETRY_ERRORS`). The
  Claude path is unchanged; `ALLOWED_MODELS` is still not added.
- Extraction grounding: every NLP signal carries a `citation.passage_verified`
  bool — True only when the quoted Japanese passage is a real (whitespace-
  normalized) substring of the cited section. The extract prompt forbids
  computing ratios and requires contiguous verbatim quotes, so quantitative
  signals like Net Debt/EBITDA come back UNSCORED here by design (Layer 1's job).
- Required env vars (set before importing `llm.py` clients): `ANTHROPIC_API_KEY`
  for Claude; `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`,
  `OPENROUTER_API_KEY` only if those providers are used. `ANTHROPIC_API_KEY` is
  already set in the shell.
- PDF loading: `perform_ingestion.load_pages()` uses pymupdf directly for
  per-page text. `load_paper()` in `perform_review.py` (pymupdf4llm →
  pymupdf → pypdf fallbacks) returns whole-doc markdown — use it only when you
  want the full text rather than TOC-addressed sections.

## Model constraint (not yet enforced — apply it)

`src/llm.py` currently ships the full AI-Scientist `AVAILABLE_LLMS` list
(claude-3-5, gpt-4o, gemini, etc.). The project spec restricts models to:

```python
ALLOWED_MODELS = {"claude-sonnet-4-6", "claude-opus-4-8"}
```

`claude-sonnet-4-6` is the default for all extraction/scoring; `claude-opus-4-8`
is for complex multi-section reasoning only. Add the guard in `llm.py`; do not
write a new client — reuse the existing batch/single-query interface.

## CSRP taxonomy (`taxonomy.json`)

Single source of truth for what gets extracted and how it maps to a premium.
Top-level key `csrp_framework` holds: `description`, `scoring_scale` (1=minimal
risk … 3=peer-average … 5=significant), `score_to_premium_bps` (the lookup
table below), and `categories`.

Categories, weights, and signals:

| id | weight | name | signals |
|----|--------|------|---------|
| REV | 0.20 | Revenue & Customer Risk | REV_01–07 |
| MGMT | 0.15 | Management & Key Person Risk | MGMT_01–06 |
| FIN | 0.20 | Financial Health & Capital Structure | FIN_01–09 |
| EQ | 0.15 | Earnings Quality & Accounting | EQ_01–08 |
| OPS | 0.15 | Operational & Business Model | OPS_01–08 |
| LEG | 0.05 | Legal, Regulatory & Compliance | LEG_01–06 |
| MKT | 0.05 | Market Position & Competitive | MKT_01–05 |
| MACRO | 0.05 | Macro & External Sensitivity | MACRO_01–03 |

Weights sum to 1.0. The schema is **bilingual inline**: each signal carries
`name_en`/`name_ja`, `extraction_target_en`/`_ja`, `source_sections_en`/`_ja`;
categories carry `name_en`/`name_ja`; top level has `description_en`/`_ja` and a
bilingual `scoring_scale` (`{"en":..., "ja":...}` per level). `source_sections_ja`
are mapped to 有価証券報告書 section names where an equivalent exists.

Score → premium: 1.0–1.5→0–50bps, 1.5–2.0→50–100, 2.0–2.5→100–150,
2.5–3.0→150–200, 3.0–3.5→200–300, 3.5–4.0→300–400, 4.0–5.0→400–500.
Aggregation: `category_score = mean(signals)`; `composite = Σ(category × weight)`.

Supplemental category **JPN** (JP_01–05: 政策保有株式, メインバンク, 持株会社,
役員構成, APPI) is present with `default_weight: 0` and `"supplemental": true` —
score these 1–5 like any signal but **exclude them from the weighted composite**.
The composite is computed only over categories where `supplemental` is not true
(those weights sum to 1.0).

**Divergence from the spec to be aware of:** the spec names the file
`csrp_taxonomy.json`; here it is `taxonomy.json` at repo root.

## Intended architecture (the design to build toward)

When extending past the skeleton, follow this layering (from the project spec).
Treat it as direction, not as existing code:

- **TOC-first reading.** Filings are hundreds of pages. Never full-text extract
  blindly. (1) Read the table of contents only → map each CSRP category to
  relevant sections; (2) fetch only those sections; (3) fetch specific
  footnotes for signals that need them (pension, debt maturity, contingencies,
  revenue recognition); (4) web-search only if a section is missing/ambiguous.
- **Layer 1 — structured extraction:** compute quantitative signals from parsed
  tables in plain Python (no LLM for the math): Net Debt/EBITDA, EBITDA/Interest,
  Capex/Revenue, DSO/DPO/DIO, revenue-growth σ, margin trend, leverage trend,
  customer concentration. Hard-threshold flags score deterministically (e.g.
  single customer >25% → REV_01; EBITDA/Interest <3.0x → FIN_02; Net
  Debt/EBITDA >4.0x → FIN_01; debt maturity <24mo → FIN_03).
- **Layer 2 — NLP extraction:** LLM reads the TOC-selected free-text sections,
  extracts qualitative evidence **with a citation**.
- **Layer 3 — scoring (in `perform_review.py`):** LLM-judge scores 1–5 (anchor
  3 = sector average), reuse the `perform_review` / `get_meta_review` ensemble
  pattern, then aggregate to bps. `comparables_mode = False` for the POC (peer-financials path raises
  `NotImplementedError`); use web search + LLM priors as the benchmark instead.
- **Citation discipline:** every score needs a structured / NLP / web citation.
  **No citation → mark the signal UNSCORED with a reason. Never hallucinate.**

Conventions for any new `.py` files (from spec):
- Prompts live as **module-level f-string constants at the top of each file**,
  parameterized for any company, **max 5 per file**. No `prompts/` directory.
- EDINET API access and Tavily web search should be **stubbed with a clear
  TODO** if no key/implementation is present.
- All user-facing output in **both English and Japanese**; reports land in
  `outputs/<company>/` (md + json, EN + JA) with reasoning logged to
  `run_log.jsonl`.

Useful glossary: CSRP = 会社固有リスクプレミアム; risk factors = リスク情報;
financial filing = 有価証券報告書.
