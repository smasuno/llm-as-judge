"""
japan_utils.py — helpers for navigating Japanese 有価証券報告書 (securities reports).

Two jobs for the CSRP pipeline:
  1. Section matching — map the taxonomy's `source_sections_ja` concepts (e.g.
     "事業等のリスク", "退職給付関係注記") onto the actual TOC titles of a specific
     filing (e.g. "３事業等のリスク"). TOC titles carry leading numbering and
     filings word headings slightly differently, so a naive substring match is
     weak. We normalize the numbering away and match bidirectionally, with a
     coarse routing rule for footnote concepts.
  2. A seed bilingual glossary for the writeup stage (extended as needed).

Deliberately dependency-free (stdlib only) so it stays easy to test and explain.
"""

import re

# The umbrella node that holds individual financial-statement footnotes. In all
# sample filings the specific notes (退職給付・リース・偶発債務 …) are NOT separate TOC
# entries — they live under this single node — so footnote concepts route here.
NOTE_NODE = "注記事項"

# Leading numbering/markers that prefix TOC titles but carry no meaning for
# matching: ASCII/fullwidth digits, CJK numerals, 第, circled numbers, brackets,
# punctuation, and (fullwidth) spaces.
_LEADING_MARKERS = re.compile(
    r"^[\s　0-9０-９一二三四五六七八九十百千第（）()\[\]【】．・。.\-]+"
)


def normalize_toc_title(title: str) -> str:
    """Strip leading numbering/markers and 【】 brackets from a TOC title.
    "３事業等のリスク" -> "事業等のリスク"; "（６）大株主の状況" -> "大株主の状況"."""
    t = title.replace("【", "").replace("】", "")
    t = _LEADING_MARKERS.sub("", t)
    return t.strip()


def match_section(term: str, toc_titles: list[str]) -> str | None:
    """Find the TOC title that best corresponds to a taxonomy section concept.

    Returns the RAW TOC title (so it can be handed straight to
    `IngestedDoc.section_text()`), or None.

    Matching is bidirectional on normalized text — a concept matches whether it
    contains, or is contained by, a heading. That handles both directions of
    wording drift (e.g. heading "株式の保有状況" inside concept
    "株式の保有状況（政策保有株式）", and concept "事業等のリスク" inside heading "３事業等のリスク").
    Footnote concepts (anything mentioning 注記) that match nothing fall back to
    the umbrella 注記事項 node when the filing has one.
    """
    nterm = term.strip()
    for raw in toc_titles:
        ntitle = normalize_toc_title(raw)
        if not ntitle:
            continue
        if ntitle in nterm or nterm in ntitle:
            return raw
    if "注記" in term:
        for raw in toc_titles:
            if NOTE_NODE in raw:
                return raw
    return None


def select_sections(terms: list[str], toc_titles: list[str]) -> tuple[dict[str, str], list[str]]:
    """Resolve a list of taxonomy section concepts against a filing's TOC.

    Returns (matched, unmatched):
      matched   — dict {concept -> raw TOC title} for concepts we resolved
                   deterministically (deduped on the concept key)
      unmatched — list of concepts with no deterministic hit (the caller hands
                  these to the LLM fallback)
    """
    matched: dict[str, str] = {}
    unmatched: list[str] = []
    for term in terms:
        hit = match_section(term, toc_titles)
        if hit is not None:
            matched[term] = hit
        else:
            unmatched.append(term)
    return matched, unmatched


# --- Seed bilingual glossary (extended by the writeup stage) -----------------

GLOSSARY: dict[str, str] = {
    "CSRP": "会社固有リスクプレミアム",
    "Company-Specific Risk Premium": "会社固有リスクプレミアム",
    "Scoring": "スコアリング",
    "Risk factors": "リスク情報",
    "Financial filing": "有価証券報告書",
}


# --- Self-check --------------------------------------------------------------

if __name__ == "__main__":
    sample_toc = [
        "表紙", "３事業等のリスク", "（６）大株主の状況", "（５）株式の保有状況",
        "注記事項", "借入金等明細表", "セグメント情報",
    ]
    checks = [
        ("事業等のリスク", "３事業等のリスク"),          # concept inside heading
        ("大株主の状況", "（６）大株主の状況"),            # numbering stripped
        ("株式の保有状況（政策保有株式）", "（５）株式の保有状況"),  # heading inside concept
        ("退職給付関係注記", "注記事項"),                 # footnote routing
        ("借入金等明細表", "借入金等明細表"),              # exact
        ("市場リスクに関する開示", None),                 # no real heading -> LLM fallback
    ]
    for term, expected in checks:
        got = match_section(term, sample_toc)
        flag = "ok " if got == expected else "BAD"
        print(f"[{flag}] {term!r:30} -> {got!r}  (expected {expected!r})")
