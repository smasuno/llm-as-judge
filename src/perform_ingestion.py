"""
perform_ingestion.py — Document loader for the CSRP pipeline (build-order step 5).

Job: load a filing, extract its table of contents, and return a section-keyed
structure the extraction layer can query. This is the foundation of the
TOC-first reading strategy — we never hand the LLM the whole document; the
extraction layer picks the few sections relevant to each risk category and
pulls only those via `section_text()`.

TOC strategy (cheapest reliable source first):
  1. Embedded outline — `doc.get_toc()`. Japanese 有価証券報告書 filings ship a
     complete, reliable outline with page numbers. This is the normal path.
  2. LLM fallback — only if no embedded outline exists, read the first pages
     and ask the model to reconstruct the TOC. Costs tokens, so it is a
     fallback, not the default.

Page numbers are 1-based (matching pymupdf's TOC), inclusive ranges.
"""

from typing import Any

import pymupdf
from .llm import get_response_from_llm, extract_json_between_markers

DEFAULT_MODEL: str = "claude-sonnet-4-6"

# A TOC outline entry as returned by pymupdf: (level, title, page_start).
TocEntry = tuple[int, str, int]
# A resolved section: {"title": str, "level": int, "page_start": int, "page_end": int}.
Section = dict[str, Any]

# --- Prompts (f-string constants, max 5 per file; 1 used here) ---------------

EXTRACT_TOC_PROMPT = """You are reading the opening pages of a corporate financial filing ({page_count} pages total). Below is the extracted text of the first pages, which typically contain the table of contents (目次).

Extract the table of contents as a flat list of sections in document order. For each section give its title exactly as written and the page number it starts on.

First pages text:
---
{first_pages_text}
---

Return JSON in this exact shape (no commentary):
```json
{{"toc": [{{"title": "...", "page_start": 1}}, {{"title": "...", "page_start": 3}}]}}
```
"""


# --- Data structures ---------------------------------------------------------


class IngestedDoc:
    """A loaded filing plus its section map. Text is sliced on demand so we
    only materialize the sections the extraction layer actually asks for."""

    def __init__(self, path: str, page_texts: list[str], sections: list[Section],
                 toc_source: str) -> None:
        self.path: str = path
        self.page_texts: list[str] = page_texts   # 0-indexed list of per-page text
        self.page_count: int = len(page_texts)
        self.sections: list[Section] = sections    # section dicts (no body text)
        self.toc_source: str = toc_source          # "embedded" | "llm"

    def toc(self) -> list[Section]:
        """Lightweight TOC for the section-selection prompt — no body text."""
        return self.sections

    def section_titles(self) -> list[str]:
        return [s["title"] for s in self.sections]

    def get_section(self, title: str) -> Section | None:
        """Exact-title lookup; returns the section dict or None."""
        for s in self.sections:
            if s["title"] == title:
                return s
        return None

    def section_text(self, title: str) -> str | None:
        """Return the text of a section by exact title (pages page_start..page_end,
        inclusive). Returns None if the title is not in the TOC."""
        s = self.get_section(title)
        if s is None:
            return None
        start, end = s["page_start"], s["page_end"]
        return "\n".join(self.page_texts[start - 1:end]).strip()


# --- PDF loading -------------------------------------------------------------


def load_pages(pdf_path: str) -> list[str]:
    """Return a list of per-page plain text (0-indexed)."""
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


def _embedded_toc(pdf_path: str) -> list[TocEntry]:
    """Return the embedded outline as a list of (level, title, page_start)
    tuples, or [] if the PDF has no outline."""
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()  # list of [level, title, page_start]
    doc.close()
    return [(lvl, title.strip(), page) for lvl, title, page in toc]


def _assign_page_ranges(entries: list[TocEntry], page_count: int) -> list[Section]:
    """Given outline entries [(level, title, page_start), ...] in document
    order, compute each section's inclusive page_end.

    A section spans until the next entry at the same or higher level (its next
    sibling or an ancestor's sibling), so a parent section's range covers all
    of its children. Leaf and parent ranges therefore overlap by design — the
    extraction layer can ask for a broad parent or a narrow leaf and get the
    right text either way."""
    sections: list[Section] = []
    for i, (level, title, page_start) in enumerate(entries):
        page_end = page_count
        for j in range(i + 1, len(entries)):
            next_level = entries[j][0]
            if next_level <= level:
                page_end = max(page_start, entries[j][2] - 1)
                break
        sections.append({
            "title": title,
            "level": level,
            "page_start": page_start,
            "page_end": page_end,
        })
    return sections


def _extract_toc_with_llm(page_texts: list[str], client: Any, model: str,
                          num_first_pages: int = 6) -> list[TocEntry]:
    """Fallback: reconstruct a flat TOC from the first pages via the LLM.
    Returns outline entries [(level, title, page_start), ...] (all level 1)."""
    first_pages_text = "\n".join(page_texts[:num_first_pages])
    prompt = EXTRACT_TOC_PROMPT.format(
        page_count=len(page_texts),
        first_pages_text=first_pages_text,
    )
    response, _ = get_response_from_llm(
        prompt,
        client=client,
        model=model,
        system_message="You extract tables of contents from financial filings as precise JSON.",
        temperature=0.0,
    )
    parsed = extract_json_between_markers(response)
    if not parsed or "toc" not in parsed:
        return []
    return [(1, e["title"].strip(), int(e["page_start"])) for e in parsed["toc"]]


# --- Public entry point ------------------------------------------------------


def extract_toc(pdf_path: str, page_texts: list[str] | None = None,
                client: Any = None, model: str = DEFAULT_MODEL) -> tuple[str, list[Section]]:
    """Return (toc_source, sections). Prefers the embedded outline; falls back
    to the LLM only when no outline exists (and a client is supplied)."""
    if page_texts is None:
        page_texts = load_pages(pdf_path)
    page_count = len(page_texts)

    entries = _embedded_toc(pdf_path)
    toc_source = "embedded"
    if not entries:
        if client is None:
            raise ValueError(
                f"{pdf_path} has no embedded TOC and no LLM client was provided "
                "for the fallback. Pass client=create_client(model)[0]."
            )
        entries = _extract_toc_with_llm(page_texts, client, model)
        toc_source = "llm"

    sections = _assign_page_ranges(entries, page_count)
    return toc_source, sections


def ingest(pdf_path: str, client: Any = None, model: str = DEFAULT_MODEL) -> IngestedDoc:
    """Load a filing and return an IngestedDoc (pages + section map)."""
    page_texts = load_pages(pdf_path)
    toc_source, sections = extract_toc(
        pdf_path, page_texts=page_texts, client=client, model=model
    )
    return IngestedDoc(pdf_path, page_texts, sections, toc_source)


# --- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sample = os.path.join(here, "data", "SAAF.pdf")

    doc = ingest(sample)
    print(f"file:        {os.path.basename(doc.path)}")
    print(f"pages:       {doc.page_count}")
    print(f"toc_source:  {doc.toc_source}")
    print(f"sections:    {len(doc.sections)}")
    print()
    print("First 12 TOC entries (indent = level):")
    for s in doc.sections[:12]:
        indent = "  " * (s["level"] - 1)
        print(f"  {indent}[p{s['page_start']}-{s['page_end']}] {s['title']}")
    print()

    target = "３事業等のリスク"
    text = doc.section_text(target)
    print(f"section_text('{target}') -> {len(text)} chars, first 200:")
    print(text[:200])
