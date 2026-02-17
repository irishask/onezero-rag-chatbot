"""
document_loader.py — Section-aware Markdown parser for ONE ZERO policy documents.

Reads a Markdown file and extracts every H3 (###) section together with its
parent H2 (##) heading. Each section is returned as a RawSection containing
the full text and rich metadata for downstream chunking / embedding.

Design decisions:
- H2 = topic group, H3 = individual policy section. Mirrors how the bank
  docs are organised (e.g. "Traveling Abroad > Card Assistance abroad").
- Markdown formatting (bullets, links) is preserved — it carries meaning
  for bank policies (fee tables, procedures, contact links).
- Content before the first H3 under an H2 is attached to a synthetic
  H3 named "(General)" so nothing is silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RawSection:
    """One logical section extracted from a Markdown document."""

    source_file: str        # e.g. "cards.md"
    h2_heading: str         # parent H2 heading (topic group)
    h3_heading: str         # H3 heading (section title)
    section_path: str       # "h2 > h3" human-readable breadcrumb
    content: str            # raw Markdown body (excluding heading lines)
    full_text: str          # heading-prefixed text ready for embedding
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.full_text)

    def __repr__(self) -> str:
        return (
            f"RawSection(source={self.source_file!r}, "
            f"path={self.section_path!r}, chars={self.char_count})"
        )


# ── Heading regex ────────────────────────────────────────────────────────────

_H2_RE = re.compile(r"^##\s+(.+)$")
_H3_RE = re.compile(r"^###\s+(.+)$")


# ── Public API ───────────────────────────────────────────────────────────────

def load_document(filepath: str | Path) -> list[RawSection]:
    """Parse a Markdown policy document into a list of RawSection objects.

    Each H3 section becomes one RawSection, with its H2 parent prepended
    as hierarchical context in full_text.

    Parameters
    ----------
    filepath : str | Path
        Path to the .md file.

    Returns
    -------
    list[RawSection]
        One entry per H3 section found in the document.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Document not found: {filepath}")

    source_name = filepath.name
    lines = filepath.read_text(encoding="utf-8").splitlines()

    sections: list[RawSection] = []
    current_h2: str = "(Preamble)"
    current_h3: str | None = None
    buffer: list[str] = []

    def _flush() -> None:
        """Persist accumulated lines as a RawSection."""
        nonlocal current_h3, buffer

        if current_h3 is None:
            # Content before first H3 under this H2
            body = "\n".join(buffer).strip()
            if not body:
                buffer = []
                return
            current_h3 = "(General)"

        body = "\n".join(buffer).strip()
        if not body:
            buffer = []
            current_h3 = None
            return

        section_path = f"{current_h2} > {current_h3}"
        full_text = f"## {current_h2}\n### {current_h3}\n{body}"

        sections.append(
            RawSection(
                source_file=source_name,
                h2_heading=current_h2,
                h3_heading=current_h3,
                section_path=section_path,
                content=body,
                full_text=full_text,
            )
        )
        buffer = []
        current_h3 = None

    for line in lines:
        h2_match = _H2_RE.match(line)
        h3_match = _H3_RE.match(line)

        if h2_match:
            _flush()
            current_h2 = h2_match.group(1).strip()
            current_h3 = None
            buffer = []
        elif h3_match:
            _flush()
            current_h3 = h3_match.group(1).strip()
            buffer = []
        else:
            buffer.append(line)

    _flush()  # close last section

    return sections


def load_all_documents(filepaths: list[str | Path]) -> list[RawSection]:
    """Load and concatenate sections from multiple Markdown documents.

    Parameters
    ----------
    filepaths : list[str | Path]
        Paths to Markdown files.

    Returns
    -------
    list[RawSection]
        Combined sections from all documents, in file order.
    """
    all_sections: list[RawSection] = []
    for fp in filepaths:
        doc_sections = load_document(fp)
        print(f"  Loaded {len(doc_sections)} sections from {Path(fp).name}")
        all_sections.extend(doc_sections)
    print(f"  Total: {len(all_sections)} sections from {len(filepaths)} files")
    return all_sections