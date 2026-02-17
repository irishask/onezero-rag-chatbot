"""
chunking.py — Section-aware chunking with recursive fallback for large sections.

Strategy:
1. PRIMARY: Each H3 section → one chunk, with H2 parent prepended as
   hierarchical context. Preserves semantic coherence (one topic = one chunk).
2. FALLBACK: Sections exceeding LARGE_SECTION_THRESHOLD are sub-split using
   LangChain's RecursiveCharacterTextSplitter. Sub-chunks inherit all metadata
   from the parent section plus a chunk_index for ordering.

Why not naive RecursiveCharacterTextSplitter everywhere?
- Bank docs have structured fee lists and bullet points — splitting mid-list
  makes the chunk meaningless.
- H3 sections average ~500-650 chars, well within embedding model context.
- Only ~14 sections (out of 228) exceed the threshold, so the fallback rarely fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_loader import RawSection
from config import (
    LARGE_SECTION_THRESHOLD,
    RECURSIVE_CHUNK_SIZE,
    RECURSIVE_CHUNK_OVERLAP,
)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A retrieval-ready chunk with text and metadata."""

    text: str
    metadata: dict[str, str | int]
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    def __repr__(self) -> str:
        src = self.metadata.get("source", "?")
        path = self.metadata.get("section_path", "?")
        idx = self.metadata.get("chunk_index", 0)
        return f"Chunk(source={src!r}, path={path!r}, idx={idx}, chars={self.char_count})"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_metadata(
    section: RawSection,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> dict[str, str | int]:
    """Build a metadata dict from a RawSection."""
    return {
        "source": section.source_file,
        "h2": section.h2_heading,
        "h3": section.h3_heading,
        "section_path": section.section_path,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }


def _sub_split(
    section: RawSection,
    chunk_size: int = RECURSIVE_CHUNK_SIZE,
    chunk_overlap: int = RECURSIVE_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a large section into smaller chunks using RecursiveCharacterTextSplitter.

    Each sub-chunk still carries the H2/H3 heading prefix so retrieval
    gets hierarchical context even on partial sections.

    Parameters
    ----------
    section : RawSection
        The oversized section to sub-split.
    chunk_size : int
        Target sub-chunk size in characters.
    chunk_overlap : int
        Overlap between consecutive sub-chunks.

    Returns
    -------
    list[Chunk]
        Sub-chunks with inherited metadata and chunk_index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )

    # Split body content only (not the heading prefix)
    sub_texts: list[str] = splitter.split_text(section.content)
    total = len(sub_texts)

    heading_prefix = f"## {section.h2_heading}\n### {section.h3_heading}\n"

    chunks: list[Chunk] = []
    for idx, sub_text in enumerate(sub_texts):
        chunks.append(
            Chunk(
                text=heading_prefix + sub_text.strip(),
                metadata=_build_metadata(section, chunk_index=idx, total_chunks=total),
            )
        )
    return chunks


# ── Public API ───────────────────────────────────────────────────────────────

def chunk_sections(
    sections: list[RawSection],
    large_threshold: int = LARGE_SECTION_THRESHOLD,
    chunk_size: int = RECURSIVE_CHUNK_SIZE,
    chunk_overlap: int = RECURSIVE_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Convert parsed sections into retrieval-ready chunks.

    Small sections (below threshold) become a single chunk.
    Large sections are recursively sub-split.

    Parameters
    ----------
    sections : list[RawSection]
        Output from document_loader.load_document or load_all_documents.
    large_threshold : int
        Sections with full_text longer than this are recursively sub-split.
    chunk_size : int
        Target size for sub-chunks (only used for large sections).
    chunk_overlap : int
        Overlap for sub-chunks (only used for large sections).

    Returns
    -------
    list[Chunk]
        Flat list of chunks ready for embedding.
    """
    chunks: list[Chunk] = []
    sub_split_count = 0

    for section in sections:
        if section.char_count > large_threshold:
            sub_chunks = _sub_split(section, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)
            sub_split_count += 1
        else:
            chunks.append(
                Chunk(
                    text=section.full_text,
                    metadata=_build_metadata(section),
                )
            )

    print(f"  Chunking complete: {len(sections)} sections → {len(chunks)} chunks")
    print(f"  Sections sub-split: {sub_split_count}")
    return chunks