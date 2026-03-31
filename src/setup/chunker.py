"""Chunking extracted text into meaningful sections for RAG."""

import re
from dataclasses import dataclass, field

from .text_extractor import ExtractedDocument


@dataclass
class RuleChunk:
    """A chunk of rules text with metadata."""

    id: str
    title: str
    content: str
    page_numbers: list[int] = field(default_factory=list)
    parent_section: str | None = None
    chunk_type: str = "section"  # section, subsection, paragraph, table
    source_file: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "page_numbers": self.page_numbers,
            "parent_section": self.parent_section,
            "chunk_type": self.chunk_type,
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RuleChunk":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            page_numbers=data.get("page_numbers", []),
            parent_section=data.get("parent_section"),
            chunk_type=data.get("chunk_type", "section"),
            source_file=data.get("source_file", ""),
        )


@dataclass
class ChunkedRulebook:
    """A rulebook split into chunks."""

    game_name: str
    chunks: list[RuleChunk]
    full_text: str
    source_file: str

    def get_chunk_by_id(self, chunk_id: str) -> RuleChunk | None:
        """Find a chunk by ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def get_chunks_by_type(self, chunk_type: str) -> list[RuleChunk]:
        """Get all chunks of a specific type."""
        return [c for c in self.chunks if c.chunk_type == chunk_type]

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "game_name": self.game_name,
            "chunks": [c.to_dict() for c in self.chunks],
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, data: dict, full_text: str = "") -> "ChunkedRulebook":
        """Create from dictionary."""
        return cls(
            game_name=data["game_name"],
            chunks=[RuleChunk.from_dict(c) for c in data["chunks"]],
            full_text=full_text,
            source_file=data.get("source_file", ""),
        )


# Common section header patterns in rulebooks
HEADER_PATTERNS = [
    # Numbered sections: "1. Setup", "2.1 Components"
    r"^(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]+)$",
    # ALL CAPS headers
    r"^([A-Z][A-Z\s]{2,}[A-Z])$",
    # Title Case with colon: "Setup:", "Components:"
    r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*:)$",
    # Bold/emphasized (markdown style)
    r"^\*\*([^\*]+)\*\*$",
    r"^#+\s+(.+)$",
    # Simple Title Case line (standalone)
    r"^([A-Z][a-z]+(?:\s+[A-Za-z]+){0,4})$",
]


def _is_header(line: str) -> tuple[bool, str | None]:
    """Check if a line is a section header.

    Returns:
        Tuple of (is_header, extracted_title).
    """
    line = line.strip()
    if not line or len(line) > 100:  # Headers shouldn't be too long
        return False, None

    for pattern in HEADER_PATTERNS:
        match = re.match(pattern, line, re.MULTILINE)
        if match:
            title = match.group(1).strip()
            # Filter out things that look like headers but aren't
            if len(title) < 3:
                continue
            if title.lower() in ("the", "a", "an", "and", "or", "but"):
                continue
            return True, title

    return False, None


def _get_header_level(title: str) -> int:
    """Determine the nesting level of a header.

    Returns:
        Level (1 = top level, 2 = subsection, etc.)
    """
    # Numbered headers: count the dots
    if re.match(r"^\d+\.", title):
        dots = title.count(".")
        return dots + 1

    # Markdown headers: count the #
    if title.startswith("#"):
        return len(title) - len(title.lstrip("#"))

    # ALL CAPS is usually top-level
    if title.isupper():
        return 1

    # Default to level 2
    return 2


def chunk_by_sections(
    document: ExtractedDocument,
    game_name: str,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000,
) -> ChunkedRulebook:
    """Chunk a document by detecting section headers.

    Args:
        document: Extracted document to chunk.
        game_name: Name of the game.
        min_chunk_size: Minimum characters per chunk.
        max_chunk_size: Maximum characters per chunk (will split if exceeded).

    Returns:
        ChunkedRulebook with identified sections.
    """
    chunks: list[RuleChunk] = []
    chunk_counter = 0

    # Process each page
    current_section: str | None = None
    current_content: list[str] = []
    current_pages: list[int] = []
    parent_section: str | None = None

    for page in document.pages:
        lines = page.text.split("\n")
        current_pages.append(page.page_number)

        for line in lines:
            is_header, title = _is_header(line)

            if is_header and title:
                # Save previous chunk if it has content
                if current_content:
                    content = "\n".join(current_content).strip()
                    if len(content) >= min_chunk_size:
                        chunk_counter += 1
                        chunks.append(
                            RuleChunk(
                                id=f"chunk_{chunk_counter:03d}",
                                title=current_section or "Introduction",
                                content=content,
                                page_numbers=list(set(current_pages)),
                                parent_section=parent_section,
                                chunk_type="section",
                                source_file=document.source_file,
                            )
                        )

                # Start new section
                level = _get_header_level(title)
                if level == 1:
                    parent_section = None
                elif current_section and level > 1:
                    parent_section = current_section

                current_section = title
                current_content = []
                current_pages = [page.page_number]
            else:
                current_content.append(line)

    # Don't forget the last chunk
    if current_content:
        content = "\n".join(current_content).strip()
        if len(content) >= min_chunk_size:
            chunk_counter += 1
            chunks.append(
                RuleChunk(
                    id=f"chunk_{chunk_counter:03d}",
                    title=current_section or "Final Section",
                    content=content,
                    page_numbers=list(set(current_pages)),
                    parent_section=parent_section,
                    chunk_type="section",
                    source_file=document.source_file,
                )
            )

    # Split any chunks that are too large
    final_chunks = []
    for chunk in chunks:
        if len(chunk.content) > max_chunk_size:
            split_chunks = _split_large_chunk(chunk, max_chunk_size)
            final_chunks.extend(split_chunks)
        else:
            final_chunks.append(chunk)

    # Add chunks for diagrams/images
    if document.images:
        for img in document.images:
            if img.description and len(img.description) > 50:  # Skip trivial descriptions
                final_chunks.append(
                    RuleChunk(
                        id=f"diagram_{img.page_number}_{img.image_index}",
                        title=f"Diagram (Page {img.page_number})",
                        content=img.description,
                        page_numbers=[img.page_number],
                        parent_section=None,
                        chunk_type="diagram",
                        source_file=document.source_file,
                    )
                )

    # Re-number chunks
    for i, chunk in enumerate(final_chunks):
        chunk.id = f"chunk_{i + 1:03d}"

    return ChunkedRulebook(
        game_name=game_name,
        chunks=final_chunks,
        full_text=document.full_text,
        source_file=document.source_file,
    )


def _split_large_chunk(chunk: RuleChunk, max_size: int) -> list[RuleChunk]:
    """Split a large chunk into smaller pieces.

    Tries to split on paragraph boundaries.
    """
    content = chunk.content
    parts = []

    # Split on double newlines (paragraphs)
    paragraphs = re.split(r"\n\s*\n", content)

    current_part: list[str] = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        if current_size + para_size > max_size and current_part:
            # Save current part
            parts.append("\n\n".join(current_part))
            current_part = [para]
            current_size = para_size
        else:
            current_part.append(para)
            current_size += para_size

    if current_part:
        parts.append("\n\n".join(current_part))

    # Create chunk objects
    result = []
    for i, part in enumerate(parts):
        result.append(
            RuleChunk(
                id=f"{chunk.id}_{i + 1}",
                title=f"{chunk.title} (Part {i + 1})" if len(parts) > 1 else chunk.title,
                content=part,
                page_numbers=chunk.page_numbers,
                parent_section=chunk.parent_section,
                chunk_type="section_part",
                source_file=chunk.source_file,
            )
        )

    return result


def chunk_by_fixed_size(
    document: ExtractedDocument,
    game_name: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> ChunkedRulebook:
    """Chunk a document by fixed character size with overlap.

    Useful as a fallback when section detection doesn't work well.

    Args:
        document: Extracted document to chunk.
        game_name: Name of the game.
        chunk_size: Target size of each chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        ChunkedRulebook with fixed-size chunks.
    """
    text = document.full_text
    chunks = []

    start = 0
    chunk_num = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        if end < len(text):
            # Look for sentence end near the boundary
            search_start = max(end - 100, start)
            search_end = min(end + 100, len(text))
            search_region = text[search_start:search_end]

            # Find last sentence boundary
            for sep in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
                last_sep = search_region.rfind(sep)
                if last_sep != -1:
                    end = search_start + last_sep + len(sep)
                    break

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk_num += 1
            chunks.append(
                RuleChunk(
                    id=f"chunk_{chunk_num:03d}",
                    title=f"Section {chunk_num}",
                    content=chunk_text,
                    page_numbers=[],  # Can't easily determine for fixed-size
                    parent_section=None,
                    chunk_type="fixed_size",
                    source_file=document.source_file,
                )
            )

        start = end - overlap

    return ChunkedRulebook(
        game_name=game_name,
        chunks=chunks,
        full_text=text,
        source_file=document.source_file,
    )
