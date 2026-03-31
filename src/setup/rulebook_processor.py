"""Main rulebook processor that orchestrates extraction, chunking, and indexing."""

from dataclasses import dataclass
from pathlib import Path

from ..engine.rules_index import RulesIndex
from ..llm.client import LLMClient
from .chunker import ChunkedRulebook, chunk_by_fixed_size, chunk_by_sections
from .text_extractor import ExtractedDocument, extract_document


@dataclass
class ProcessingResult:
    """Result of rulebook processing."""

    game_name: str
    index: RulesIndex
    chunk_count: int
    page_count: int
    source_type: str
    source_files: list[str]

    def summary(self) -> str:
        """Get a human-readable summary."""
        return (
            f"Processed rulebook for '{self.game_name}':\n"
            f"  - Source: {self.source_type} ({len(self.source_files)} file(s))\n"
            f"  - Pages: {self.page_count}\n"
            f"  - Chunks indexed: {self.chunk_count}"
        )


class RulebookProcessor:
    """Processes rulebooks into searchable indexes."""

    def __init__(
        self,
        game_name: str,
        session_dir: Path,
        client: LLMClient | None = None,
    ):
        """Initialize the processor.

        Args:
            game_name: Name of the game.
            session_dir: Directory for the game session.
            client: LLM client (required for image processing).
        """
        self.game_name = game_name
        self.session_dir = Path(session_dir)
        self.client = client

        # Rules index directory
        self.index_dir = self.session_dir / "rules_index"

    async def process(
        self,
        source: Path | str | list[Path],
        chunking_strategy: str = "sections",
    ) -> ProcessingResult:
        """Process a rulebook source into a searchable index.

        Args:
            source: Path to rulebook file(s), or raw text string.
            chunking_strategy: "sections" (detect headers) or "fixed" (fixed size).

        Returns:
            ProcessingResult with index and metadata.

        Raises:
            ValueError: If source type requires client but none provided.
        """
        # Determine source type
        source_type, source_files = self._analyze_source(source)

        # Check if we need a client
        if source_type == "images" and self.client is None:
            raise ValueError("LLM client required for image processing")

        # Extract text
        print(f"Extracting text from {source_type}...")
        document = await extract_document(source, self.client)
        print(f"  Extracted {document.total_pages} page(s)")

        # Chunk the document
        print(f"Chunking with '{chunking_strategy}' strategy...")
        if chunking_strategy == "sections":
            rulebook = chunk_by_sections(document, self.game_name)
        else:
            rulebook = chunk_by_fixed_size(document, self.game_name)
        print(f"  Created {len(rulebook.chunks)} chunks")

        # Index the chunks
        print("Indexing chunks with vector embeddings...")
        index = RulesIndex(self.index_dir, self.game_name)
        chunk_count = index.index_rulebook(rulebook)
        print(f"  Indexed {chunk_count} chunks")

        return ProcessingResult(
            game_name=self.game_name,
            index=index,
            chunk_count=chunk_count,
            page_count=document.total_pages,
            source_type=source_type,
            source_files=source_files,
        )

    def load_existing_index(self) -> RulesIndex | None:
        """Load an existing rules index if available.

        Returns:
            RulesIndex if exists, None otherwise.
        """
        if not self.index_dir.exists():
            return None

        chunks_file = self.index_dir / "chunks.json"
        if not chunks_file.exists():
            return None

        index = RulesIndex(self.index_dir, self.game_name)
        if index.chunk_count > 0:
            return index

        return None

    @staticmethod
    def _analyze_source(source: Path | str | list[Path]) -> tuple[str, list[str]]:
        """Analyze the source type.

        Returns:
            Tuple of (source_type, list of source file paths).
        """
        # Raw text
        if isinstance(source, str) and not Path(source).exists():
            return "text", ["<raw_text>"]

        # List of files (images)
        if isinstance(source, list):
            return "images", [str(p) for p in source]

        # Single file
        path = Path(source)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return "pdf", [str(path)]
        elif suffix in (".txt", ".md"):
            return "text", [str(path)]
        elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            return "images", [str(path)]
        else:
            return "unknown", [str(path)]


async def process_rulebook(
    game_name: str,
    source: Path | str | list[Path],
    session_dir: Path,
    client: LLMClient | None = None,
) -> ProcessingResult:
    """Convenience function to process a rulebook.

    Args:
        game_name: Name of the game.
        source: Path to rulebook or raw text.
        session_dir: Game session directory.
        client: LLM client (required for images).

    Returns:
        ProcessingResult with the created index.
    """
    processor = RulebookProcessor(game_name, session_dir, client)
    return await processor.process(source)
