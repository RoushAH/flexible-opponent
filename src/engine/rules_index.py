"""Rules index with vector embeddings for RAG retrieval."""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

from ..setup.chunker import ChunkedRulebook, RuleChunk


class RulesIndex:
    """Vector index for rulebook chunks using ChromaDB."""

    def __init__(self, index_dir: Path, game_name: str):
        """Initialize the rules index.

        Args:
            index_dir: Directory to store the index.
            game_name: Name of the game (used as collection name).
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.game_name = game_name

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.index_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Create or get collection
        # Using default embedding function (all-MiniLM-L6-v2)
        self.collection = self.client.get_or_create_collection(
            name=self._sanitize_collection_name(game_name),
            metadata={"game": game_name},
        )

        # Store chunks metadata separately for full retrieval
        self.chunks_file = self.index_dir / "chunks.json"
        self._chunks_cache: dict[str, RuleChunk] | None = None

    @staticmethod
    def _sanitize_collection_name(name: str) -> str:
        """Sanitize game name for use as collection name."""
        # ChromaDB collection names must be 3-63 chars, alphanumeric with underscores
        sanitized = "".join(c if c.isalnum() else "_" for c in name.lower())
        sanitized = sanitized.strip("_")
        if len(sanitized) < 3:
            sanitized = sanitized + "_rules"
        return sanitized[:63]

    def index_rulebook(self, rulebook: ChunkedRulebook) -> int:
        """Index a chunked rulebook.

        Args:
            rulebook: The chunked rulebook to index.

        Returns:
            Number of chunks indexed.
        """
        if not rulebook.chunks:
            return 0

        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in rulebook.chunks]
        documents = [chunk.content for chunk in rulebook.chunks]
        metadatas = [
            {
                "title": chunk.title,
                "chunk_type": chunk.chunk_type,
                "parent_section": chunk.parent_section or "",
                "page_numbers": ",".join(str(p) for p in chunk.page_numbers),
            }
            for chunk in rulebook.chunks
        ]

        # Add to collection (will embed automatically)
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        # Save full chunk data to JSON for later retrieval
        chunks_data = {chunk.id: chunk.to_dict() for chunk in rulebook.chunks}
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

        self._chunks_cache = {
            chunk_id: RuleChunk.from_dict(data) for chunk_id, data in chunks_data.items()
        }

        return len(rulebook.chunks)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_type: str | None = None,
    ) -> list[RuleChunk]:
        """Query the index for relevant rule chunks.

        Args:
            query_text: The query to search for.
            n_results: Maximum number of results to return.
            filter_type: Optional chunk_type filter.

        Returns:
            List of relevant RuleChunks, most relevant first.
        """
        # Build where filter if specified
        where = None
        if filter_type:
            where = {"chunk_type": filter_type}

        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Load full chunks from cache/file
        if self._chunks_cache is None:
            self._load_chunks_cache()

        # Build result list
        chunks = []
        if results["ids"] and results["ids"][0]:
            for chunk_id in results["ids"][0]:
                if chunk_id in self._chunks_cache:
                    chunks.append(self._chunks_cache[chunk_id])

        return chunks

    def query_by_section(self, section_title: str) -> RuleChunk | None:
        """Find a chunk by its section title.

        Args:
            section_title: Title to search for.

        Returns:
            Matching RuleChunk or None.
        """
        if self._chunks_cache is None:
            self._load_chunks_cache()

        # Exact match first
        for chunk in self._chunks_cache.values():
            if chunk.title.lower() == section_title.lower():
                return chunk

        # Partial match
        for chunk in self._chunks_cache.values():
            if section_title.lower() in chunk.title.lower():
                return chunk

        return None

    def get_all_chunks(self) -> list[RuleChunk]:
        """Get all indexed chunks.

        Returns:
            List of all RuleChunks.
        """
        if self._chunks_cache is None:
            self._load_chunks_cache()

        return list(self._chunks_cache.values())

    def get_relevant_rules(
        self,
        phase: str,
        action_type: str | None = None,
        context: str | None = None,
        n_results: int = 3,
    ) -> str:
        """Get relevant rules for a game situation.

        Convenience method that builds a query from game context.

        Args:
            phase: Current game phase.
            action_type: Type of action being considered.
            context: Additional context string.
            n_results: Number of chunks to retrieve.

        Returns:
            Concatenated relevant rules text.
        """
        # Build query from context
        query_parts = [f"During {phase} phase"]
        if action_type:
            query_parts.append(f"rules for {action_type}")
        if context:
            query_parts.append(context)

        query = " ".join(query_parts)
        chunks = self.query(query, n_results=n_results)

        if not chunks:
            return "(No relevant rules found)"

        # Format results
        sections = []
        for chunk in chunks:
            sections.append(f"## {chunk.title}\n{chunk.content}")

        return "\n\n".join(sections)

    def _load_chunks_cache(self) -> None:
        """Load chunks from JSON file into cache."""
        if self.chunks_file.exists():
            with open(self.chunks_file, encoding="utf-8") as f:
                data = json.load(f)
            self._chunks_cache = {
                chunk_id: RuleChunk.from_dict(chunk_data)
                for chunk_id, chunk_data in data.items()
            }
        else:
            self._chunks_cache = {}

    def clear(self) -> None:
        """Clear the index."""
        # Delete and recreate collection
        self.client.delete_collection(self._sanitize_collection_name(self.game_name))
        self.collection = self.client.create_collection(
            name=self._sanitize_collection_name(self.game_name),
            metadata={"game": self.game_name},
        )

        # Clear chunks file
        if self.chunks_file.exists():
            self.chunks_file.unlink()

        self._chunks_cache = {}

    @property
    def chunk_count(self) -> int:
        """Get the number of indexed chunks."""
        return self.collection.count()


async def create_rules_index(
    game_name: str,
    source: Path | str | list[Path],
    index_dir: Path,
    client=None,  # LLMClient, optional for images
) -> RulesIndex:
    """Create a rules index from a rulebook source.

    Convenience function that handles extraction, chunking, and indexing.

    Args:
        game_name: Name of the game.
        source: Path to rulebook (PDF, text, images) or raw text.
        index_dir: Directory to store the index.
        client: LLM client (required for image extraction).

    Returns:
        Populated RulesIndex.
    """
    from ..setup.chunker import chunk_by_sections
    from ..setup.text_extractor import extract_document

    # Extract text
    document = await extract_document(source, client)

    # Chunk the document
    rulebook = chunk_by_sections(document, game_name)

    # Create and populate index
    index = RulesIndex(index_dir, game_name)
    index.index_rulebook(rulebook)

    return index
