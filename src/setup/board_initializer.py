"""Board initializer - creates initial state from board photos."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.client import LLMClient


@dataclass
class BoardAnalysis:
    """Analysis of a board photo."""

    elements_found: list[dict]  # List of identified elements
    uncertainties: list[str]  # Things the AI wasn't sure about
    suggested_state: dict  # Suggested state values


@dataclass
class InitializationResult:
    """Result of board initialization."""

    state: dict
    confidence: float
    clarifications_needed: list[str]
    analysis_notes: str


BOARD_ANALYSIS_PROMPT = """You are analyzing a board game setup photo to extract the initial game state.

GAME: {game_name}

STATE SCHEMA (what we need to track):
{schema}

RULES CONTEXT (for reference):
{rules_context}

Analyze this photo and identify:
1. Board regions and their current state
2. Player pieces and their positions
3. Resources visible on the board
4. Any cards, tokens, or markers visible
5. Score tracks or round markers

Return ONLY valid JSON:
{{
  "elements_found": [
    {{
      "type": "resource|piece|card|marker|area",
      "name": "element name",
      "location": "where it is",
      "count": 1,
      "player": "owner if applicable",
      "notes": "any relevant details"
    }}
  ],
  "uncertainties": [
    "List anything you're not sure about"
  ],
  "suggested_state": {{
    // Partial state matching the schema
  }},
  "analysis_notes": "Brief summary of what you see"
}}"""


STATE_ASSEMBLY_PROMPT = """Combine these board analyses into a complete initial game state.

GAME: {game_name}

STATE SCHEMA:
{schema}

ANALYSES FROM PHOTOS:
{analyses}

PLAYER NAMES: {players}

Create a complete initial state that:
1. Follows the schema structure exactly
2. Incorporates all findings from the photos
3. Uses reasonable defaults for anything not visible
4. Is internally consistent

Return ONLY valid JSON:
{{
  "state": {{
    // Complete state matching schema
  }},
  "confidence": 0.85,
  "assumptions": [
    "List assumptions made for missing information"
  ]
}}"""


class BoardInitializer:
    """Initializes game state from board photos."""

    def __init__(
        self,
        client: LLMClient,
        game_name: str,
        schema: dict | None = None,
    ):
        """Initialize the board initializer.

        Args:
            client: LLM client with vision capability.
            game_name: Name of the game.
            schema: State schema (optional, helps guide analysis).
        """
        self.client = client
        self.game_name = game_name
        self.schema = schema or {}

    async def analyze_photo(
        self,
        photo_path: Path,
        rules_context: str = "",
    ) -> BoardAnalysis:
        """Analyze a single board photo.

        Args:
            photo_path: Path to the photo.
            rules_context: Optional rules text for context.

        Returns:
            BoardAnalysis with identified elements.
        """
        # Read photo
        with open(photo_path, "rb") as f:
            image_bytes = f.read()

        prompt = BOARD_ANALYSIS_PROMPT.format(
            game_name=self.game_name,
            schema=json.dumps(self.schema, indent=2) if self.schema else "(No schema provided)",
            rules_context=rules_context[:2000] if rules_context else "(No rules provided)",
        )

        response = await self.client.complete_with_images(
            system_prompt="You are a precise board game analyst. Return only valid JSON.",
            user_prompt=prompt,
            images=[image_bytes],
            temperature=0.3,
        )

        try:
            data = response.as_json()
            return BoardAnalysis(
                elements_found=data.get("elements_found", []),
                uncertainties=data.get("uncertainties", []),
                suggested_state=data.get("suggested_state", {}),
            )
        except json.JSONDecodeError:
            return BoardAnalysis(
                elements_found=[],
                uncertainties=["Failed to parse analysis"],
                suggested_state={},
            )

    async def initialize_from_photos(
        self,
        photo_paths: list[Path],
        players: list[str],
        rules_context: str = "",
    ) -> InitializationResult:
        """Initialize state from multiple board photos.

        Args:
            photo_paths: Paths to board photos.
            players: List of player names.
            rules_context: Optional rules text.

        Returns:
            InitializationResult with the initial state.
        """
        # Analyze each photo
        analyses = []
        all_uncertainties = []

        for path in photo_paths:
            print(f"  Analyzing: {path.name}")
            analysis = await self.analyze_photo(path, rules_context)
            analyses.append(analysis)
            all_uncertainties.extend(analysis.uncertainties)

        # Combine analyses into complete state
        analyses_text = "\n\n".join(
            f"Photo {i+1}:\n{json.dumps(a.suggested_state, indent=2)}\nElements: {a.elements_found}"
            for i, a in enumerate(analyses)
        )

        prompt = STATE_ASSEMBLY_PROMPT.format(
            game_name=self.game_name,
            schema=json.dumps(self.schema, indent=2) if self.schema else "{}",
            analyses=analyses_text,
            players=", ".join(players),
        )

        response = await self.client.complete(
            system_prompt="You are assembling game state. Return only valid JSON.",
            user_prompt=prompt,
            temperature=0.3,
        )

        try:
            data = response.as_json()
            return InitializationResult(
                state=data.get("state", {}),
                confidence=data.get("confidence", 0.5),
                clarifications_needed=all_uncertainties + data.get("assumptions", []),
                analysis_notes=f"Analyzed {len(photo_paths)} photo(s)",
            )
        except json.JSONDecodeError:
            # Return minimal state from first analysis
            return InitializationResult(
                state=analyses[0].suggested_state if analyses else {},
                confidence=0.3,
                clarifications_needed=all_uncertainties + ["State assembly failed"],
                analysis_notes="Fallback to single photo analysis",
            )

    async def initialize_manual(
        self,
        players: list[str],
        rules_context: str = "",
    ) -> InitializationResult:
        """Initialize state without photos (manual setup).

        Creates a reasonable initial state based on rules.

        Args:
            players: List of player names.
            rules_context: Rules text for context.

        Returns:
            InitializationResult with default initial state.
        """
        prompt = f"""Create an initial game state for starting a new game.

GAME: {self.game_name}

STATE SCHEMA:
{json.dumps(self.schema, indent=2) if self.schema else "{}"}

RULES CONTEXT:
{rules_context[:4000] if rules_context else "(No rules provided)"}

PLAYERS: {", ".join(players)}

Return ONLY valid JSON:
{{
  "state": {{
    // Initial state for a new game
  }},
  "setup_notes": "Brief description of the starting setup"
}}"""

        response = await self.client.complete(
            system_prompt="You are setting up a board game. Return only valid JSON.",
            user_prompt=prompt,
            temperature=0.5,
        )

        try:
            data = response.as_json()
            return InitializationResult(
                state=data.get("state", {}),
                confidence=0.7,
                clarifications_needed=[],
                analysis_notes=data.get("setup_notes", ""),
            )
        except json.JSONDecodeError:
            # Return minimal default
            return InitializationResult(
                state={
                    "phase": "setup",
                    "round": 1,
                    "current_player": players[0] if players else "player1",
                    "players": {p: {} for p in players},
                    "board": {},
                },
                confidence=0.3,
                clarifications_needed=["Could not generate initial state from rules"],
                analysis_notes="Using minimal default state",
            )


async def initialize_board_interactive(
    client: LLMClient,
    game_name: str,
    players: list[str],
    schema: dict | None = None,
    rules_context: str = "",
) -> dict:
    """Interactive board initialization.

    Args:
        client: LLM client.
        game_name: Game name.
        players: Player names.
        schema: State schema.
        rules_context: Rules text.

    Returns:
        Confirmed initial state.
    """
    initializer = BoardInitializer(client, game_name, schema)

    print("\nBoard Setup:")
    print("  1. Provide photo path(s) for automatic setup")
    print("  2. Press Enter for manual/rules-based setup")

    photo_input = input("\nPhoto path (or Enter to skip): ").strip()

    if photo_input:
        # Handle single or multiple photos
        paths = []
        if Path(photo_input).exists():
            paths.append(Path(photo_input))

            # Ask for more photos
            while True:
                more = input("Additional photo (or Enter when done): ").strip()
                if not more:
                    break
                if Path(more).exists():
                    paths.append(Path(more))
                else:
                    print(f"File not found: {more}")

        if paths:
            print(f"\nAnalyzing {len(paths)} photo(s)...")
            result = await initializer.initialize_from_photos(
                paths, players, rules_context
            )
        else:
            print("No valid photos found, using manual setup...")
            result = await initializer.initialize_manual(players, rules_context)
    else:
        print("Generating initial state from rules...")
        result = await initializer.initialize_manual(players, rules_context)

    # Show result
    print(f"\n=== Initial State (confidence: {result.confidence:.0%}) ===")
    print(json.dumps(result.state, indent=2))

    if result.clarifications_needed:
        print("\nClarifications/assumptions:")
        for item in result.clarifications_needed[:5]:
            print(f"  - {item}")

    # Confirm or edit
    response = input("\nAccept this state? (yes/edit/no): ").strip().lower()

    if response in ("yes", "y", ""):
        return result.state
    elif response == "edit":
        print("Enter corrections as JSON (or 'done' when finished):")
        corrections = []
        while True:
            line = input()
            if line.strip().lower() == "done":
                break
            corrections.append(line)

        if corrections:
            try:
                edits = json.loads("\n".join(corrections))
                # Merge edits into state
                result.state.update(edits)
                print("State updated.")
            except json.JSONDecodeError:
                print("Invalid JSON, keeping original state.")

        return result.state
    else:
        # Return empty state for manual entry
        return {
            "phase": "setup",
            "round": 1,
            "current_player": players[0] if players else "player1",
            "players": {p: {} for p in players},
            "board": {},
        }
