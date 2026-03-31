"""State recovery from photos - resync when physical and digital diverge."""

import json
from dataclasses import dataclass
from pathlib import Path

from ..engine.state_manager import StateManager, StateDiff, compute_diff
from ..llm.client import LLMClient


@dataclass
class RecoveryResult:
    """Result of state recovery."""

    proposed_state: dict
    diff_from_current: StateDiff
    confidence: float
    notes: list[str]
    applied: bool = False


RECOVERY_PROMPT = """You are recovering the current game state from a photo of the board.

GAME: {game_name}

STATE SCHEMA (what we track):
{schema}

CURRENT DIGITAL STATE (may be wrong):
{current_state}

RULES CONTEXT:
{rules_context}

Analyze this photo and determine the ACTUAL current state. Compare with the digital state and correct any discrepancies.

Focus on:
1. Resource counts for each player
2. Piece positions on the board
3. Cards/tokens in play
4. Round/phase indicators
5. Anything that looks different from the digital state

Return ONLY valid JSON:
{{
  "corrected_state": {{
    // Full state matching the schema, reflecting what you see in the photo
  }},
  "corrections": [
    "List of specific corrections made"
  ],
  "confidence": 0.85,
  "unclear": [
    "Things that are hard to see or ambiguous"
  ]
}}"""


class StateRecovery:
    """Recovers game state from photos."""

    def __init__(
        self,
        client: LLMClient,
        game_name: str,
        state_manager: StateManager,
        schema: dict | None = None,
        rules_index=None,
    ):
        """Initialize state recovery.

        Args:
            client: LLM client with vision capability.
            game_name: Name of the game.
            state_manager: State manager for the session.
            schema: State schema (helps guide analysis).
            rules_index: Optional rules index for context.
        """
        self.client = client
        self.game_name = game_name
        self.state_manager = state_manager
        self.schema = schema or {}
        self.rules_index = rules_index

    async def recover_from_photo(
        self,
        photo_path: Path,
        rules_context: str = "",
    ) -> RecoveryResult:
        """Recover state from a photo.

        Args:
            photo_path: Path to the board photo.
            rules_context: Optional rules text for context.

        Returns:
            RecoveryResult with proposed state.
        """
        # Read photo
        with open(photo_path, "rb") as f:
            image_bytes = f.read()

        # Get current state
        current_state = self.state_manager.get_state()

        # Get rules context if not provided
        if not rules_context and self.rules_index is not None:
            rules_context = self.rules_index.get_relevant_rules(
                phase=current_state.get("phase", "unknown"),
                n_results=3,
            )

        # Build prompt
        prompt = RECOVERY_PROMPT.format(
            game_name=self.game_name,
            schema=json.dumps(self.schema, indent=2) if self.schema else "{}",
            current_state=json.dumps(current_state, indent=2),
            rules_context=rules_context or "(No rules context)",
        )

        # Analyze photo
        response = await self.client.complete_with_images(
            system_prompt="You are recovering game state from a photo. Return only valid JSON.",
            user_prompt=prompt,
            images=[image_bytes],
            temperature=0.2,
        )

        try:
            data = response.as_json()
            proposed_state = data.get("corrected_state", current_state)
            corrections = data.get("corrections", [])
            confidence = data.get("confidence", 0.5)
            unclear = data.get("unclear", [])

            # Compute diff
            diff = compute_diff(current_state, proposed_state)

            return RecoveryResult(
                proposed_state=proposed_state,
                diff_from_current=diff,
                confidence=confidence,
                notes=corrections + [f"Unclear: {u}" for u in unclear],
            )

        except json.JSONDecodeError:
            return RecoveryResult(
                proposed_state=current_state,
                diff_from_current=StateDiff(),
                confidence=0.0,
                notes=["Failed to parse recovery response"],
            )

    async def recover_from_multiple_photos(
        self,
        photo_paths: list[Path],
        rules_context: str = "",
    ) -> RecoveryResult:
        """Recover state from multiple photos (different angles/areas).

        Args:
            photo_paths: Paths to board photos.
            rules_context: Optional rules text.

        Returns:
            Combined RecoveryResult.
        """
        if len(photo_paths) == 1:
            return await self.recover_from_photo(photo_paths[0], rules_context)

        # Read all photos
        images = []
        for path in photo_paths:
            with open(path, "rb") as f:
                images.append(f.read())

        current_state = self.state_manager.get_state()

        if not rules_context and self.rules_index is not None:
            rules_context = self.rules_index.get_relevant_rules(
                phase=current_state.get("phase", "unknown"),
                n_results=3,
            )

        prompt = f"""You are recovering game state from {len(images)} photos of the board.

GAME: {self.game_name}

STATE SCHEMA:
{json.dumps(self.schema, indent=2) if self.schema else "{}"}

CURRENT DIGITAL STATE:
{json.dumps(current_state, indent=2)}

RULES CONTEXT:
{rules_context or "(No rules context)"}

Analyze ALL photos together to determine the complete current state. Different photos may show different parts of the board.

Return ONLY valid JSON:
{{
  "corrected_state": {{ ... }},
  "corrections": ["list of corrections"],
  "confidence": 0.85,
  "unclear": ["things that are ambiguous"]
}}"""

        response = await self.client.complete_with_images(
            system_prompt="You are recovering game state from photos. Return only valid JSON.",
            user_prompt=prompt,
            images=images,
            temperature=0.2,
        )

        try:
            data = response.as_json()
            proposed_state = data.get("corrected_state", current_state)
            diff = compute_diff(current_state, proposed_state)

            return RecoveryResult(
                proposed_state=proposed_state,
                diff_from_current=diff,
                confidence=data.get("confidence", 0.5),
                notes=data.get("corrections", []) + [f"Unclear: {u}" for u in data.get("unclear", [])],
            )
        except json.JSONDecodeError:
            return RecoveryResult(
                proposed_state=current_state,
                diff_from_current=StateDiff(),
                confidence=0.0,
                notes=["Failed to parse recovery response"],
            )

    def apply_recovery(self, result: RecoveryResult) -> StateDiff:
        """Apply a recovery result to the state.

        Args:
            result: Recovery result to apply.

        Returns:
            StateDiff showing what changed.
        """
        diff = self.state_manager.save_state(result.proposed_state)
        result.applied = True
        return diff


async def recover_state_interactive(
    client: LLMClient,
    game_name: str,
    state_manager: StateManager,
    schema: dict | None = None,
    rules_index=None,
) -> bool:
    """Interactive state recovery from photo(s).

    Args:
        client: LLM client.
        game_name: Game name.
        state_manager: State manager.
        schema: State schema.
        rules_index: Rules index.

    Returns:
        True if recovery was applied, False otherwise.
    """
    recovery = StateRecovery(client, game_name, state_manager, schema, rules_index)

    print("\n=== State Recovery ===")
    print("Provide photo(s) of the current board state to resync.")
    print()

    # Get photo paths
    paths: list[Path] = []
    while True:
        prompt = "Photo path" if not paths else "Additional photo (or Enter when done)"
        photo_input = input(f"{prompt}: ").strip()

        if not photo_input:
            if not paths:
                print("No photos provided, recovery cancelled.")
                return False
            break

        path = Path(photo_input)
        if path.exists():
            paths.append(path)
            print(f"  Added: {path.name}")
        else:
            print(f"  File not found: {photo_input}")

    # Run recovery
    print(f"\nAnalyzing {len(paths)} photo(s)...")
    result = await recovery.recover_from_multiple_photos(paths)

    # Show results
    print(f"\n=== Recovery Result (confidence: {result.confidence:.0%}) ===")

    if result.diff_from_current.is_empty():
        print("No changes detected - state appears correct.")
        return False

    print("\nProposed changes:")
    print(result.diff_from_current.to_human_readable())

    if result.notes:
        print("\nNotes:")
        for note in result.notes[:5]:
            print(f"  - {note}")

    # Confirm
    response = input("\nApply these changes? (yes/no): ").strip().lower()

    if response in ("yes", "y"):
        recovery.apply_recovery(result)
        print("State updated.")
        return True
    else:
        print("Recovery cancelled.")
        return False
