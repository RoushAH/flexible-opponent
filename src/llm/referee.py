"""Referee - validates move legality with full game knowledge."""

import json
from dataclasses import dataclass
from pathlib import Path

from .client import LLMClient, Role


@dataclass
class ValidationResult:
    """Result of move validation."""

    valid: bool
    reason: str
    rule_citation: str | None = None
    suggested_fix: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "rule_citation": self.rule_citation,
            "suggested_fix": self.suggested_fix,
        }


# Load the prompt template
_PROMPT_TEMPLATE: str | None = None


def _get_prompt_template() -> str:
    """Load the prompt template lazily."""
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        prompt_file = Path(__file__).parent / "prompts" / "referee.txt"
        with open(prompt_file, encoding="utf-8") as f:
            _PROMPT_TEMPLATE = f.read()
    return _PROMPT_TEMPLATE


class Referee:
    """Referee with full game knowledge for move validation."""

    def __init__(
        self,
        client: LLMClient,
        game_name: str,
        rules_index=None,
    ):
        """Initialize the referee.

        Args:
            client: LLM client.
            game_name: Name of the game.
            rules_index: Optional RulesIndex for rule lookups.
        """
        self.client = client
        self.game_name = game_name
        self.rules_index = rules_index

        # Referee maintains knowledge of all hidden state
        self._human_hidden: dict = {}

    def set_human_hidden(self, hidden: dict) -> None:
        """Set the human player's hidden information.

        This is called during setup when the human reveals their hand to the referee.

        Args:
            hidden: Human's hidden state (hand, bonuses, etc.)
        """
        self._human_hidden = hidden

    def update_human_hidden(self, updates: dict) -> None:
        """Update specific fields in human's hidden state.

        Args:
            updates: Fields to update.
        """
        self._human_hidden.update(updates)

    async def validate_move(
        self,
        player: str,
        action_id: str,
        action_description: str,
        visible_state: dict,
        ai_hidden: dict,
        rules_text: str | None = None,
    ) -> ValidationResult:
        """Validate whether a move is legal.

        Args:
            player: Player making the move ("ai" or human name).
            action_id: ID of the action.
            action_description: Human-readable action description.
            visible_state: Current visible game state.
            ai_hidden: AI's hidden information.
            rules_text: Relevant rules (if None, uses rules_index).

        Returns:
            ValidationResult indicating if move is legal.
        """
        # Get relevant rules if not provided
        if rules_text is None and self.rules_index is not None:
            rules_text = self.rules_index.get_relevant_rules(
                phase=visible_state.get("phase", "unknown"),
                action_type=action_id,
                n_results=3,
            )
        elif rules_text is None:
            rules_text = "(No rules available for validation)"

        # Build prompt
        prompt = _get_prompt_template().format(
            game_name=self.game_name,
            visible_state=json.dumps(visible_state, indent=2),
            ai_hidden=json.dumps(ai_hidden, indent=2),
            human_hidden=json.dumps(self._human_hidden, indent=2),
            rules_text=rules_text,
            player=player,
            action_id=action_id,
            action_description=action_description,
        )

        response = await self.client.complete(
            system_prompt="You are a strict but fair game referee. Return only valid JSON.",
            user_prompt=prompt,
            role=Role.REFEREE,
            temperature=0.2,  # Low temperature for consistent rulings
        )

        try:
            data = response.as_json()
            return ValidationResult(
                valid=data.get("valid", False),
                reason=data.get("reason", "Unknown"),
                rule_citation=data.get("rule_citation"),
                suggested_fix=data.get("suggested_fix"),
            )
        except json.JSONDecodeError:
            # If we can't parse, assume valid (fail open for playability)
            return ValidationResult(
                valid=True,
                reason="Validation inconclusive - allowing move",
                rule_citation=None,
                suggested_fix=None,
            )

    async def validate_state_consistency(
        self,
        visible_state: dict,
        ai_hidden: dict,
    ) -> list[str]:
        """Check if the game state is internally consistent.

        Args:
            visible_state: Current visible state.
            ai_hidden: AI's hidden state.

        Returns:
            List of inconsistencies found (empty if consistent).
        """
        prompt = f"""Check this game state for internal consistency.

GAME: {self.game_name}

VISIBLE STATE:
{json.dumps(visible_state, indent=2)}

AI HIDDEN STATE:
{json.dumps(ai_hidden, indent=2)}

HUMAN HIDDEN STATE:
{json.dumps(self._human_hidden, indent=2)}

Look for:
1. Resource totals that don't add up
2. Pieces in invalid locations
3. Cards that shouldn't exist (duplicates, wrong deck)
4. Phase/turn inconsistencies
5. Any impossible game states

Return JSON:
{{
  "consistent": true|false,
  "issues": ["list of issues found"]
}}"""

        response = await self.client.complete(
            system_prompt="You are checking game state consistency. Return only valid JSON.",
            user_prompt=prompt,
            role=Role.REFEREE,
            temperature=0.2,
        )

        try:
            data = response.as_json()
            return data.get("issues", [])
        except json.JSONDecodeError:
            return []

    async def resolve_ambiguity(
        self,
        situation: str,
        rules_text: str | None = None,
    ) -> str:
        """Get a referee ruling on an ambiguous situation.

        Args:
            situation: Description of the ambiguous situation.
            rules_text: Relevant rules.

        Returns:
            Referee's ruling/interpretation.
        """
        if rules_text is None and self.rules_index is not None:
            rules_text = self.rules_index.get_relevant_rules(
                phase="unknown",
                context=situation,
                n_results=5,
            )

        prompt = f"""A rules question has come up during the game.

GAME: {self.game_name}

SITUATION:
{situation}

RELEVANT RULES:
{rules_text or "(No specific rules found)"}

Provide a fair ruling. If the rules are genuinely ambiguous, say so and suggest the most reasonable interpretation. Keep your ruling brief and clear."""

        response = await self.client.complete(
            system_prompt="You are a fair game referee making a ruling.",
            user_prompt=prompt,
            role=Role.REFEREE,
            temperature=0.3,
        )

        return response.content.strip()
