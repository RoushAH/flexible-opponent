"""Rules Interpreter - lists plausible legal actions."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import LLMClient, Role


@dataclass
class Action:
    """A plausible legal action."""

    id: str
    description: str
    cost: dict[str, Any] = field(default_factory=dict)
    gains: dict[str, Any] = field(default_factory=dict)
    effects: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    confidence: str = "medium"  # high, medium, low
    rules_basis: str = ""
    ambiguity: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        """Create an Action from a dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            cost=data.get("cost", {}),
            gains=data.get("gains", {}),
            effects=data.get("effects", []),
            prerequisites=data.get("prerequisites", []),
            confidence=data.get("confidence", "medium"),
            rules_basis=data.get("rules_basis", ""),
            ambiguity=data.get("ambiguity"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "cost": self.cost,
            "gains": self.gains,
            "effects": self.effects,
            "prerequisites": self.prerequisites,
            "confidence": self.confidence,
            "rules_basis": self.rules_basis,
            "ambiguity": self.ambiguity,
        }


@dataclass
class LegalActionsResult:
    """Result of enumerating legal actions."""

    actions: list[Action]
    phase_note: str
    ambiguities: list[str]

    def get_action_by_id(self, action_id: str) -> Action | None:
        """Find an action by its ID."""
        for action in self.actions:
            if action.id == action_id:
                return action
        return None

    def format_for_strategist(self) -> str:
        """Format actions as a numbered list for the strategist."""
        lines = [f"Phase: {self.phase_note}", "", "Available Actions:"]

        for i, action in enumerate(self.actions, 1):
            conf_marker = {"high": "", "medium": "(?)", "low": "(?)"}[action.confidence]
            lines.append(f"{i}. [{action.id}] {action.description} {conf_marker}")

            if action.cost:
                cost_str = ", ".join(f"{k}: {v}" for k, v in action.cost.items())
                lines.append(f"   Cost: {cost_str}")

            if action.gains:
                gains_str = ", ".join(f"{k}: {v}" for k, v in action.gains.items())
                lines.append(f"   Gains: {gains_str}")

        if self.ambiguities:
            lines.extend(["", "Rule Ambiguities:"])
            for amb in self.ambiguities:
                lines.append(f"  - {amb}")

        return "\n".join(lines)


# Load the prompt template
_PROMPT_TEMPLATE: str | None = None


def _get_prompt_template() -> str:
    """Load the prompt template lazily."""
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        prompt_file = Path(__file__).parent / "prompts" / "rules_interpreter.txt"
        with open(prompt_file, encoding="utf-8") as f:
            _PROMPT_TEMPLATE = f.read()
    return _PROMPT_TEMPLATE


class RulesInterpreter:
    """Interprets game rules to enumerate legal actions."""

    def __init__(self, client: LLMClient, game_name: str, rules_index=None):
        """Initialize the rules interpreter.

        Args:
            client: The LLM client to use.
            game_name: Name of the game being played.
            rules_index: Optional RulesIndex for RAG retrieval.
        """
        self.client = client
        self.game_name = game_name
        self.rules_index = rules_index  # RulesIndex instance

    def set_rules_index(self, rules_index) -> None:
        """Set the rules index for RAG retrieval.

        Args:
            rules_index: RulesIndex instance.
        """
        self.rules_index = rules_index

    async def enumerate_actions(
        self,
        rules_text: str | None,
        visible_state: dict,
        ai_hidden: dict,
        phase: str = "unknown",
        recent_moves: list[dict] | None = None,
    ) -> LegalActionsResult:
        """Enumerate plausible legal actions for the AI player.

        Args:
            rules_text: Relevant rules excerpt. If None and rules_index is set,
                       will use RAG to retrieve relevant rules.
            visible_state: Current visible game state.
            ai_hidden: AI's hidden information (hand, bonuses, etc.).
            phase: Current game phase/stage.
            recent_moves: Recent move history for context.

        Returns:
            LegalActionsResult containing the enumerated actions.
        """
        # Get rules text via RAG if not provided
        if rules_text is None and self.rules_index is not None:
            # Build context for RAG query
            context_parts = [f"phase: {phase}"]
            if ai_hidden:
                # Include card names from hand for relevant rules
                hand = ai_hidden.get("hand", ai_hidden.get("cards", []))
                if hand:
                    context_parts.append(f"cards in hand: {hand}")
            context = " ".join(context_parts)

            rules_text = self.rules_index.get_relevant_rules(
                phase=phase,
                context=context,
                n_results=5,
            )
        elif rules_text is None:
            rules_text = "(No rules provided)"

        # Format the prompt
        prompt = _get_prompt_template().format(
            game_name=self.game_name,
            phase=phase,
            rules_text=rules_text,
            visible_state=json.dumps(visible_state, indent=2),
            ai_hidden=json.dumps(ai_hidden, indent=2),
            recent_moves=self._format_recent_moves(recent_moves),
        )

        # Get LLM response
        response = await self.client.complete(
            system_prompt="You are a precise rules interpreter. Return only valid JSON.",
            user_prompt=prompt,
            role=Role.RULES_INTERPRETER,
            temperature=0.3,  # Lower temperature for more consistent rule interpretation
        )

        # Parse response
        try:
            data = response.as_json()
        except json.JSONDecodeError as e:
            # If parsing fails, return empty result with error
            return LegalActionsResult(
                actions=[],
                phase_note=f"Error parsing response: {e}",
                ambiguities=[f"LLM response was not valid JSON: {response.content[:200]}"],
            )

        actions = [Action.from_dict(a) for a in data.get("actions", [])]

        return LegalActionsResult(
            actions=actions,
            phase_note=data.get("phase_note", ""),
            ambiguities=data.get("ambiguities", []),
        )

    @staticmethod
    def _format_recent_moves(moves: list[dict] | None) -> str:
        """Format recent moves for the prompt."""
        if not moves:
            return "(No previous moves)"

        lines = []
        for move in moves[-5:]:  # Last 5 moves
            player = move.get("player", "?")
            action = move.get("action_id", "?")
            desc = move.get("description", "")
            lines.append(f"- {player}: {action} - {desc}")

        return "\n".join(lines)
