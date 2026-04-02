"""AI Strategist - picks the best move and manages strategy."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import LLMClient, Role
from .rules_interpreter import Action, LegalActionsResult


@dataclass
class StrategyUpdate:
    """Proposed updates to the live strategy."""

    turn_focus: str | None = None
    blocked_goals: list[dict] = field(default_factory=list)
    priority_changes: list[str] = field(default_factory=list)


@dataclass
class MoveDecision:
    """The strategist's decision."""

    chosen_action: Action
    reasoning: str
    strategy_status: str  # continue, delay, adapt, pivot
    strategy_update: StrategyUpdate
    concerns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "chosen_action": self.chosen_action.to_dict(),
            "reasoning": self.reasoning,
            "strategy_status": self.strategy_status,
            "strategy_update": {
                "turn_focus": self.strategy_update.turn_focus,
                "blocked_goals": self.strategy_update.blocked_goals,
                "priority_changes": self.strategy_update.priority_changes,
            },
            "concerns": self.concerns,
        }


# Load the prompt template
_PROMPT_TEMPLATE: str | None = None


def _get_prompt_template() -> str:
    """Load the prompt template lazily."""
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        prompt_file = Path(__file__).parent / "prompts" / "strategist.txt"
        with open(prompt_file, encoding="utf-8") as f:
            _PROMPT_TEMPLATE = f.read()
    return _PROMPT_TEMPLATE


class Strategist:
    """AI strategist that chooses moves and manages strategy."""

    def __init__(self, client: LLMClient, game_name: str):
        """Initialize the strategist.

        Args:
            client: The LLM client to use.
            game_name: Name of the game being played.
        """
        self.client = client
        self.game_name = game_name

    async def choose_move(
        self,
        legal_actions: LegalActionsResult,
        visible_state: dict,
        ai_hidden: dict,
        strategy_core: dict,
        strategy_live: dict,
    ) -> MoveDecision:
        """Choose the best move from legal actions.

        Args:
            legal_actions: Available legal actions from the rules interpreter.
            visible_state: Current visible game state.
            ai_hidden: AI's hidden information.
            strategy_core: Stable long-term strategy.
            strategy_live: Current tactical strategy.

        Returns:
            MoveDecision with the chosen action and reasoning.

        Raises:
            ValueError: If no valid action can be chosen.
        """
        if not legal_actions.actions:
            raise ValueError("No legal actions available")

        # Format the prompt
        prompt = _get_prompt_template().format(
            game_name=self.game_name,
            strategy_core=json.dumps(strategy_core, indent=2),
            strategy_live=json.dumps(strategy_live, indent=2),
            visible_state=json.dumps(visible_state, indent=2),
            ai_hidden=json.dumps(ai_hidden, indent=2),
            legal_actions=legal_actions.format_for_strategist(),
        )

        # Get LLM response
        response = await self.client.complete(
            system_prompt="You are a strategic game player. Return only valid JSON.",
            user_prompt=prompt,
            role=Role.STRATEGIST,
            temperature=0.7,  # Slightly higher for creative play
        )

        # Parse response
        data = None
        try:
            data = response.as_json()
        except json.JSONDecodeError:
            # Try to extract JSON from response if LLM added text around it
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        if data is None:
            # Fallback: pick the first high-confidence action
            return self._fallback_decision(legal_actions, response.content[:200])

        # Find the chosen action
        action_id = data.get("chosen_action_id", "")
        chosen_action = legal_actions.get_action_by_id(action_id)

        if chosen_action is None:
            # Action ID not found, try to match by description or fall back
            return self._fallback_decision(
                legal_actions, f"action_id '{action_id}' not found in legal actions"
            )

        # Parse strategy update
        update_data = data.get("strategy_update", {})
        strategy_update = StrategyUpdate(
            turn_focus=update_data.get("turn_focus"),
            blocked_goals=update_data.get("blocked_goals", []),
            priority_changes=update_data.get("priority_changes", []),
        )

        return MoveDecision(
            chosen_action=chosen_action,
            reasoning=data.get("reasoning", ""),
            strategy_status=data.get("strategy_status", "continue"),
            strategy_update=strategy_update,
            concerns=data.get("concerns", []),
        )

    def _fallback_decision(
        self, legal_actions: LegalActionsResult, error_preview: str = ""
    ) -> MoveDecision:
        """Make a fallback decision when LLM fails.

        Picks the first high-confidence action, or just the first action.
        """
        error_note = f" (response: {error_preview}...)" if error_preview else ""

        # Try to find a high-confidence action
        for action in legal_actions.actions:
            if action.confidence == "high":
                return MoveDecision(
                    chosen_action=action,
                    reasoning=f"Fallback: chose first high-confidence action due to parsing error{error_note}",
                    strategy_status="continue",
                    strategy_update=StrategyUpdate(),
                    concerns=["LLM response could not be parsed"],
                )

        # Just pick the first action
        return MoveDecision(
            chosen_action=legal_actions.actions[0],
            reasoning=f"Fallback: chose first available action due to parsing error{error_note}",
            strategy_status="continue",
            strategy_update=StrategyUpdate(),
            concerns=["LLM response could not be parsed"],
        )

    async def initialize_strategy(
        self,
        rules_summary: str,
        initial_state: dict,
        ai_hidden: dict,
    ) -> tuple[dict, dict]:
        """Initialize strategy at the start of a game.

        Args:
            rules_summary: Summary of game rules and win conditions.
            initial_state: Initial game state.
            ai_hidden: AI's starting hidden information.

        Returns:
            Tuple of (strategy_core, strategy_live) dictionaries.
        """
        prompt = f"""You are starting a new game of {self.game_name}.

RULES SUMMARY:
{rules_summary}

INITIAL STATE:
{json.dumps(initial_state, indent=2)}

YOUR STARTING HIDDEN INFORMATION:
{json.dumps(ai_hidden, indent=2)}

Based on the rules, your starting position, and your hidden information, create an initial strategy.

Return ONLY valid JSON:
{{
  "strategy_core": {{
    "long_term_plan": "brief description of your overall approach",
    "plan_confidence": 0.7,
    "pivot_conditions": ["condition that would make you change plans"]
  }},
  "strategy_live": {{
    "current_priorities": ["priority1", "priority2", "priority3"],
    "blocked_goals": [],
    "turn_focus": "what to focus on in early game",
    "strategy_status": "continue",
    "fallbacks": {{}}
  }}
}}"""

        response = await self.client.complete(
            system_prompt="You are a strategic game player. Return only valid JSON.",
            user_prompt=prompt,
            role=Role.STRATEGIST,
            temperature=0.7,
        )

        try:
            data = response.as_json()
            return data.get("strategy_core", {}), data.get("strategy_live", {})
        except json.JSONDecodeError:
            # Return default strategy
            return (
                {
                    "long_term_plan": "play adaptively based on opportunities",
                    "plan_confidence": 0.5,
                    "pivot_conditions": [],
                },
                {
                    "current_priorities": [],
                    "blocked_goals": [],
                    "turn_focus": "assess the board and find opportunities",
                    "strategy_status": "continue",
                    "fallbacks": {},
                },
            )
