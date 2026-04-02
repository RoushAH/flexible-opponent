"""Move interpreter - extracts state changes from human move descriptions."""

import json
import re
from dataclasses import dataclass

from .client import LLMClient


@dataclass
class InterpretedMove:
    """Result of interpreting a human move."""

    action_id: str
    state_updates: dict
    interpretation: str  # What the LLM understood
    confidence: str  # high, medium, low


INTERPRET_PROMPT = """You interpret board game moves and extract state changes.

GAME: {game_name}

CURRENT STATE:
{current_state}

PLAYER: {player}

MOVE DESCRIPTION:
{move_description}

Interpret this move and extract the state changes. Consider:
- Resources gained or spent
- Board positions changed (workers placed, cards played)
- Accumulation spaces that were emptied or partially taken

RESPOND WITH ONLY THIS JSON:
{{
  "action_id": "short_action_name",
  "state_updates": {{
    "players.{player}.resources.wood": 2,
    "board.forest.wood": -2
  }},
  "interpretation": "What the move means",
  "confidence": "high|medium|low"
}}

Use dot notation for nested paths (players.human.resources.wood).
Use positive numbers for gains, negative for losses/costs.
For board spaces that accumulate, set to new value OR use negative to subtract."""


class MoveInterpreter:
    """Interprets human moves and extracts state changes."""

    def __init__(self, client: LLMClient, game_name: str):
        """Initialize the interpreter.

        Args:
            client: LLM client.
            game_name: Name of the game.
        """
        self.client = client
        self.game_name = game_name
        self.rules_index = None

    def set_rules_index(self, rules_index) -> None:
        """Set the rules index for context.

        Args:
            rules_index: RulesIndex instance.
        """
        self.rules_index = rules_index

    async def interpret_move(
        self,
        player: str,
        move_description: str,
        current_state: dict,
    ) -> InterpretedMove:
        """Interpret a human move description into state changes.

        Args:
            player: Name of the player who made the move.
            move_description: Natural language description of the move.
            current_state: Current game state.

        Returns:
            InterpretedMove with extracted state updates.
        """
        # Get relevant rules if available
        rules_context = ""
        if self.rules_index:
            chunks = self.rules_index.query(move_description, n_results=2)
            if chunks:
                rules_context = "\n".join(c.content[:300] for c in chunks)

        prompt = INTERPRET_PROMPT.format(
            game_name=self.game_name,
            current_state=json.dumps(current_state, indent=2)[:2000],
            player=player,
            move_description=move_description,
        )

        if rules_context:
            prompt += f"\n\nRELEVANT RULES:\n{rules_context}"

        response = await self.client.complete(
            system_prompt="You extract state changes from move descriptions. Respond with ONLY valid JSON.",
            user_prompt=prompt,
            temperature=0.3,  # Low temperature for consistency
        )

        # Parse response
        try:
            data = response.as_json()
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    data = None

        if data:
            # Convert dot notation to nested dict
            state_updates = self._expand_dot_notation(data.get("state_updates", {}))

            return InterpretedMove(
                action_id=data.get("action_id", "human_move"),
                state_updates=state_updates,
                interpretation=data.get("interpretation", move_description),
                confidence=data.get("confidence", "medium"),
            )
        else:
            # Fallback - no state updates
            return InterpretedMove(
                action_id="human_move",
                state_updates={},
                interpretation=f"Could not interpret: {move_description}",
                confidence="low",
            )

    def _expand_dot_notation(self, flat_updates: dict) -> dict:
        """Convert dot-notation keys to nested dict.

        Example: {"players.human.wood": 5} -> {"players": {"human": {"wood": 5}}}
        """
        result = {}

        for key, value in flat_updates.items():
            parts = key.split(".")
            current = result

            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Handle numeric values (could be negative for "subtract")
            final_key = parts[-1]
            current[final_key] = value

        return result

    def apply_delta_updates(self, state: dict, updates: dict) -> dict:
        """Apply updates to state, treating negative values as deltas.

        Args:
            state: Current state dict.
            updates: Updates to apply (nested dict).

        Returns:
            New state with updates applied.
        """
        import copy
        new_state = copy.deepcopy(state)

        def apply_recursive(target: dict, source: dict, path: str = ""):
            for key, value in source.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    if key not in target:
                        target[key] = {}
                    if isinstance(target[key], dict):
                        apply_recursive(target[key], value, current_path)
                    else:
                        target[key] = value
                elif isinstance(value, (int, float)):
                    # For numbers, apply as delta if key exists
                    if key in target and isinstance(target[key], (int, float)):
                        target[key] = target[key] + value
                    else:
                        target[key] = value
                else:
                    target[key] = value

        apply_recursive(new_state, updates)
        return new_state
