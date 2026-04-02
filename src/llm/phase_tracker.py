"""Phase tracker - detects and handles round/phase transitions."""

import json
import re
from dataclasses import dataclass, field

from .client import LLMClient


@dataclass
class PhaseChange:
    """Detected phase or round change."""

    change_type: str  # "round_end", "phase_end", "game_end"
    description: str
    expected_updates: dict  # State changes that should happen
    confirmation_prompt: str  # Question to ask user
    auto_apply: bool = False  # Can we auto-apply these changes?


DETECT_PHASE_PROMPT = """You detect phase and round changes in board games.

GAME: {game_name}

RULES ABOUT ROUNDS/PHASES:
{rules_text}

CURRENT STATE:
{current_state}

MOVE JUST PLAYED:
Player: {player}
Action: {action}

RECENT MOVES THIS ROUND:
{recent_moves}

Analyze whether this move triggers any phase/round transitions:
1. Does this move end the current round? (all workers placed, all players passed, etc.)
2. Does this trigger a phase change? (harvest, feeding, cleanup, etc.)
3. What effects should happen? (accumulation, worker return, resource gain/loss)

RESPOND WITH ONLY THIS JSON:
{{
  "triggers_change": true|false,
  "change_type": "round_end|phase_end|none",
  "phase_name": "name of phase ending or starting",
  "effects": [
    {{"description": "Add 3 wood to Forest", "path": "board.forest.wood", "delta": 3}},
    {{"description": "Return all workers to supply", "path": "board.action_spaces", "value": {{}}}}
  ],
  "expected_state": {{
    "board.forest.wood": 4,
    "round": 2
  }},
  "confirmation_needed": "Just to confirm: Round 1 ended. The Forest should now have 4 wood, Clay Pit should have 2 clay. Is that correct?"
}}

If no phase change, return:
{{"triggers_change": false}}"""


class PhaseTracker:
    """Tracks and handles phase/round transitions."""

    def __init__(self, client: LLMClient, game_name: str):
        """Initialize the phase tracker.

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

    async def check_phase_change(
        self,
        player: str,
        action: str,
        current_state: dict,
        recent_moves: list[dict],
    ) -> PhaseChange | None:
        """Check if a move triggers a phase/round change.

        Args:
            player: Player who just moved.
            action: Description of the action taken.
            current_state: Current game state.
            recent_moves: Recent moves for context.

        Returns:
            PhaseChange if a transition occurred, None otherwise.
        """
        # Get relevant rules about phases/rounds
        rules_text = ""
        if self.rules_index:
            chunks = self.rules_index.query(
                "round end phase cleanup accumulation harvest feeding",
                n_results=3,
            )
            if chunks:
                rules_text = "\n\n".join(c.content[:500] for c in chunks)

        if not rules_text:
            rules_text = "(No specific phase rules found)"

        # Format recent moves
        moves_text = "\n".join(
            f"- {m.get('player', '?')}: {m.get('description', m.get('action_id', '?'))}"
            for m in recent_moves[-10:]
        ) if recent_moves else "(none)"

        prompt = DETECT_PHASE_PROMPT.format(
            game_name=self.game_name,
            rules_text=rules_text,
            current_state=json.dumps(current_state, indent=2)[:2000],
            player=player,
            action=action,
            recent_moves=moves_text,
        )

        response = await self.client.complete(
            system_prompt="You detect phase transitions in board games. Respond with ONLY valid JSON.",
            user_prompt=prompt,
            temperature=0.3,
        )

        # Parse response
        data = None
        try:
            data = response.as_json()
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        if data is None or not data.get("triggers_change", False):
            return None

        # Build expected updates from effects
        expected_updates = {}
        for effect in data.get("effects", []):
            if "path" in effect:
                if "delta" in effect:
                    expected_updates[effect["path"]] = {"delta": effect["delta"]}
                elif "value" in effect:
                    expected_updates[effect["path"]] = {"value": effect["value"]}

        # Also include explicit expected_state
        for path, value in data.get("expected_state", {}).items():
            if path not in expected_updates:
                expected_updates[path] = {"expected": value}

        return PhaseChange(
            change_type=data.get("change_type", "round_end"),
            description=data.get("phase_name", "Phase transition"),
            expected_updates=expected_updates,
            confirmation_prompt=data.get(
                "confirmation_needed",
                "Did the round/phase change? Please confirm the board state."
            ),
        )

    def apply_phase_effects(self, state: dict, phase_change: PhaseChange) -> dict:
        """Apply phase change effects to state.

        Args:
            state: Current state.
            phase_change: Phase change with expected updates.

        Returns:
            Updated state.
        """
        import copy
        new_state = copy.deepcopy(state)

        for path, update in phase_change.expected_updates.items():
            parts = path.split(".")
            target = new_state

            # Navigate to parent
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]

            final_key = parts[-1]

            if "delta" in update:
                # Apply delta
                current = target.get(final_key, 0)
                target[final_key] = current + update["delta"]
            elif "value" in update:
                # Set absolute value
                target[final_key] = update["value"]
            elif "expected" in update:
                # Set to expected value
                target[final_key] = update["expected"]

        return new_state

    def format_phase_summary(self, phase_change: PhaseChange) -> str:
        """Format a summary of phase changes for display.

        Args:
            phase_change: The phase change.

        Returns:
            Human-readable summary.
        """
        lines = [f"=== {phase_change.description} ==="]

        for path, update in phase_change.expected_updates.items():
            if "delta" in update:
                sign = "+" if update["delta"] > 0 else ""
                lines.append(f"  {path}: {sign}{update['delta']}")
            elif "value" in update:
                lines.append(f"  {path} -> {update['value']}")
            elif "expected" in update:
                lines.append(f"  {path} should be {update['expected']}")

        return "\n".join(lines)
