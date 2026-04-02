"""Main game loop orchestration."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from ..engine.state_manager import MoveLogEntry, StateManager, StateDiff
from ..llm.client import LLMClient
from ..llm.referee import Referee, ValidationResult
from ..llm.rules_interpreter import LegalActionsResult, RulesInterpreter
from ..llm.strategist import MoveDecision, Strategist


@dataclass
class TurnResult:
    """Result of a completed turn."""

    player: str
    action_id: str
    description: str
    reasoning: str | None
    state_diff: StateDiff
    strategy_status: str | None = None


class GameLoop:
    """Orchestrates the game turn loop."""

    def __init__(
        self,
        state_manager: StateManager,
        client: LLMClient,
        game_name: str,
        player_order: list[str],
        ai_player: str = "ai",
    ):
        """Initialize the game loop.

        Args:
            state_manager: State manager for the session.
            client: LLM client for AI roles.
            game_name: Name of the game being played.
            player_order: List of player names in turn order.
            ai_player: Name of the AI player (default "ai").
        """
        self.state_manager = state_manager
        self.client = client
        self.game_name = game_name
        self.player_order = player_order
        self.ai_player = ai_player

        self.rules_interpreter = RulesInterpreter(client, game_name)
        self.strategist = Strategist(client, game_name)
        self.referee = Referee(client, game_name)

        self._current_player_idx = 0
        self._turn_number = 1
        self._rules_text = None  # None = use RAG; set via set_rules() for direct text
        self._game_active = True
        self._validation_enabled = True
        self._max_retries = 3  # Max retries if move is invalid

    @property
    def current_player(self) -> str:
        """Get the current player's name."""
        return self.player_order[self._current_player_idx]

    @property
    def is_ai_turn(self) -> bool:
        """Check if it's the AI's turn."""
        return self.current_player == self.ai_player

    def set_client(self, client: LLMClient) -> None:
        """Change the LLM client for all roles.

        Args:
            client: New LLM client to use.
        """
        self.client = client
        self.rules_interpreter = RulesInterpreter(client, self.game_name)
        self.strategist = Strategist(client, self.game_name)
        self.referee = Referee(client, self.game_name)

        # Re-set rules index if we have one
        if hasattr(self, '_rules_index') and self._rules_index:
            self.rules_interpreter.set_rules_index(self._rules_index)
            self.referee.rules_index = self._rules_index

    def set_rules(self, rules_text: str) -> None:
        """Set the rules text for the game.

        Args:
            rules_text: Full rules or rules excerpt for interpretation.
        """
        self._rules_text = rules_text

    def set_rules_index(self, rules_index) -> None:
        """Set the rules index for RAG retrieval.

        Args:
            rules_index: RulesIndex instance.
        """
        self._rules_index = rules_index  # Store for set_client
        self.rules_interpreter.set_rules_index(rules_index)
        self.referee.rules_index = rules_index

    def set_human_hidden(self, hidden: dict) -> None:
        """Set the human's hidden state for the referee.

        Args:
            hidden: Human's hidden information.
        """
        self.referee.set_human_hidden(hidden)

    def set_validation_enabled(self, enabled: bool) -> None:
        """Enable or disable referee validation.

        Args:
            enabled: Whether to validate moves.
        """
        self._validation_enabled = enabled

    def advance_turn(self) -> None:
        """Advance to the next player's turn."""
        self._current_player_idx = (self._current_player_idx + 1) % len(self.player_order)
        if self._current_player_idx == 0:
            self._turn_number += 1

    async def execute_ai_turn(self) -> TurnResult:
        """Execute the AI's turn.

        Returns:
            TurnResult with the AI's move and state changes.
        """
        if not self.is_ai_turn:
            raise RuntimeError("Not AI's turn")

        # Load current state
        visible_state = self.state_manager.get_state()
        ai_hidden = self.state_manager.load_hidden_ai()
        strategy_core = self.state_manager.load_strategy_core()
        strategy_live = self.state_manager.load_strategy_live()

        # Get recent moves for context
        recent_moves = self.state_manager.load_moves(last_n=5)
        recent_moves_dicts = [
            {"player": m.player, "action_id": m.action_id, "description": m.description}
            for m in recent_moves
        ]

        # Get current phase from state
        phase = visible_state.get("phase", "unknown")

        # Step 1: Rules Interpreter - enumerate legal actions
        legal_actions = await self.rules_interpreter.enumerate_actions(
            rules_text=self._rules_text,
            visible_state=visible_state,
            ai_hidden=ai_hidden,
            phase=phase,
            recent_moves=recent_moves_dicts,
        )

        if not legal_actions.actions:
            # No legal actions - this might be a pass or game over
            return TurnResult(
                player=self.ai_player,
                action_id="pass",
                description="No legal actions available - passing",
                reasoning="No actions enumerated by rules interpreter",
                state_diff=StateDiff(),
                strategy_status="continue",
            )

        # Step 2: Strategist - choose best move (with validation retry)
        decision = None
        validation_result = None
        excluded_actions = set()

        for attempt in range(self._max_retries):
            # Filter out previously rejected actions
            available_actions = LegalActionsResult(
                actions=[a for a in legal_actions.actions if a.id not in excluded_actions],
                phase_note=legal_actions.phase_note,
                ambiguities=legal_actions.ambiguities,
            )

            if not available_actions.actions:
                break

            decision = await self.strategist.choose_move(
                legal_actions=available_actions,
                visible_state=visible_state,
                ai_hidden=ai_hidden,
                strategy_core=strategy_core,
                strategy_live=strategy_live,
            )

            # Step 3: Referee validation (if enabled)
            if self._validation_enabled:
                validation_result = await self.referee.validate_move(
                    player=self.ai_player,
                    action_id=decision.chosen_action.id,
                    action_description=decision.chosen_action.description,
                    visible_state=visible_state,
                    ai_hidden=ai_hidden,
                    rules_text=self._rules_text if self._rules_text else None,
                )

                if validation_result.valid:
                    break
                else:
                    # Move rejected - exclude and retry
                    excluded_actions.add(decision.chosen_action.id)
                    decision = None
            else:
                break

        if decision is None:
            # All moves rejected or no valid moves
            return TurnResult(
                player=self.ai_player,
                action_id="pass",
                description="No valid moves available after validation",
                reasoning=validation_result.reason if validation_result else "No moves",
                state_diff=StateDiff(),
                strategy_status="continue",
            )

        # Step 4: Apply the move (update state based on action effects)
        new_state = self._apply_action(visible_state, decision)
        state_diff = self.state_manager.save_state(new_state)

        # Step 5: Update strategy
        self._update_strategy(strategy_live, decision)

        # Step 6: Log the move
        self.state_manager.append_move(
            MoveLogEntry(
                turn=self._turn_number,
                player=self.ai_player,
                action_id=decision.chosen_action.id,
                description=decision.chosen_action.description,
                timestamp=datetime.now().isoformat(),
                reasoning=decision.reasoning,
                strategy_status=decision.strategy_status,
            )
        )

        # Advance turn
        self.advance_turn()

        return TurnResult(
            player=self.ai_player,
            action_id=decision.chosen_action.id,
            description=decision.chosen_action.description,
            reasoning=decision.reasoning,
            state_diff=state_diff,
            strategy_status=decision.strategy_status,
        )

    def process_human_move(
        self,
        player: str,
        action_id: str,
        description: str,
        state_updates: dict | None = None,
    ) -> TurnResult:
        """Process a human player's move.

        Args:
            player: Name of the player making the move.
            action_id: Identifier for the action taken.
            description: Human-readable description of the move.
            state_updates: Optional state changes to apply.

        Returns:
            TurnResult with state changes.
        """
        # Load current state
        current_state = self.state_manager.get_state()

        # Apply updates
        if state_updates:
            new_state = self._merge_state(current_state, state_updates)
        else:
            new_state = current_state

        state_diff = self.state_manager.save_state(new_state)

        # Log the move
        self.state_manager.append_move(
            MoveLogEntry(
                turn=self._turn_number,
                player=player,
                action_id=action_id,
                description=description,
                timestamp=datetime.now().isoformat(),
            )
        )

        # Advance turn
        self.advance_turn()

        return TurnResult(
            player=player,
            action_id=action_id,
            description=description,
            reasoning=None,
            state_diff=state_diff,
        )

    def _apply_action(self, state: dict, decision: MoveDecision) -> dict:
        """Apply an action to the state.

        This is a simplified version - in practice, the action effects
        would need to be interpreted and applied properly.
        """
        # For MVP, we'll rely on the state updates from the move description
        # In a full implementation, this would interpret action.gains, action.cost, etc.
        action = decision.chosen_action

        new_state = state.copy()

        # Apply gains to AI player
        if action.gains and "players" in new_state and self.ai_player in new_state["players"]:
            player_state = new_state["players"][self.ai_player]
            for resource, amount in action.gains.items():
                if resource in player_state:
                    player_state[resource] = player_state.get(resource, 0) + amount

        # Apply costs to AI player
        if action.cost and "players" in new_state and self.ai_player in new_state["players"]:
            player_state = new_state["players"][self.ai_player]
            for resource, amount in action.cost.items():
                if resource in player_state:
                    player_state[resource] = player_state.get(resource, 0) - amount

        return new_state

    def _merge_state(self, current: dict, updates: dict) -> dict:
        """Merge state updates into current state."""
        import copy

        new_state = copy.deepcopy(current)

        def deep_merge(base: dict, overlay: dict) -> None:
            for key, value in overlay.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        deep_merge(new_state, updates)
        return new_state

    def _update_strategy(self, strategy_live: dict, decision: MoveDecision) -> None:
        """Update the live strategy based on the decision."""
        update = decision.strategy_update

        if update.turn_focus:
            strategy_live["turn_focus"] = update.turn_focus

        if update.blocked_goals:
            strategy_live["blocked_goals"] = update.blocked_goals

        if update.priority_changes:
            # Apply priority changes (this could be more sophisticated)
            for change in update.priority_changes:
                if change not in strategy_live.get("current_priorities", []):
                    strategy_live.setdefault("current_priorities", []).insert(0, change)

        strategy_live["strategy_status"] = decision.strategy_status

        self.state_manager.save_strategy_live(strategy_live)

    def get_state_display(self) -> str:
        """Get formatted state for display."""
        return self.state_manager.format_state_for_display()

    def end_game(self) -> None:
        """End the game."""
        self._game_active = False

    @property
    def is_active(self) -> bool:
        """Check if the game is still active."""
        return self._game_active
