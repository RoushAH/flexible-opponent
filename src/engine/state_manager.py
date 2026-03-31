"""State management for game sessions."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class StateDiff:
    """Represents changes between two states."""

    added: dict[str, Any] = field(default_factory=dict)
    removed: dict[str, Any] = field(default_factory=dict)
    changed: dict[str, tuple[Any, Any]] = field(default_factory=dict)  # (old, new)

    def is_empty(self) -> bool:
        """Check if there are no changes."""
        return not self.added and not self.removed and not self.changed

    def to_human_readable(self) -> str:
        """Format the diff for human consumption."""
        lines = []

        if self.added:
            lines.append("Added:")
            for key, value in self.added.items():
                lines.append(f"  + {key}: {value}")

        if self.removed:
            lines.append("Removed:")
            for key, value in self.removed.items():
                lines.append(f"  - {key}: {value}")

        if self.changed:
            lines.append("Changed:")
            for key, (old, new) in self.changed.items():
                lines.append(f"  ~ {key}: {old} -> {new}")

        return "\n".join(lines) if lines else "No changes"


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary with dot notation keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compute_diff(old_state: dict, new_state: dict) -> StateDiff:
    """Compute the difference between two states."""
    old_flat = _flatten_dict(old_state)
    new_flat = _flatten_dict(new_state)

    old_keys = set(old_flat.keys())
    new_keys = set(new_flat.keys())

    diff = StateDiff()

    # Added keys
    for key in new_keys - old_keys:
        diff.added[key] = new_flat[key]

    # Removed keys
    for key in old_keys - new_keys:
        diff.removed[key] = old_flat[key]

    # Changed values
    for key in old_keys & new_keys:
        if old_flat[key] != new_flat[key]:
            diff.changed[key] = (old_flat[key], new_flat[key])

    return diff


@dataclass
class MoveLogEntry:
    """A single entry in the move log."""

    turn: int
    player: str
    action_id: str
    description: str
    timestamp: str
    state_after: dict[str, Any] | None = None
    reasoning: str | None = None
    strategy_status: str | None = None  # continue, delay, adapt, pivot

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "turn": self.turn,
            "player": self.player,
            "action_id": self.action_id,
            "description": self.description,
            "timestamp": self.timestamp,
            "state_after": self.state_after,
            "reasoning": self.reasoning,
            "strategy_status": self.strategy_status,
        }


class StateManager:
    """Manages game state files for a session."""

    def __init__(self, session_dir: Path):
        """Initialize the state manager.

        Args:
            session_dir: Path to the game session directory.
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.state_file = self.session_dir / "state.json"
        self.hidden_ai_file = self.session_dir / "hidden_ai.json"
        self.strategy_core_file = self.session_dir / "strategy_core.json"
        self.strategy_live_file = self.session_dir / "strategy_live.json"
        self.move_log_file = self.session_dir / "move_log.jsonl"
        self.schema_file = self.session_dir / "schema.json"

        # In-memory state cache
        self._state_cache: dict | None = None
        self._previous_state: dict | None = None

    # --- State ---

    def load_state(self) -> dict:
        """Load the current game state."""
        if self.state_file.exists():
            with open(self.state_file, encoding="utf-8") as f:
                self._state_cache = json.load(f)
        else:
            self._state_cache = {}
        return self._state_cache

    def save_state(self, state: dict) -> StateDiff:
        """Save the game state and return what changed.

        Args:
            state: The new state to save.

        Returns:
            StateDiff showing what changed from the previous state.
        """
        # Compute diff before saving
        old_state = self._state_cache or {}
        diff = compute_diff(old_state, state)

        # Save to file
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # Update cache
        self._previous_state = old_state
        self._state_cache = state

        return diff

    def get_state(self) -> dict:
        """Get the current state (from cache or file)."""
        if self._state_cache is None:
            return self.load_state()
        return self._state_cache

    # --- Hidden AI State ---

    def load_hidden_ai(self) -> dict:
        """Load the AI's hidden information."""
        if self.hidden_ai_file.exists():
            with open(self.hidden_ai_file, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_hidden_ai(self, hidden: dict) -> None:
        """Save the AI's hidden information."""
        with open(self.hidden_ai_file, "w", encoding="utf-8") as f:
            json.dump(hidden, f, indent=2)

    # --- Strategy ---

    def load_strategy_core(self) -> dict:
        """Load the stable long-term strategy."""
        if self.strategy_core_file.exists():
            with open(self.strategy_core_file, encoding="utf-8") as f:
                return json.load(f)
        return {
            "long_term_plan": None,
            "plan_confidence": 0.0,
            "pivot_conditions": [],
        }

    def save_strategy_core(self, strategy: dict) -> None:
        """Save the stable long-term strategy."""
        with open(self.strategy_core_file, "w", encoding="utf-8") as f:
            json.dump(strategy, f, indent=2)

    def load_strategy_live(self) -> dict:
        """Load the turn-by-turn tactical strategy."""
        if self.strategy_live_file.exists():
            with open(self.strategy_live_file, encoding="utf-8") as f:
                return json.load(f)
        return {
            "current_priorities": [],
            "blocked_goals": [],
            "turn_focus": None,
            "strategy_status": "continue",
            "fallbacks": {},
        }

    def save_strategy_live(self, strategy: dict) -> None:
        """Save the turn-by-turn tactical strategy."""
        with open(self.strategy_live_file, "w", encoding="utf-8") as f:
            json.dump(strategy, f, indent=2)

    # --- Move Log ---

    def append_move(self, entry: MoveLogEntry) -> None:
        """Append a move to the log."""
        with open(self.move_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def load_moves(self, last_n: int | None = None) -> list[MoveLogEntry]:
        """Load moves from the log.

        Args:
            last_n: If specified, only load the last N moves.

        Returns:
            List of move entries.
        """
        if not self.move_log_file.exists():
            return []

        entries = []
        with open(self.move_log_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(
                        MoveLogEntry(
                            turn=data["turn"],
                            player=data["player"],
                            action_id=data["action_id"],
                            description=data["description"],
                            timestamp=data["timestamp"],
                            state_after=data.get("state_after"),
                            reasoning=data.get("reasoning"),
                            strategy_status=data.get("strategy_status"),
                        )
                    )

        if last_n is not None:
            entries = entries[-last_n:]

        return entries

    def get_turn_number(self) -> int:
        """Get the current turn number."""
        moves = self.load_moves()
        if not moves:
            return 1
        return moves[-1].turn + 1

    # --- Schema ---

    def load_schema(self) -> dict | None:
        """Load the state schema."""
        if self.schema_file.exists():
            with open(self.schema_file, encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_schema(self, schema: dict) -> None:
        """Save the state schema."""
        with open(self.schema_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

    # --- Utility ---

    def format_state_for_display(self, include_hidden: bool = False) -> str:
        """Format the current state for human display.

        Args:
            include_hidden: If True, include AI hidden state (for debugging).

        Returns:
            Human-readable state summary.
        """
        state = self.get_state()
        lines = ["=== Current Game State ===", ""]

        if not state:
            lines.append("(No state loaded)")
        else:
            lines.append(json.dumps(state, indent=2))

        if include_hidden:
            hidden = self.load_hidden_ai()
            lines.extend(["", "=== AI Hidden State ===", ""])
            if hidden:
                lines.append(json.dumps(hidden, indent=2))
            else:
                lines.append("(No hidden state)")

        return "\n".join(lines)


def create_session(game_name: str, base_dir: Path | None = None) -> StateManager:
    """Create a new game session.

    Args:
        game_name: Name of the game being played.
        base_dir: Base directory for game sessions. Defaults to ./games/

    Returns:
        StateManager for the new session.
    """
    if base_dir is None:
        base_dir = Path("games")

    # Create session directory with timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    session_name = f"{game_name}_{timestamp}"
    session_dir = base_dir / session_name

    return StateManager(session_dir)


def load_session(session_dir: Path) -> StateManager:
    """Load an existing game session.

    Args:
        session_dir: Path to the session directory.

    Returns:
        StateManager for the session.

    Raises:
        FileNotFoundError: If the session directory doesn't exist.
    """
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    return StateManager(session_dir)
