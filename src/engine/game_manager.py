"""Game manager - handles sessions, re-play, and game history."""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .rules_index import RulesIndex
from .state_manager import StateManager, create_session


@dataclass
class GameInfo:
    """Information about a previously played game."""

    game_name: str
    session_dir: Path
    created: datetime
    has_rules_index: bool
    has_postmortem: bool
    total_moves: int


@dataclass
class ReplaySetup:
    """Setup data for replaying a game."""

    game_name: str
    rules_index: RulesIndex
    postmortem: str | None
    previous_sessions: list[GameInfo]


class GameManager:
    """Manages game sessions and enables replay."""

    def __init__(self, games_dir: Path | None = None):
        """Initialize the game manager.

        Args:
            games_dir: Directory where game sessions are stored.
        """
        self.games_dir = Path(games_dir) if games_dir else Path("games")
        self.games_dir.mkdir(parents=True, exist_ok=True)

    def list_games(self) -> dict[str, list[GameInfo]]:
        """List all games and their sessions.

        Returns:
            Dict mapping game names to lists of session info.
        """
        games: dict[str, list[GameInfo]] = {}

        for session_dir in self.games_dir.iterdir():
            if not session_dir.is_dir():
                continue

            # Parse session name: gamename_YYYY_MM_DD_HHMMSS
            parts = session_dir.name.rsplit("_", 4)
            if len(parts) >= 5:
                game_name = "_".join(parts[:-4])
                try:
                    timestamp = datetime.strptime(
                        "_".join(parts[-4:]), "%Y_%m_%d_%H%M%S"
                    )
                except ValueError:
                    game_name = session_dir.name
                    timestamp = datetime.fromtimestamp(session_dir.stat().st_mtime)
            else:
                game_name = session_dir.name
                timestamp = datetime.fromtimestamp(session_dir.stat().st_mtime)

            # Check what exists in the session
            rules_index_dir = session_dir / "rules_index"
            postmortem_file = session_dir / "postmortem.md"
            move_log_file = session_dir / "move_log.jsonl"

            # Count moves
            total_moves = 0
            if move_log_file.exists():
                with open(move_log_file, encoding="utf-8") as f:
                    total_moves = sum(1 for _ in f)

            info = GameInfo(
                game_name=game_name,
                session_dir=session_dir,
                created=timestamp,
                has_rules_index=rules_index_dir.exists() and (rules_index_dir / "chunks.json").exists(),
                has_postmortem=postmortem_file.exists(),
                total_moves=total_moves,
            )

            if game_name not in games:
                games[game_name] = []
            games[game_name].append(info)

        # Sort sessions by date (newest first)
        for sessions in games.values():
            sessions.sort(key=lambda x: x.created, reverse=True)

        return games

    def find_game(self, game_name: str) -> list[GameInfo]:
        """Find all sessions for a specific game.

        Args:
            game_name: Name of the game (case-insensitive).

        Returns:
            List of matching session info, newest first.
        """
        all_games = self.list_games()
        game_name_lower = game_name.lower()

        for name, sessions in all_games.items():
            if name.lower() == game_name_lower:
                return sessions

        # Partial match
        for name, sessions in all_games.items():
            if game_name_lower in name.lower():
                return sessions

        return []

    def can_replay(self, game_name: str) -> bool:
        """Check if a game can be replayed (has existing rules index).

        Args:
            game_name: Name of the game.

        Returns:
            True if replay is possible.
        """
        sessions = self.find_game(game_name)
        return any(s.has_rules_index for s in sessions)

    def prepare_replay(self, game_name: str) -> ReplaySetup | None:
        """Prepare to replay a game using existing rules index.

        Args:
            game_name: Name of the game to replay.

        Returns:
            ReplaySetup if possible, None if no existing index.
        """
        sessions = self.find_game(game_name)
        if not sessions:
            return None

        # Find session with rules index
        source_session = None
        for session in sessions:
            if session.has_rules_index:
                source_session = session
                break

        if source_session is None:
            return None

        # Load the rules index
        rules_index_dir = source_session.session_dir / "rules_index"
        rules_index = RulesIndex(rules_index_dir, game_name)

        # Load postmortem if exists (from most recent session with one)
        postmortem = None
        for session in sessions:
            if session.has_postmortem:
                postmortem_file = session.session_dir / "postmortem.md"
                with open(postmortem_file, encoding="utf-8") as f:
                    postmortem = f.read()
                break

        return ReplaySetup(
            game_name=game_name,
            rules_index=rules_index,
            postmortem=postmortem,
            previous_sessions=sessions,
        )

    def create_replay_session(
        self,
        replay_setup: ReplaySetup,
    ) -> tuple[StateManager, RulesIndex]:
        """Create a new session for replaying a game.

        Copies the rules index from a previous session.

        Args:
            replay_setup: Setup data from prepare_replay.

        Returns:
            Tuple of (new StateManager, copied RulesIndex).
        """
        # Create new session
        state_manager = create_session(replay_setup.game_name, self.games_dir)

        # Copy rules index to new session
        source_index_dir = replay_setup.previous_sessions[0].session_dir / "rules_index"
        dest_index_dir = state_manager.session_dir / "rules_index"

        shutil.copytree(source_index_dir, dest_index_dir)

        # Create rules index for new session
        rules_index = RulesIndex(dest_index_dir, replay_setup.game_name)

        return state_manager, rules_index

    def get_session_stats(self, session_dir: Path) -> dict:
        """Get statistics for a session.

        Args:
            session_dir: Session directory.

        Returns:
            Dict with session statistics.
        """
        stats = {
            "moves": 0,
            "has_state": False,
            "has_rules": False,
            "has_strategy": False,
            "has_postmortem": False,
        }

        move_log = session_dir / "move_log.jsonl"
        if move_log.exists():
            with open(move_log, encoding="utf-8") as f:
                stats["moves"] = sum(1 for _ in f)

        stats["has_state"] = (session_dir / "state.json").exists()
        stats["has_rules"] = (session_dir / "rules_index" / "chunks.json").exists()
        stats["has_strategy"] = (session_dir / "strategy_core.json").exists()
        stats["has_postmortem"] = (session_dir / "postmortem.md").exists()

        return stats


def list_available_games(games_dir: Path | None = None) -> None:
    """Print a list of available games for replay.

    Args:
        games_dir: Games directory.
    """
    manager = GameManager(games_dir)
    games = manager.list_games()

    if not games:
        print("No previous games found.")
        return

    print("\n=== Available Games ===\n")

    for game_name, sessions in sorted(games.items()):
        replay_marker = " [can replay]" if any(s.has_rules_index for s in sessions) else ""
        print(f"{game_name}{replay_marker}")

        for session in sessions[:3]:  # Show last 3 sessions
            date_str = session.created.strftime("%Y-%m-%d %H:%M")
            postmortem = " (has notes)" if session.has_postmortem else ""
            print(f"    {date_str} - {session.total_moves} moves{postmortem}")

        if len(sessions) > 3:
            print(f"    ... and {len(sessions) - 3} more sessions")
        print()
