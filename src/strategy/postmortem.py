"""Postmortem - AI reflects on the game and writes notes for future sessions."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..engine.state_manager import StateManager
from ..llm.client import LLMClient


@dataclass
class PostmortemReport:
    """AI's post-game analysis."""

    outcome: str  # win, loss, draw, incomplete
    strategy_assessment: str
    what_worked: list[str]
    what_didnt_work: list[str]
    key_moments: list[str]
    lessons_learned: list[str]
    suggestions_for_next_time: list[str]
    opponent_tendencies: list[str]


POSTMORTEM_PROMPT = """You just finished playing a board game. Reflect on how it went and write notes to help yourself play better next time.

GAME: {game_name}

FINAL STATE:
{final_state}

YOUR STRATEGY THROUGHOUT:
Core strategy: {strategy_core}
Final tactical state: {strategy_live}

MOVE HISTORY:
{move_log}

OUTCOME: {outcome}

Write a thoughtful postmortem. Be specific about:
1. What strategy did you pursue? Did it work?
2. What moves or decisions worked well?
3. What moves or decisions were mistakes?
4. Were there key turning points?
5. What did you learn about this game?
6. What should you do differently next time?
7. Did you notice any patterns in how your opponent played?

Return ONLY valid JSON:
{{
  "outcome": "win|loss|draw|incomplete",
  "strategy_assessment": "Brief assessment of your overall strategy",
  "what_worked": ["specific things that went well"],
  "what_didnt_work": ["specific mistakes or problems"],
  "key_moments": ["turning points in the game"],
  "lessons_learned": ["insights about the game mechanics or strategy"],
  "suggestions_for_next_time": ["concrete advice for future games"],
  "opponent_tendencies": ["patterns noticed in opponent's play"]
}}"""


class PostmortemWriter:
    """Generates post-game analysis and notes."""

    def __init__(
        self,
        client: LLMClient,
        game_name: str,
        state_manager: StateManager,
    ):
        """Initialize the postmortem writer.

        Args:
            client: LLM client.
            game_name: Name of the game.
            state_manager: State manager for the session.
        """
        self.client = client
        self.game_name = game_name
        self.state_manager = state_manager

    async def generate_postmortem(
        self,
        outcome: str = "incomplete",
    ) -> PostmortemReport:
        """Generate a postmortem analysis of the game.

        Args:
            outcome: Game outcome (win, loss, draw, incomplete).

        Returns:
            PostmortemReport with analysis.
        """
        # Gather game data
        final_state = self.state_manager.get_state()
        strategy_core = self.state_manager.load_strategy_core()
        strategy_live = self.state_manager.load_strategy_live()
        moves = self.state_manager.load_moves()

        # Format move log
        move_log_text = self._format_move_log(moves)

        prompt = POSTMORTEM_PROMPT.format(
            game_name=self.game_name,
            final_state=json.dumps(final_state, indent=2),
            strategy_core=json.dumps(strategy_core, indent=2),
            strategy_live=json.dumps(strategy_live, indent=2),
            move_log=move_log_text,
            outcome=outcome,
        )

        response = await self.client.complete(
            system_prompt="You are reflecting on a game you just played. Be honest and analytical.",
            user_prompt=prompt,
            temperature=0.7,
        )

        try:
            data = response.as_json()
            return PostmortemReport(
                outcome=data.get("outcome", outcome),
                strategy_assessment=data.get("strategy_assessment", ""),
                what_worked=data.get("what_worked", []),
                what_didnt_work=data.get("what_didnt_work", []),
                key_moments=data.get("key_moments", []),
                lessons_learned=data.get("lessons_learned", []),
                suggestions_for_next_time=data.get("suggestions_for_next_time", []),
                opponent_tendencies=data.get("opponent_tendencies", []),
            )
        except json.JSONDecodeError:
            return PostmortemReport(
                outcome=outcome,
                strategy_assessment="Unable to generate detailed analysis",
                what_worked=[],
                what_didnt_work=[],
                key_moments=[],
                lessons_learned=[],
                suggestions_for_next_time=[],
                opponent_tendencies=[],
            )

    def _format_move_log(self, moves: list, max_moves: int = 50) -> str:
        """Format move log for the prompt.

        Args:
            moves: List of move entries.
            max_moves: Maximum moves to include.

        Returns:
            Formatted move log string.
        """
        if not moves:
            return "(No moves recorded)"

        # Take last N moves if too many
        if len(moves) > max_moves:
            moves = moves[-max_moves:]
            prefix = f"(Showing last {max_moves} of {len(moves)} moves)\n\n"
        else:
            prefix = ""

        lines = [prefix]
        for move in moves:
            reasoning = f" - {move.reasoning}" if move.reasoning else ""
            status = f" [{move.strategy_status}]" if move.strategy_status else ""
            lines.append(f"Turn {move.turn}: {move.player} -> {move.description}{reasoning}{status}")

        return "\n".join(lines)

    def format_report_markdown(self, report: PostmortemReport) -> str:
        """Format the postmortem as markdown.

        Args:
            report: PostmortemReport to format.

        Returns:
            Markdown string.
        """
        lines = [
            f"# Postmortem: {self.game_name}",
            f"",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Outcome:** {report.outcome}",
            f"",
            f"## Strategy Assessment",
            f"",
            report.strategy_assessment,
            f"",
        ]

        if report.what_worked:
            lines.extend([
                "## What Worked",
                "",
            ])
            for item in report.what_worked:
                lines.append(f"- {item}")
            lines.append("")

        if report.what_didnt_work:
            lines.extend([
                "## What Didn't Work",
                "",
            ])
            for item in report.what_didnt_work:
                lines.append(f"- {item}")
            lines.append("")

        if report.key_moments:
            lines.extend([
                "## Key Moments",
                "",
            ])
            for item in report.key_moments:
                lines.append(f"- {item}")
            lines.append("")

        if report.lessons_learned:
            lines.extend([
                "## Lessons Learned",
                "",
            ])
            for item in report.lessons_learned:
                lines.append(f"- {item}")
            lines.append("")

        if report.suggestions_for_next_time:
            lines.extend([
                "## For Next Time",
                "",
            ])
            for item in report.suggestions_for_next_time:
                lines.append(f"- {item}")
            lines.append("")

        if report.opponent_tendencies:
            lines.extend([
                "## Opponent Tendencies",
                "",
            ])
            for item in report.opponent_tendencies:
                lines.append(f"- {item}")
            lines.append("")

        return "\n".join(lines)

    async def write_postmortem(
        self,
        outcome: str = "incomplete",
    ) -> Path:
        """Generate and save a postmortem to the session directory.

        Args:
            outcome: Game outcome.

        Returns:
            Path to the saved postmortem file.
        """
        report = await self.generate_postmortem(outcome)
        markdown = self.format_report_markdown(report)

        postmortem_path = self.state_manager.session_dir / "postmortem.md"
        with open(postmortem_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return postmortem_path


async def write_postmortem_interactive(
    client: LLMClient,
    game_name: str,
    state_manager: StateManager,
) -> Path | None:
    """Interactive postmortem writing at end of game.

    Args:
        client: LLM client.
        game_name: Game name.
        state_manager: State manager.

    Returns:
        Path to postmortem file, or None if skipped.
    """
    print("\n=== Game Over ===")

    # Ask about outcome
    outcome = input("Game outcome (win/loss/draw/incomplete) [incomplete]: ").strip().lower()
    if outcome not in ("win", "loss", "draw", "incomplete"):
        outcome = "incomplete"

    # Ask if they want a postmortem
    write = input("Generate AI postmortem notes for future games? (yes/no) [yes]: ").strip().lower()
    if write in ("no", "n"):
        return None

    print("\nGenerating postmortem analysis...")

    writer = PostmortemWriter(client, game_name, state_manager)
    postmortem_path = await writer.write_postmortem(outcome)

    print(f"Postmortem saved to: {postmortem_path}")

    # Show summary
    report = await writer.generate_postmortem(outcome)
    print(f"\n{report.strategy_assessment}")

    if report.suggestions_for_next_time:
        print("\nKey takeaways for next time:")
        for suggestion in report.suggestions_for_next_time[:3]:
            print(f"  - {suggestion}")

    return postmortem_path
