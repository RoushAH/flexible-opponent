"""Command-line interface for the game opponent."""

import asyncio
import json
import sys
from pathlib import Path

from ..engine.state_manager import StateManager, create_session, load_session
from ..llm.client import create_client, list_providers, get_provider, PROVIDERS
from .game_loop import GameLoop


class CLI:
    """Command-line interface for playing against the AI."""

    def __init__(self):
        """Initialize the CLI."""
        self.game_loop: GameLoop | None = None
        self.state_manager: StateManager | None = None
        self.rules_index = None  # RulesIndex instance
        self._running = True
        self._current_provider = "anthropic"
        self._current_model = None

    def print(self, message: str) -> None:
        """Print a message to the console."""
        print(message)

    def print_separator(self) -> None:
        """Print a visual separator."""
        print("-" * 50)

    def print_help(self) -> None:
        """Print available commands."""
        self.print("""
Commands:
  /state           - Show full current game state
  /diff            - Show what changed last turn
  /moves [n]       - Show last n moves (default 5)
  /strategy        - Show AI's current strategy
  /rules <query>   - Search rules for a topic
  /model           - Change LLM provider/model
  /recover         - Resync state from a photo
  /games           - List previous games (for replay)
  /endgame         - End game and write postmortem
  /quit            - Quit without postmortem
  /help            - Show this help

During play:
  - When it's your turn, describe your move
  - When it's AI's turn, type 'go' or just press Enter
  - To correct state: "Actually I have 5 wood not 4"
""")

    async def setup_new_game(self) -> bool:
        """Interactive setup for a new game.

        Returns:
            True if setup completed successfully.
        """
        self.print("\n=== New Game Setup ===\n")

        # Get game name
        game_name = input("Game name: ").strip()
        if not game_name:
            self.print("Game name is required")
            return False

        # Create session
        self.state_manager = create_session(game_name)
        self.print(f"Created session: {self.state_manager.session_dir}")

        # Get player configuration
        self.print("\nPlayer setup:")
        human_name = input("Your name (default 'human'): ").strip() or "human"
        ai_name = "ai"

        # Ask who goes first
        first = input(f"Who goes first? ({human_name}/ai) [ai]: ").strip().lower()
        if first == human_name.lower():
            player_order = [human_name, ai_name]
        else:
            player_order = [ai_name, human_name]

        # Create LLM client - allow provider selection
        self.print("\nLLM Provider setup:")
        providers = list_providers()
        for i, p in enumerate(providers, 1):
            self.print(f"  {i}. {p.display_name}")

        provider_choice = input(f"Choose provider (1-{len(providers)}) [1]: ").strip()
        if provider_choice.isdigit() and 1 <= int(provider_choice) <= len(providers):
            chosen_provider = providers[int(provider_choice) - 1]
        else:
            chosen_provider = providers[0]  # Default to first (Anthropic)

        self._current_provider = chosen_provider.name

        # Show available models for this provider
        self.print(f"\nAvailable models for {chosen_provider.display_name}:")
        for i, m in enumerate(chosen_provider.available_models, 1):
            default_marker = " (default)" if m == chosen_provider.default_model else ""
            self.print(f"  {i}. {m}{default_marker}")

        model_choice = input(f"Choose model (1-{len(chosen_provider.available_models)}) [1]: ").strip()
        if model_choice.isdigit() and 1 <= int(model_choice) <= len(chosen_provider.available_models):
            chosen_model = chosen_provider.available_models[int(model_choice) - 1]
        else:
            chosen_model = chosen_provider.default_model

        self._current_model = chosen_model

        self.print(f"\nConnecting to {chosen_provider.display_name} ({chosen_model})...")
        try:
            client = create_client(provider=self._current_provider, model=chosen_model)
        except Exception as e:
            self.print(f"Failed to create LLM client: {e}")
            if chosen_provider.requires_env:
                self.print(f"Make sure these are set: {', '.join(chosen_provider.requires_env)}")
            return False

        # Create game loop
        self.game_loop = GameLoop(
            state_manager=self.state_manager,
            client=client,
            game_name=game_name,
            player_order=player_order,
            ai_player=ai_name,
        )

        # Get rules
        self.print("\nRules setup:")
        self.print("Options:")
        self.print("  1. Enter path to rulebook file (PDF, .txt, or images)")
        self.print("  2. Enter path to folder containing all rulebook files")
        self.print("  3. Paste rules text (type 'END' on a new line when done)")
        self.print("  4. Press Enter to skip")

        rules_input = input("\nRulebook path or text: ").strip()
        rules_text = ""

        if rules_input:
            # Check if it's a file path
            path = Path(rules_input)
            if path.exists():
                self.print(f"Processing rulebook: {path}")
                try:
                    from ..setup.rulebook_processor import RulebookProcessor

                    processor = RulebookProcessor(
                        game_name=game_name,
                        session_dir=self.state_manager.session_dir,
                        client=client,
                    )
                    result = await processor.process(path)
                    self.rules_index = result.index
                    self.print(result.summary())

                    # Set rules index on rules interpreter
                    self.game_loop.rules_interpreter.set_rules_index(self.rules_index)

                    # Get full text for strategy init
                    rules_text = result.index.get_all_chunks()
                    if rules_text:
                        rules_text = "\n\n".join(c.content[:500] for c in rules_text[:5])

                except Exception as e:
                    self.print(f"Failed to process rulebook: {e}")
                    self.print("Falling back to text input...")
                    rules_input = ""

            if not path.exists() or not self.rules_index:
                # Treat as start of pasted text
                rules_lines = [rules_input] if rules_input else []
                self.print("Paste rules text (type 'END' on a new line when done):")
                while True:
                    line = input()
                    if line.strip().upper() == "END":
                        break
                    rules_lines.append(line)
                rules_text = "\n".join(rules_lines)

                if rules_text.strip():
                    # Index the pasted text too
                    try:
                        from ..setup.rulebook_processor import RulebookProcessor

                        processor = RulebookProcessor(
                            game_name=game_name,
                            session_dir=self.state_manager.session_dir,
                            client=client,
                        )
                        result = await processor.process(rules_text)
                        self.rules_index = result.index
                        self.game_loop.rules_interpreter.set_rules_index(self.rules_index)
                        self.print(f"Indexed {result.chunk_count} rule chunks")
                    except Exception as e:
                        self.print(f"Indexing failed: {e}, using raw text")
                        self.game_loop.set_rules(rules_text)
        else:
            self.print("No rules provided - AI will play without rule knowledge")

        # Get initial state
        self.print("\nInitial state setup:")
        self.print("Enter initial game state as JSON (or press Enter to skip):")
        state_input = input().strip()

        if state_input:
            try:
                initial_state = json.loads(state_input)
                self.state_manager.save_state(initial_state)
                self.print("Initial state loaded")
            except json.JSONDecodeError as e:
                self.print(f"Invalid JSON: {e}")
                self.print("Starting with empty state")
                self.state_manager.save_state({})
        else:
            self.state_manager.save_state({"phase": "setup", "turn": 1})

        # Initialize AI strategy
        self.print("\nInitializing AI strategy...")
        try:
            from ..llm.strategist import Strategist

            strategist = Strategist(client, game_name)
            strategy_core, strategy_live = await strategist.initialize_strategy(
                rules_summary=rules_text[:2000] if rules_text else "No rules provided",
                initial_state=self.state_manager.get_state(),
                ai_hidden=self.state_manager.load_hidden_ai(),
            )
            self.state_manager.save_strategy_core(strategy_core)
            self.state_manager.save_strategy_live(strategy_live)
            self.print(f"AI strategy: {strategy_core.get('long_term_plan', 'adaptive')}")
        except Exception as e:
            self.print(f"Strategy init failed: {e}")
            # Use defaults

        self.print_separator()
        self.print(f"\nGame ready! {player_order[0]} goes first.")
        self.print("Type /help for commands\n")

        return True

    async def run_game(self) -> None:
        """Run the main game loop."""
        if not self.game_loop or not self.state_manager:
            self.print("No game loaded")
            return

        while self._running and self.game_loop.is_active:
            current = self.game_loop.current_player
            is_ai = self.game_loop.is_ai_turn

            if is_ai:
                self.print(f"\n[{current}'s turn - AI thinking...]")

                try:
                    result = await self.game_loop.execute_ai_turn()

                    self.print_separator()
                    self.print(f"AI plays: {result.description}")
                    if result.reasoning:
                        self.print(f"Reasoning: {result.reasoning}")
                    if not result.state_diff.is_empty():
                        self.print(f"\n{result.state_diff.to_human_readable()}")
                    self.print_separator()

                except Exception as e:
                    self.print(f"AI turn failed: {e}")
                    # In case of error, let human intervene
                    self.print("Enter 'skip' to skip AI turn, or describe AI's move:")
                    response = input("> ").strip()
                    if response.lower() != "skip":
                        self.game_loop.process_human_move(
                            player=self.game_loop.ai_player,
                            action_id="manual",
                            description=response,
                        )
                    else:
                        self.game_loop.advance_turn()

            else:
                self.print(f"\n[{current}'s turn]")
                user_input = input("> ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    await self.handle_command(user_input)
                    continue

                # Process as a move
                self.game_loop.process_human_move(
                    player=current,
                    action_id="human_move",
                    description=user_input,
                )
                self.print("Move recorded.")

        self.print("\nGame ended.")

    async def handle_command(self, command: str) -> None:
        """Handle a slash command."""
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "help":
            self.print_help()

        elif cmd == "quit":
            self._running = False
            if self.game_loop:
                self.game_loop.end_game()

        elif cmd == "state":
            if self.state_manager:
                self.print(self.state_manager.format_state_for_display())
            else:
                self.print("No game loaded")

        elif cmd == "diff":
            if self.state_manager and self.state_manager._previous_state:
                from ..engine.state_manager import compute_diff

                diff = compute_diff(
                    self.state_manager._previous_state,
                    self.state_manager.get_state(),
                )
                self.print(diff.to_human_readable())
            else:
                self.print("No previous state to compare")

        elif cmd == "moves":
            if self.state_manager:
                n = int(args) if args.isdigit() else 5
                moves = self.state_manager.load_moves(last_n=n)
                if moves:
                    for move in moves:
                        self.print(f"Turn {move.turn} - {move.player}: {move.description}")
                else:
                    self.print("No moves yet")
            else:
                self.print("No game loaded")

        elif cmd == "strategy":
            if self.state_manager:
                core = self.state_manager.load_strategy_core()
                live = self.state_manager.load_strategy_live()
                self.print("=== AI Strategy ===")
                self.print(f"Long-term plan: {core.get('long_term_plan', 'none')}")
                self.print(f"Confidence: {core.get('plan_confidence', 0):.0%}")
                self.print(f"Status: {live.get('strategy_status', 'unknown')}")
                self.print(f"Focus: {live.get('turn_focus', 'none')}")
                priorities = live.get("current_priorities", [])
                if priorities:
                    self.print(f"Priorities: {', '.join(priorities)}")
            else:
                self.print("No game loaded")

        elif cmd == "rules":
            if not args:
                self.print("Usage: /rules <query>")
                self.print("Example: /rules how does harvesting work")
            elif self.rules_index:
                chunks = self.rules_index.query(args, n_results=3)
                if chunks:
                    for chunk in chunks:
                        self.print_separator()
                        self.print(f"## {chunk.title}")
                        self.print(chunk.content[:500])
                        if len(chunk.content) > 500:
                            self.print("...")
                else:
                    self.print("No matching rules found")
            else:
                self.print("No rules indexed for this game")

        elif cmd == "recover":
            if self.state_manager and self.game_loop:
                from .recovery import recover_state_interactive

                await recover_state_interactive(
                    client=self.game_loop.client,
                    game_name=self.game_loop.game_name,
                    state_manager=self.state_manager,
                    schema=self.state_manager.load_schema(),
                    rules_index=self.rules_index,
                )
            else:
                self.print("No game loaded")

        elif cmd == "games":
            from ..engine.game_manager import list_available_games

            list_available_games()

        elif cmd == "model":
            await self.handle_model_change()

        elif cmd == "endgame":
            if self.state_manager and self.game_loop:
                from ..strategy.postmortem import write_postmortem_interactive

                await write_postmortem_interactive(
                    client=self.game_loop.client,
                    game_name=self.game_loop.game_name,
                    state_manager=self.state_manager,
                )
                self._running = False
                self.game_loop.end_game()
            else:
                self.print("No game loaded")

        else:
            self.print(f"Unknown command: {cmd}")
            self.print("Type /help for available commands")

    async def handle_model_change(self) -> None:
        """Handle the /model command to change LLM provider/model."""
        self.print(f"\nCurrent: {self._current_provider} / {self._current_model}")
        self.print("\nAvailable providers:")

        providers = list_providers()
        for i, p in enumerate(providers, 1):
            marker = " *" if p.name == self._current_provider else ""
            self.print(f"  {i}. {p.display_name}{marker}")

        self.print(f"\n  0. Keep current ({self._current_provider})")

        provider_choice = input("Choose provider: ").strip()

        if not provider_choice or provider_choice == "0":
            self.print("Keeping current model")
            return

        if not provider_choice.isdigit() or not (1 <= int(provider_choice) <= len(providers)):
            self.print("Invalid choice")
            return

        chosen_provider = providers[int(provider_choice) - 1]

        # Show models
        self.print(f"\nModels for {chosen_provider.display_name}:")
        for i, m in enumerate(chosen_provider.available_models, 1):
            default_marker = " (default)" if m == chosen_provider.default_model else ""
            self.print(f"  {i}. {m}{default_marker}")

        model_choice = input(f"Choose model (1-{len(chosen_provider.available_models)}) [1]: ").strip()
        if model_choice.isdigit() and 1 <= int(model_choice) <= len(chosen_provider.available_models):
            chosen_model = chosen_provider.available_models[int(model_choice) - 1]
        else:
            chosen_model = chosen_provider.default_model

        # Create new client and swap it
        self.print(f"\nSwitching to {chosen_provider.display_name} ({chosen_model})...")
        try:
            new_client = create_client(provider=chosen_provider.name, model=chosen_model)
            if self.game_loop:
                self.game_loop.set_client(new_client)
            self._current_provider = chosen_provider.name
            self._current_model = chosen_model
            self.print("Model switched successfully!")
        except Exception as e:
            self.print(f"Failed to switch: {e}")
            if chosen_provider.requires_env:
                self.print(f"Make sure these are set: {', '.join(chosen_provider.requires_env)}")


async def async_main() -> None:
    """Async main entry point."""
    cli = CLI()

    print("=" * 50)
    print("  Flexible Opponent - Board Game AI")
    print("=" * 50)

    # Setup new game
    success = await cli.setup_new_game()
    if not success:
        print("Setup failed")
        return

    # Run the game
    await cli.run_game()


def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n\nGame interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
