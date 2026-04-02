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

                    # Set rules index on game loop (propagates to interpreter + referee)
                    self.game_loop.set_rules_index(self.rules_index)

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
                        self.game_loop.set_rules_index(self.rules_index)
                        self.print(f"Indexed {result.chunk_count} rule chunks")
                    except Exception as e:
                        self.print(f"Indexing failed: {e}, using raw text")
                        self.game_loop.set_rules(rules_text)
        else:
            self.print("No rules provided - AI will play without rule knowledge")

        # Generate initial state from rules
        self.print("\nGenerating initial game state from rules...")
        initial_state = {"phase": "setup", "turn": 1, "players": {human_name: {}, ai_name: {}}}

        if self.rules_index or rules_text:
            try:
                from ..setup.schema_generator import SchemaGenerator

                generator = SchemaGenerator(client, game_name)
                proposal = await generator.propose_schema(
                    rules_index=self.rules_index,
                    rules_text=rules_text,
                )

                # Normalize player names before showing preview
                normalized_state = self._normalize_player_names(
                    proposal.example_state, human_name, ai_name, player_order
                )

                # Show what we generated
                self.print("\n=== Proposed Initial State ===")
                self.print(proposal.description)
                self.print("\nTracking: " + ", ".join(proposal.tracked_elements[:5]))
                self.print(f"\nInitial state preview:")
                preview = json.dumps(normalized_state, indent=2)
                # Show truncated preview
                if len(preview) > 500:
                    self.print(preview[:500] + "\n  ...")
                else:
                    self.print(preview)

                # Ask for confirmation
                self.print("\nDoes this look correct?")
                response = input("(yes/no/describe changes) [yes]: ").strip().lower()

                if response in ("", "yes", "y"):
                    initial_state = normalized_state
                    self.state_manager.save_schema(proposal.schema)
                    self.print("Initial state accepted")
                elif response in ("no", "n"):
                    self.print("Using minimal state - you can correct it during play")
                else:
                    # Treat as feedback, refine
                    self.print("Refining based on your feedback...")
                    refined = await generator.refine_schema(
                        proposal.schema,
                        response,
                        rules_text,
                    )
                    initial_state = self._normalize_player_names(
                        refined.example_state, human_name, ai_name, player_order
                    )
                    self.state_manager.save_schema(refined.schema)
                    self.print("Refined state applied")

            except Exception as e:
                self.print(f"State generation failed: {e}")
                self.print("Using minimal initial state")

        self.state_manager.save_state(initial_state)

        # Check if game has hidden state (cards dealt to players)
        if self.rules_index or rules_text:
            await self._setup_hidden_state(client, game_name, human_name, rules_text)

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

        # Sync state with game loop's turn order
        self.game_loop.sync_state_with_turn()

        self.print_separator()
        self.print(f"\nGame ready! {player_order[0]} goes first.")
        self.print("Type /help for commands\n")

        return True

    def _normalize_player_names(
        self,
        state: dict,
        human_name: str,
        ai_name: str,
        player_order: list[str] | None = None,
    ) -> dict:
        """Replace generic player names with actual names and fix turn order.

        Schema generator often uses 'player1'/'player2' - replace with real names.
        Also sets correct turn_order and current_player based on who goes first.
        """
        import copy
        state = copy.deepcopy(state)

        # Map generic names to actual names
        name_map = {
            "player1": human_name,
            "player2": ai_name,
            "Player1": human_name,
            "Player2": ai_name,
            "Player 1": human_name,
            "Player 2": ai_name,
        }

        def replace_in_obj(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    # Replace key if it's a player name
                    new_key = name_map.get(k, k)
                    new_dict[new_key] = replace_in_obj(v)
                return new_dict
            elif isinstance(obj, list):
                return [replace_in_obj(item) for item in obj]
            elif isinstance(obj, str):
                # Replace string values that are player names
                return name_map.get(obj, obj)
            else:
                return obj

        state = replace_in_obj(state)

        # Fix turn order and current player if player_order provided
        if player_order:
            state["turn_order"] = player_order
            state["current_player"] = player_order[0]

        return state

    async def _setup_hidden_state(
        self,
        client,
        game_name: str,
        human_name: str,
        rules_text: str | None,
    ) -> None:
        """Check if game has hidden state and set it up.

        Args:
            client: LLM client.
            game_name: Name of the game.
            human_name: Human player's name.
            rules_text: Rules text for context.
        """
        # First, ask LLM what types of hidden items are dealt in this game
        structure_prompt = f"""What hidden cards/items are dealt to each player at the start of "{game_name}"?

Rules context:
{(rules_text or "")[:3000]}

Return ONLY valid JSON:
{{
  "has_hidden_items": true|false,
  "item_types": [
    {{"name": "occupations", "count": 7}},
    {{"name": "minor improvements", "count": 7}}
  ],
  "dealing_notes": "brief description of how cards are dealt"
}}

If no hidden items are dealt, set has_hidden_items to false and item_types to empty array."""

        try:
            response = await client.complete(
                system_prompt="Return only valid JSON.",
                user_prompt=structure_prompt,
                temperature=0.2,
                max_tokens=500,
            )

            try:
                structure = response.as_json()
            except:
                # Fallback to simple yes/no check
                structure = {"has_hidden_items": "agricola" in game_name.lower(), "item_types": []}

            if not structure.get("has_hidden_items", False):
                return  # No hidden state needed

            self.print("\n=== Hidden Cards/Items Setup ===")

            item_types = structure.get("item_types", [])
            human_hidden = {"hand": []}

            if item_types:
                # Ask for each type separately
                for item_type in item_types:
                    type_name = item_type.get("name", "items")
                    count = item_type.get("count", "some")

                    self.print(f"\n{human_name}, what {type_name} did you receive? (You should have {count})")
                    self.print("(Enter comma-separated list, or press Enter if none)")

                    user_input = input("> ").strip()
                    if user_input:
                        items = [item.strip() for item in user_input.split(",")]
                        human_hidden.setdefault(type_name.replace(" ", "_"), []).extend(items)
                        human_hidden["hand"].extend(items)
                        self.print(f"  Recorded {len(items)} {type_name}")
            else:
                # Fallback to generic question
                self.print(f"\n{human_name}, what cards/items were you dealt?")
                self.print("(Enter comma-separated list, or press Enter if none)")

                user_input = input("> ").strip()
                if user_input:
                    items = [item.strip() for item in user_input.split(",")]
                    human_hidden["hand"] = items

            if not human_hidden.get("hand"):
                self.print("No hidden items recorded.")
                return

            total_items = len(human_hidden.get("hand", []))
            self.print(f"\nRecorded {total_items} total item(s) for {human_name}.")

            # Deal to AI with structured info
            self.print("Dealing hidden items to AI based on rules...")

            from ..setup.hidden_dealer import HiddenDealer

            dealer = HiddenDealer(client, game_name, self.rules_index)
            result = await dealer.deal_hidden_state(
                human_hidden,
                rules_text=rules_text,
            )

            self.print(f"  {result.dealing_notes}")
            ai_total = len(result.ai_hidden.get("hand", []))
            self.print(f"  AI received {ai_total} item(s)")

            # Save hidden states
            self.state_manager.save_hidden_ai(result.ai_hidden)
            if self.game_loop:
                self.game_loop.set_human_hidden(result.human_hidden)

        except Exception as e:
            self.print(f"Hidden state setup skipped: {e}")

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

                    # Check for phase/round changes after AI turn
                    phase_change = await self.game_loop.check_phase_change(
                        player=current,
                        action=result.description,
                    )
                    if phase_change:
                        self.print(f"\n{self.game_loop.phase_tracker.format_phase_summary(phase_change)}")
                        self.game_loop.apply_phase_effects(phase_change)

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

                # Process as a move - interpret with LLM to extract state changes
                self.print("Interpreting move...")
                result, interpreted = await self.game_loop.interpret_and_process_human_move(
                    player=current,
                    description=user_input,
                )
                self.print(f"Move recorded. ({interpreted.confidence} confidence)")
                if not result.state_diff.is_empty():
                    self.print(f"{result.state_diff.to_human_readable()}")

                # Check for phase/round changes after human turn
                phase_change = await self.game_loop.check_phase_change(
                    player=current,
                    action=user_input,
                )
                if phase_change:
                    self.print(f"\n{self.game_loop.phase_tracker.format_phase_summary(phase_change)}")
                    self.print(f"\n{phase_change.confirmation_prompt}")
                    confirm = input("(y/n or describe corrections): ").strip().lower()
                    if confirm == "y" or confirm == "yes":
                        self.game_loop.apply_phase_effects(phase_change)
                        self.print("Phase effects applied.")
                    elif confirm == "n" or confirm == "no":
                        self.print("Phase effects skipped. Use /state to verify current state.")
                    else:
                        # User provided corrections - interpret and apply
                        self.print("Interpreting corrections...")
                        _, _ = await self.game_loop.interpret_and_process_human_move(
                            player="system",
                            description=f"Correction: {confirm}",
                        )
                        self.print("Corrections applied.")

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
