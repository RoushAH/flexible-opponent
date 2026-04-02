# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

LLM-powered board game opponent capable of playing **any** turn-based board game by reading the rulebook and observing board state. The human acts as final arbiter of legality. The system maintains a parallel digital state alongside the physical board.

Key principle: **The LLM never decides what is legal. It only decides what it wants to try.**

## Build & Run

```bash
# Install dependencies
pip install -e .

# Install dev dependencies  
pip install -e ".[dev]"

# Run the CLI (requires ANTHROPIC_API_KEY env var)
opponent

# Run tests
pytest

# Run a single test
pytest tests/test_foo.py::test_specific_function
pytest -k "test_name"  # match by name
```

## Architecture

### LLM Roles (Separate API Calls)

Each role gets isolated context to prevent information leakage:

| Role | Visible State | AI Hidden | Opponent Hidden | Rules |
|------|--------------|-----------|-----------------|-------|
| Referee | ✓ | ✓ | ✓ | ✓ |
| Rules Interpreter | ✓ | ✓ | ✗ | ✓ |
| AI Strategist | ✓ | ✓ | ✗ | (via moves) |

- **Referee**: Full knowledge, validates legality, deals hidden cards
- **Rules Interpreter**: Lists plausible legal moves with confidence levels
- **AI Strategist**: Picks best move, manages hierarchical strategy

### Code Patterns

**Async everywhere**: All LLM calls are async. Use `await` and ensure callers are async.

**Prompt templates**: Text files in `src/llm/prompts/`, loaded lazily via module-level functions like `_get_prompt_template()`.

**JSON response parsing**: LLM responses often come wrapped in markdown code fences. Use `LLMResponse.as_json()` which strips these automatically.

**Multi-provider support**: Four providers available via `create_client(provider=...)`:
- `anthropic`: Direct Anthropic API (requires `ANTHROPIC_API_KEY`)
- `bedrock`: AWS Bedrock (uses AWS credentials)
- `openai`: OpenAI API (requires `OPENAI_API_KEY`)
- `ollama`: Local models via Ollama (no API key needed)

Use `/model` in CLI to switch providers mid-game.

### Strategy System

Hierarchical with controlled adaptation:
- **strategy_core.json**: Stable long-term plan, pivot conditions
- **strategy_live.json**: Turn-by-turn priorities, blocked goals, fallbacks

Adaptation classifications: `continue` | `delay` | `adapt` | `pivot`

### Directory Structure

```
src/
├── controller/
│   ├── cli.py              # Interactive CLI with all commands
│   ├── game_loop.py        # Turn orchestration with referee validation
│   └── recovery.py         # Photo-based state recovery
├── engine/
│   ├── state_manager.py    # State files, diffing, move logging
│   ├── rules_index.py      # ChromaDB vector index for rules RAG
│   └── game_manager.py     # Session management, replay shortcuts
├── llm/
│   ├── client.py           # Abstracted LLM client (Claude, swappable)
│   ├── referee.py          # Full-knowledge move validation
│   ├── rules_interpreter.py # Lists legal moves (uses RAG)
│   ├── strategist.py       # Picks moves, manages strategy
│   └── prompts/            # Prompt templates
├── setup/
│   ├── text_extractor.py   # PDF (text+images), images, text extraction
│   ├── chunker.py          # Text → semantic chunks (incl. diagrams)
│   ├── rulebook_processor.py # Orchestrates extraction + chunking + indexing
│   ├── schema_generator.py # Rules → state schema proposal
│   ├── board_initializer.py # Photos → initial state via Claude vision
│   └── hidden_dealer.py    # Referee deals hidden cards per rules
└── strategy/
    └── postmortem.py       # End-of-game analysis and notes

games/                      # Per-game session data
```

## Game Session Files

Each session in `games/<game_name>_<timestamp>/`:
- `state.json` - Current visible game state
- `hidden_ai.json` - AI's private information (hand, bonuses)
- `strategy_core.json` - Stable long-term strategy
- `strategy_live.json` - Turn-by-turn adaptation
- `move_log.jsonl` - All moves, both players
- `schema.json` - State structure definition
- `rules_index/` - Chunked rulebook with ChromaDB vector embeddings
- `postmortem.md` - Post-game strategic notes for future sessions

## Implementation Status

### Complete
- **Core Loop**: CLI, turn orchestration, state management, move logging
- **LLM Roles**: Rules interpreter, AI strategist, referee (with validation retry)
- **Rulebook Processing**: PDF (text + image extraction), images, plain text → RAG chunks
- **Setup Automation**: Schema generator, board initializer (photos), hidden dealer
- **State Sync**: Diff reporting, `/state` command, photo-based recovery
- **Game Management**: Session history, replay shortcuts (reuse rules_index)
- **Postmortem**: AI writes strategic lessons after each game

### Future Enhancements
- Web/GUI interface
- Multi-game tournaments
- Opening book learning
- Difficulty adjustment

## CLI Commands

- `/state` - Show full visible game state
- `/diff` - Show what changed last turn
- `/moves [n]` - Show last n moves (default 5)
- `/strategy` - Show AI's current strategy
- `/rules <query>` - Search indexed rules (e.g., `/rules harvesting`)
- `/model` - Change LLM provider/model mid-game
- `/recover` - Resync state from a photo
- `/games` - List previous games (for replay)
- `/endgame` - End game and write postmortem
- `/quit` - Quit without postmortem
- `/help` - Show all commands

## Design Principles

1. **LLM proposes, human disposes** - AI suggests moves, human validates
2. **Separate contexts** - Each LLM role gets only the info it should see
3. **Structured state** - JSON, not prose; boring state is reliable state
4. **Bounded adaptation** - Strategy changes are classified and justified
5. **State sync** - Diff reported each turn, `/state` for full view, photo recovery if desynced
