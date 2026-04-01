# Flexible Opponent

An LLM-powered board game opponent that can play **any** turn-based board game by reading the rulebook and observing the board state.

You play the physical game. The AI maintains a parallel digital state and tells you its moves. You execute them on the board and report the results back.

**Key principle:** The LLM never decides what is legal. It only decides what it *wants* to try. You are the final arbiter.

---

## How to Use It

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/flexible-opponent.git
cd flexible-opponent

# Install
pip install -e .

# Set your API key
export ANTHROPIC_API_KEY=your_key_here
```

### Starting a Game

```bash
opponent
```

The CLI walks you through setup:

1. **Game name** - e.g., "Agricola", "Catan", "Wingspan"
2. **Your name** - defaults to "human"
3. **Turn order** - who goes first
4. **Rules** - provide a rulebook (PDF, images, or paste text)
5. **Initial state** - optionally provide starting state as JSON

### Playing

When it's the AI's turn, it thinks and announces its move:

```
[ai's turn - AI thinking...]
--------------------------------------------------
AI plays: Take 3 Wood from the Forest
Reasoning: Building materials needed for room expansion

State changes:
  + ai.resources.wood: 0 â†’ 3
  - board.forest.wood: 5 â†’ 2
--------------------------------------------------
```

Execute the move on your physical board and report your own move:

```
[human's turn]
> I took the Clay Pit for 2 clay
Move recorded.
```

### Commands

| Command | Description |
|---------|-------------|
| `/state` | Show full current game state |
| `/diff` | Show what changed last turn |
| `/moves [n]` | Show last n moves (default 5) |
| `/strategy` | Show AI's current strategy |
| `/rules <query>` | Search rules (e.g., `/rules harvesting`) |
| `/recover` | Resync state from a photo |
| `/games` | List previous games for replay |
| `/endgame` | End game and write postmortem |
| `/quit` | Quit without postmortem |
| `/help` | Show all commands |

### State Corrections

If the digital state gets out of sync:

```
> Actually I have 5 wood not 4
```

Or take a photo and run `/recover` to resync from the board image.

### Replaying a Game

If you've played a game before, the rules index is saved. Starting a new session of the same game skips rulebook processing:

```bash
opponent
# Enter game name: Agricola
# [System detects existing rules index and offers to reuse it]
```

---

## How It Works

### Architecture Overview

```
  ┌─────────────────────────────────────────────────────────┐
  │                      SETUP PHASE                        │
  │  Rulebook → Chunk/Index → Schema → Board Photos → Deal  │
  └─────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │                      GAME LOOP                          │
  │                                                         │
  │   Rules Interpreter → Strategist → Referee → Apply      │
  │         │                 │           │                 │
  │    "What can I do?"  "What's best?" "Is it legal?"      │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

### LLM Role Separation

Each role gets a **separate API call** with different context. This prevents information leakage (e.g., the strategist can't see the opponent's hidden cards).

| Role | What it sees | What it does |
|------|--------------|--------------|
| **Referee** | Everything (all hidden info, full rules) | Validates legality, deals hidden cards |
| **Rules Interpreter** | Visible state + AI's hidden info + rules | Lists plausible legal moves with confidence |
| **Strategist** | Visible state + AI's hidden info + move list | Picks best move, updates strategy |

The Rules Interpreter sees the AI's hidden info so it can suggest moves like "play your Knight card" - but it never sees the opponent's hand.

### Turn Flow

1. **Rules Interpreter** queries the rules index and enumerates legal actions:
   ```json
   {
     "id": "take_wood",
     "description": "Take 3 Wood from Forest",
     "cost": {},
     "gains": {"wood": 3},
     "confidence": "high"
   }
   ```

2. **Strategist** picks the best move considering the current strategy:
   ```
   Chosen: take_wood
   Reasoning: Need materials for room before family growth
   Strategy status: continue
   ```

3. **Referee** validates the move is actually legal (catches hallucinations)

4. If invalid, the move is excluded and steps 2-3 retry (up to 3 times)

5. **State Manager** applies the move and logs it

### Strategy System

The AI maintains a hierarchical strategy:

**Core Strategy** (stable, changes rarely):
```json
{
  "long_term_plan": "growth_into_crops",
  "plan_confidence": 0.81,
  "pivot_conditions": [
    "if food_shortfall >= 3 for 2 rounds, shift to food_stabilization"
  ]
}
```

**Live Strategy** (updates each turn):
```json
{
  "current_priorities": ["family_growth", "food_security"],
  "blocked_goals": [{"goal": "family_growth", "reason": "space_taken"}],
  "turn_focus": "secure food and room materials",
  "strategy_status": "delay"
}
```

Each turn, the strategist classifies its situation:
- **Continue** - plan is working, keep going
- **Delay** - plan is good but current step blocked, take supporting move
- **Adapt** - plan broadly good, reorder near-term priorities
- **Pivot** - plan no longer viable, choose new direction

### Rules Indexing (RAG)

Rulebooks are processed into searchable chunks:

1. **Extract** text and images from PDF (or process images directly)
2. **Chunk** by section headers and semantic boundaries
3. **Index** into ChromaDB with vector embeddings
4. **Query** relevant chunks when interpreting rules

Board diagrams are extracted and analyzed by Claude Vision, stored as special "diagram" chunks.

### Session Files

Each game session creates a directory:

```
games/agricola_2024_03_31_143022/
- state.json           # Current visible game state
- hidden_ai.json       # AI's hand, bonuses, secrets
- strategy_core.json   # Long-term plan
- strategy_live.json   # Turn-by-turn adaptation
- move_log.jsonl       # Complete move history
- schema.json          # State structure definition
- rules_index/         # Chunked rulebook (ChromaDB)
- postmortem.md        # AI's notes for next time
```

### Photo-Based Recovery

If state gets out of sync:

1. Run `/recover`
2. Provide photo(s) of the current board
3. Claude Vision analyzes the photo against the schema
4. Proposed corrections shown with confidence score
5. Confirm to apply changes

### Postmortem Learning

At game end, the AI writes a postmortem analyzing:
- What strategy worked / didn't work
- Key turning points
- Lessons learned
- Opponent tendencies noticed

These notes are loaded into context when replaying the same game.

---

## Requirements

- Python 3.11+
- Anthropic API key (Claude)
- Dependencies: `anthropic`, `chromadb`, `pypdf`, `pydantic`

---

## Limitations

- **No true rules enforcement** - the AI proposes, you validate
- **State sync requires trust** - if you misreport, the AI's state diverges
- **No real-time board tracking** - you report moves manually
- **Complex games need patience** - setup may require iteration on the schema

---

## License

MIT
