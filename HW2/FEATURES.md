# HW2 Chess AI - Complete Feature Guide

## Overview

The Chess AI project now includes:

1. **Complete Chess Environment** - Legal move generation, board state management
2. **Two AI Agents**:
   - Alpha-Beta Pruning with move ordering, transposition table, and quiescence search
   - Monte Carlo Tree Search with UCB1 selection, rollout simulation
3. **Interactive Graphical UI** - Click-to-move chess board with move history and FEN display
4. **Comparison Analysis Tools** - Side-by-side performance metrics with interactive plots
5. **Command-Line Tools** - Flexible CLI for all operations

---

## Installation

```bash
cd HW2
pip install -r requirements.txt
```

---

## Features

### 1. Play with Graphical UI

**Interactive chess board with click-to-move interface, move history, and game status display.**

```bash
# Play against Alpha-Beta AI
python -m chess_ai.main play-ui --white human --black alphabeta --ab-depth 4

# Play against MCTS AI
python -m chess_ai.main play-ui --white human --black mcts --mcts-iterations 1500

# Watch two AIs play each other
python -m chess_ai.main play-ui --white alphabeta --black mcts --ab-depth 3 --mcts-iterations 800
```

**Features:**

- Visual chess board with proper piece symbols
- Click-to-move piece selection with highlighting
- Move history with algebraic notation (e.g., e4, Nf3)
- FEN display for debugging
- Game status (whose turn, game over conditions)
- Reset and Resign buttons

### 2. Compare Agents with Visualization

**Run multiple games between agents and see comprehensive analysis charts.**

```bash
# Display interactive plots (recommended for analysis)
python -m chess_ai.main compare-plots --games 10 \
  --ab-depth 4 --mcts-iterations 1500 \
  --output-json results.json

# Save plot to file for reports
python -m chess_ai.main compare-plots --games 20 \
  --ab-depth 4 --mcts-iterations 1500 \
  --output-plot comparison_analysis.png \
  --output-json detailed_results.json
```

**Visualization includes:**

- **Total Score Bar Chart** - Points earned (Alpha-Beta vs MCTS)
- **Outcome Distribution** - Wins, Draws, Losses breakdown
- **Cumulative Score Progression** - Score over all games
- **Game Length Distribution** - Histogram of game durations
- **Win Rate by Color** - Performance when playing as White vs Black
- **Statistics Table** - Comprehensive metrics summary

### 3. Text-Based Comparison (No UI)

```bash
python -m chess_ai.main compare --games 10 \
  --ab-depth 4 --mcts-iterations 1500 \
  --output-json results.json
```

Output: Text summary to console + optional JSON report

### 4. Text-Based Play

```bash
# Human vs AI (command-line input)
python -m chess_ai.main play --white human --black alphabeta --ab-depth 4

# AI vs AI (watch it unfold)
python -m chess_ai.main play --white alphabeta --black mcts
```

Input format: UCI notation (e.g., `e2e4` moves pawn from e2 to e4)

---

## Configuration Reference

### Alpha-Beta Agent

```
--ab-depth              Maximum search depth (default: 4)
                        Higher = stronger but slower
                        Recommended: 4-6

--ab-quiescence-depth   Quiescence search depth (default: 3)
                        Extends search in tactical positions
                        Recommended: 2-3
```

**Features:**

- Negamax with alpha-beta pruning
- Transposition table for position reuse
- Move ordering (PV, captures, checks, history heuristic)
- Quiescence search for tactical stability
- Piece-square tables for position evaluation
- Mobility bonus

### MCTS Agent

```
--mcts-iterations       MCTS simulations per move (default: 1500)
                        Higher = stronger but slower
                        Recommended: 1000-2000

--mcts-exploration      UCB1 exploration constant (default: 1.41421356237)
                        Balances exploration vs exploitation
                        Default is sqrt(2)

--mcts-rollout-depth    Maximum rollout depth (default: 40)
                        Balances simulation speed vs accuracy
```

**Features:**

- UCB1-based selection
- Tactical move preference during expansion (captures/checks)
- Smart rollout policy (captures > checks > random moves)
- Outcome-based rewards (+1 win, 0.5 draw, 0 loss)

### Match Settings

```
--games                 Number of games (default: 10)

--max-plies            Max half-moves per game (default: 300)
                        ~150 moves per side

--random-opening-plies Random moves at start (default: 2)
                        Varies starting positions

--seed                 Random seed for reproducibility (default: 42)
```

---

## Examples

### Example 1: Quick Test (5 seconds)

```bash
python -m chess_ai.main compare-plots --games 2 \
  --ab-depth 2 --mcts-iterations 100 --output-plot quick_test.png
```

### Example 2: Fair Comparison (1-2 minutes)

```bash
python -m chess_ai.main compare-plots --games 10 \
  --ab-depth 3 --mcts-iterations 800 \
  --output-json comparison_3v800.json \
  --output-plot comparison_3v800.png
```

### Example 3: Detailed Analysis (5-10 minutes)

```bash
python -m chess_ai.main compare-plots --games 20 \
  --ab-depth 4 --mcts-iterations 1500 \
  --output-json final_comparison.json \
  --output-plot final_comparison.png \
  --max-plies 400
```

### Example 4: Play vs Strong Opponent

```bash
python -m chess_ai.main play-ui --white human --black alphabeta --ab-depth 4
```

---

## Output Files

### JSON Report (`--output-json <file>`)

Contains:

- Game-by-game results (winner, plies, FEN)
- Cumulative statistics (wins, draws, points)
- Agent names and configurations

### PNG Plot (`--output-plot <file>`)

Multi-panel visualization:

- 6 analysis charts
- Statistics summary
- High-resolution (150 DPI) for reports

---

## Testing

Run the test suite to verify installation:

```bash
python -m unittest discover -s tests -v
```

Expected output: All 5 tests pass in ~5 seconds

---

## Performance Notes

- **Alpha-Beta Depth 2**: ~100ms per move
- **Alpha-Beta Depth 4**: ~1-2s per move
- **MCTS 1000 iterations**: ~500-1000ms per move
- **MCTS 1500 iterations**: ~1-2s per move

For real-time interactive play, recommend:

- Alpha-Beta: depth 3-4
- MCTS: 800-1500 iterations

---

## Architecture

```
chess_ai/
├── environment.py      # ChessBoard wrapper (python-chess)
├── evaluation.py       # Static position evaluator
├── agents/
│   ├── base.py         # Abstract agent interface
│   ├── alphabeta.py    # Alpha-Beta implementation
│   └── mcts.py         # MCTS implementation
├── ui.py               # Tkinter GUI (optional)
├── visualization.py    # Matplotlib charts
├── comparison.py       # Match runner and scoring
├── config.py           # Configuration dataclasses
├── main.py             # CLI entry point
└── tests/
    ├── test_environment.py
    └── test_agents.py
```

---

## Troubleshooting

**"No module named tkinter"** when running `play-ui`

- Tkinter is part of standard Python but may not be installed on Linux
- On Windows: Reinstall Python with "tcl/tk and IDLE" option
- On Linux: `sudo apt-get install python3-tk`
- The `compare-plots` command still works without tkinter

**"No module named matplotlib"**

- Run: `pip install matplotlib`

**Game seems to freeze**

- MCTS with many iterations can take time
- Try lower iteration count (e.g., 500 instead of 1500)
- Alpha-Beta with depth 4 is usually faster

**Plots look strange**

- Update matplotlib: `pip install --upgrade matplotlib`

---

## Advanced: Custom Agent

To implement your own agent, inherit from `BaseAgent`:

```python
from chess_ai.agents.base import BaseAgent
import chess

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="MyAgent")

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        moves = list(board.legal_moves)
        return moves[0] if moves else None

# Use in compare or play
from chess_ai.comparison import compare_agents
summary = compare_agents(
    games=5,
    alpha_beta_factory=...,
    mcts_factory=...,
    # Custom agents work the same way
)
```

---

## References

- **python-chess**: Official chess library
- **Alpha-Beta**: Classic minimax variant from game AI
- **MCTS**: Modern AI used in Go (AlphaGo) and chess engines
