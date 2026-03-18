# HW2 - Chess AI

This project implements a full chess environment and two AI agents with a complete graphical UI and analysis tools:

- **Alpha-Beta Pruning** with move ordering and a robust heuristic evaluator
- **Monte Carlo Tree Search (MCTS)** with Selection, Expansion, Simulation, and Backpropagation
- **Interactive GUI** for playing games with visual board and move history
- **Comparison Analysis** with charts, statistics, and performance visualization

## Install

```bash
pip install -r requirements.txt
```

## Play with Graphical UI

```bash
# Play against Alpha-Beta AI
python -m chess_ai.main play-ui --white human --black alphabeta --ab-depth 4

# Play against MCTS AI
python -m chess_ai.main play-ui --white human --black mcts --mcts-iterations 1500

# Watch AI vs AI match
python -m chess_ai.main play-ui --white alphabeta --black mcts --ab-depth 3 --mcts-iterations 800
```

## Compare both agents with Plots

```bash
# Run comparison and display interactive plots
python -m chess_ai.main compare-plots --games 5 --ab-depth 3 --mcts-iterations 300

# Save comparison visualization to file
python -m chess_ai.main compare-plots --games 10 --ab-depth 4 --mcts-iterations 1500 --output-plot comparison.png
```

## Command-Line Play (Text-Based)

```bash
python -m chess_ai.main play --white human --black alphabeta --ab-depth 4
```

## Compare without UI (Text Output Only)

```bash
python -m chess_ai.main compare --games 5 --ab-depth 3 --mcts-iterations 300

# Save detailed results to JSON
python -m chess_ai.main compare --games 5 --ab-depth 3 --mcts-iterations 300 --output-json results.json
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Configuration

### Alpha-Beta Agent
- `--ab-depth`: Maximum search depth (default: 4)
- `--ab-quiescence-depth`: Quiescence search depth (default: 3)

### MCTS Agent
- `--mcts-iterations`: Number of MCTS simulations per move (default: 1500)
- `--mcts-exploration`: UCB1 exploration constant (default: 1.41421356237)
- `--mcts-rollout-depth`: Maximum rollout depth (default: 40)

### Match Settings
- `--games`: Number of games to play (default: 10)
- `--max-plies`: Maximum half-moves per game (default: 300)
- `--random-opening-plies`: Random opening moves (default: 2)
- `--seed`: Random seed for reproducibility (default: 42)

