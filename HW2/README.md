# HW2 - Chess AI

This project implements a full chess environment and two AI agents:

- Alpha-Beta pruning with move ordering and a robust heuristic evaluator
- Monte Carlo Tree Search (MCTS) with Selection, Expansion, Simulation, and Backpropagation

## Install

```bash
pip install -r requirements.txt
```

## Compare both agents

```bash
python -m chess_ai.main compare --games 20 --ab-depth 4 --mcts-iterations 1500
```

## Play a game

```bash
python -m chess_ai.main play --white human --black alphabeta --ab-depth 4
```

## Run tests

```bash
python -m unittest discover -s tests -v
```
