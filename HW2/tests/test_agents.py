import unittest
import os
import sys

import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chess_ai.agents.alphabeta import AlphaBetaAgent
from chess_ai.agents.mcts import MCTSAgent
from chess_ai.comparison import compare_agents
from chess_ai.config import AlphaBetaConfig, MCTSConfig, MatchConfig


class TestAgents(unittest.TestCase):
    def test_alphabeta_returns_legal_move(self):
        board = chess.Board()
        agent = AlphaBetaAgent(config=AlphaBetaConfig(max_depth=2, quiescence_depth=1))
        move = agent.choose_move(board)
        self.assertIsNotNone(move)
        self.assertIn(move, board.legal_moves)

    def test_mcts_returns_legal_move(self):
        board = chess.Board()
        agent = MCTSAgent(config=MCTSConfig(iterations=80, rollout_depth=10), seed=1)
        move = agent.choose_move(board)
        self.assertIsNotNone(move)
        self.assertIn(move, board.legal_moves)

    def test_compare_agents_returns_consistent_scores(self):
        summary = compare_agents(
            games=2,
            alpha_beta_factory=lambda: AlphaBetaAgent(
                config=AlphaBetaConfig(max_depth=1, quiescence_depth=0)
            ),
            mcts_factory=lambda: MCTSAgent(
                config=MCTSConfig(iterations=40, rollout_depth=6), seed=2
            ),
            config=MatchConfig(max_plies=40, random_opening_plies=1, seed=10),
        )
        self.assertEqual(summary.games, 2)
        self.assertAlmostEqual(summary.alpha_beta_points + summary.mcts_points, 2.0)


if __name__ == "__main__":
    unittest.main()
