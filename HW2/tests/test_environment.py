import unittest
import os
import sys

import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chess_ai.environment import ChessEnvironment


class TestChessEnvironment(unittest.TestCase):
    def test_starting_position_has_20_moves(self):
        env = ChessEnvironment()
        self.assertEqual(len(env.legal_moves()), 20)

    def test_push_pop_consistency(self):
        env = ChessEnvironment()
        start_fen = env.to_fen()
        move = chess.Move.from_uci("e2e4")
        env.push(move)
        env.pop()
        self.assertEqual(env.to_fen(), start_fen)


if __name__ == "__main__":
    unittest.main()
