from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import chess


@dataclass
class ChessEnvironment:
    """Thin wrapper over python-chess to provide a clean environment API."""

    board: chess.Board

    def __init__(self, fen: str | None = None) -> None:
        self.board = chess.Board(fen) if fen else chess.Board()

    def reset(self, fen: str | None = None) -> None:
        self.board = chess.Board(fen) if fen else chess.Board()

    def copy(self) -> "ChessEnvironment":
        clone = ChessEnvironment()
        clone.board = self.board.copy(stack=True)
        return clone

    @property
    def turn(self) -> chess.Color:
        return self.board.turn

    def legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    def legal_moves_uci(self) -> list[str]:
        return [m.uci() for m in self.board.legal_moves]

    def push(self, move: chess.Move) -> None:
        self.board.push(move)

    def pop(self) -> chess.Move:
        return self.board.pop()

    def parse_uci(self, uci: str) -> chess.Move:
        return chess.Move.from_uci(uci)

    def is_legal(self, move: chess.Move) -> bool:
        return move in self.board.legal_moves

    def to_fen(self) -> str:
        return self.board.fen()

    def set_fen(self, fen: str) -> None:
        self.board.set_fen(fen)

    def is_game_over(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def result(self) -> str:
        return self.board.result(claim_draw=True)

    def outcome(self) -> chess.Outcome | None:
        return self.board.outcome(claim_draw=True)

    def push_many(self, moves: Iterable[chess.Move]) -> None:
        for move in moves:
            self.push(move)
