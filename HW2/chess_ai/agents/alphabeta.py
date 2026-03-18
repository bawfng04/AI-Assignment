from __future__ import annotations

from dataclasses import dataclass
import time

import chess

from ..config import AlphaBetaConfig
from ..evaluation import MATE_SCORE, evaluate_board, terminal_score
from .base import BaseAgent


class SearchTimeout(Exception):
    pass


@dataclass
class _TTEntry:
    depth: int
    score: int
    best_move: chess.Move | None


class AlphaBetaAgent(BaseAgent):
    def __init__(
        self, config: AlphaBetaConfig | None = None, time_limit_sec: float | None = None
    ) -> None:
        super().__init__(name="AlphaBeta")
        self.config = config or AlphaBetaConfig()
        self.time_limit_sec = time_limit_sec
        self._deadline: float | None = None
        self._tt: dict[str, _TTEntry] = {}
        self._history_heuristic: dict[chess.Move, int] = {}

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        self._deadline = (
            time.perf_counter() + self.time_limit_sec if self.time_limit_sec else None
        )
        best_move = legal_moves[0]

        try:
            for depth in range(1, self.config.max_depth + 1):
                score, candidate = self._search_root(board, depth)
                if candidate is not None:
                    best_move = candidate
                if abs(score) >= MATE_SCORE - 500:
                    break
        except SearchTimeout:
            pass

        return best_move

    def _search_root(
        self, board: chess.Board, depth: int
    ) -> tuple[int, chess.Move | None]:
        best_score = -(10**9)
        best_move: chess.Move | None = None
        alpha = -(10**9)
        beta = 10**9

        tt_best = self._tt.get(board.fen())
        ordered_moves = self._order_moves(
            board, list(board.legal_moves), tt_best.best_move if tt_best else None
        )

        for move in ordered_moves:
            self._check_timeout()
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

        return best_score, best_move

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int) -> int:
        self._check_timeout()

        term = terminal_score(board)
        if term is not None:
            return term

        if depth == 0:
            return self._quiescence(board, alpha, beta, self.config.quiescence_depth)

        key = board.fen()
        if self.config.use_transposition_table:
            entry = self._tt.get(key)
            if entry and entry.depth >= depth:
                return entry.score

        best_score = -(10**9)
        best_move: chess.Move | None = None
        tt_best = self._tt.get(key)
        ordered_moves = self._order_moves(
            board, list(board.legal_moves), tt_best.best_move if tt_best else None
        )

        for move in ordered_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                self._history_heuristic[move] = (
                    self._history_heuristic.get(move, 0) + depth * depth
                )
                break

        if self.config.use_transposition_table:
            self._tt[key] = _TTEntry(
                depth=depth,
                score=best_score,
                best_move=best_move,
            )

        return best_score

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, depth: int) -> int:
        self._check_timeout()

        stand_pat = evaluate_board(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        if depth <= 0:
            return stand_pat

        capture_moves = [m for m in board.legal_moves if board.is_capture(m)]
        ordered = self._order_moves(board, capture_moves)

        for move in ordered:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, depth - 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        pv_move: chess.Move | None = None,
    ) -> list[chess.Move]:
        if not self.config.move_ordering:
            return moves

        def move_score(move: chess.Move) -> int:
            score = 0
            if pv_move is not None and move == pv_move:
                score += 100000

            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += 10000 + 10 * victim.piece_type - attacker.piece_type
                else:
                    score += 10000

            if move.promotion:
                score += 8000 + move.promotion

            if board.gives_check(move):
                score += 2000

            score += self._history_heuristic.get(move, 0)
            return score

        return sorted(moves, key=move_score, reverse=True)

    def _check_timeout(self) -> None:
        if self._deadline and time.perf_counter() > self._deadline:
            raise SearchTimeout
