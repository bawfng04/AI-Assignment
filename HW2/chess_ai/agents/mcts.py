from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time

import chess

from ..config import MCTSConfig
from ..evaluation import evaluate_board
from .base import BaseAgent


@dataclass
class MCTSNode:
    board: chess.Board
    root_player: chess.Color
    parent: "MCTSNode | None" = None
    move: chess.Move | None = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    untried_moves: list[chess.Move] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.untried_moves:
            self.untried_moves = list(self.board.legal_moves)

    def is_terminal(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def best_child(self, exploration_constant: float) -> "MCTSNode":
        assert self.children, "best_child called with no children"

        best_score = -float("inf")
        best_node = self.children[0]

        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploitation = child.value_sum / child.visits
                exploration = exploration_constant * math.sqrt(
                    math.log(self.visits) / child.visits
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_node = child

        return best_node

    def expand(self, rng: random.Random) -> "MCTSNode":
        move = self._pick_expansion_move(rng)
        new_board = self.board.copy(stack=False)
        new_board.push(move)

        child = MCTSNode(
            board=new_board,
            root_player=self.root_player,
            parent=self,
            move=move,
        )
        self.children.append(child)
        return child

    def _pick_expansion_move(self, rng: random.Random) -> chess.Move:
        # Prefer tactical moves during expansion for faster tree convergence.
        capture_moves = [m for m in self.untried_moves if self.board.is_capture(m)]
        check_moves = [m for m in self.untried_moves if self.board.gives_check(m)]

        if capture_moves:
            move = rng.choice(capture_moves)
        elif check_moves:
            move = rng.choice(check_moves)
        else:
            move = rng.choice(self.untried_moves)

        self.untried_moves.remove(move)
        return move


class MCTSAgent(BaseAgent):
    def __init__(self, config: MCTSConfig | None = None, seed: int | None = 42) -> None:
        super().__init__(name="MCTS")
        self.config = config or MCTSConfig()
        self.rng = random.Random(seed)
        self._nodes_created = 0

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        started_at = time.perf_counter()
        self._nodes_created = 0
        legal = list(board.legal_moves)
        if not legal:
            self.last_search_stats.move_time_ms = (
                time.perf_counter() - started_at
            ) * 1000.0
            self.last_search_stats.nodes_visited = 0
            return None

        root = MCTSNode(board=board.copy(stack=False), root_player=board.turn)
        self._nodes_created = 1

        for _ in range(self.config.iterations):
            node = self._select(root)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand(self.rng)
                self._nodes_created += 1

            reward = self._simulate(
                node.board.copy(stack=False), root_player=root.root_player
            )
            self._backpropagate(node, reward)

        best_child = max(root.children, key=lambda n: n.visits, default=None)
        self.last_search_stats.move_time_ms = (
            time.perf_counter() - started_at
        ) * 1000.0
        self.last_search_stats.nodes_visited = self._nodes_created
        return best_child.move if best_child else self.rng.choice(legal)

    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while (
            not current.is_terminal()
            and current.is_fully_expanded()
            and current.children
        ):
            current = current.best_child(self.config.exploration_constant)
        return current

    def _simulate(self, board: chess.Board, root_player: chess.Color) -> float:
        for _ in range(self.config.rollout_depth):
            if board.is_game_over(claim_draw=True):
                break

            move = self._rollout_policy(board)
            board.push(move)

        return self._reward(board, root_player)

    def _rollout_policy(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        capture_moves = [m for m in moves if board.is_capture(m)]
        if capture_moves and self.rng.random() < 0.7:
            return self.rng.choice(capture_moves)

        check_moves = [m for m in moves if board.gives_check(m)]
        if check_moves and self.rng.random() < 0.5:
            return self.rng.choice(check_moves)

        return self.rng.choice(moves)

    def _reward(self, board: chess.Board, root_player: chess.Color) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                return 0.5
            return 1.0 if outcome.winner == root_player else 0.0

        # Non-terminal rollout fallback using normalized static evaluation.
        score = evaluate_board(board)
        if board.turn != root_player:
            score = -score
        clipped = max(-1500, min(1500, score))
        return (clipped + 1500) / 3000

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.value_sum += reward
            current = current.parent
