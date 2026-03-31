from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import chess


@dataclass
class AgentSearchStats:
    move_time_ms: float = 0.0
    nodes_visited: int = 0


class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.last_search_stats = AgentSearchStats()

    @abstractmethod
    def choose_move(self, board: chess.Board) -> chess.Move | None:
        raise NotImplementedError
