from __future__ import annotations

from abc import ABC, abstractmethod

import chess


class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def choose_move(self, board: chess.Board) -> chess.Move | None:
        raise NotImplementedError
