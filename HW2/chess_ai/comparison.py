from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Callable

import chess

from .config import MatchConfig
from .agents.base import BaseAgent


@dataclass
class GameResult:
    game_index: int
    white_agent: str
    black_agent: str
    result: str
    winner: str
    plies: int
    final_fen: str


@dataclass
class ComparisonSummary:
    games: int
    alpha_beta_points: float
    mcts_points: float
    draws: int
    alpha_beta_wins: int
    mcts_wins: int
    game_results: list[GameResult]


def play_game(
    white_agent: BaseAgent,
    black_agent: BaseAgent,
    config: MatchConfig,
    rng: random.Random,
) -> GameResult:
    board = chess.Board()

    for _ in range(config.random_opening_plies):
        if board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(list(board.legal_moves)))

    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < config.max_plies:
        current = white_agent if board.turn == chess.WHITE else black_agent
        move = current.choose_move(board)
        if move is None or move not in board.legal_moves:
            winner = "black" if board.turn == chess.WHITE else "white"
            result = "0-1" if winner == "black" else "1-0"
            return GameResult(
                game_index=-1,
                white_agent=white_agent.name,
                black_agent=black_agent.name,
                result=result,
                winner=winner,
                plies=plies,
                final_fen=board.fen(),
            )

        board.push(move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        winner = "draw"
        result = "1/2-1/2"
    elif outcome.winner == chess.WHITE:
        winner = "white"
        result = "1-0"
    else:
        winner = "black"
        result = "0-1"

    return GameResult(
        game_index=-1,
        white_agent=white_agent.name,
        black_agent=black_agent.name,
        result=result,
        winner=winner,
        plies=plies,
        final_fen=board.fen(),
    )


def compare_agents(
    games: int,
    alpha_beta_factory: Callable[[], BaseAgent],
    mcts_factory: Callable[[], BaseAgent],
    config: MatchConfig | None = None,
) -> ComparisonSummary:
    config = config or MatchConfig()
    rng = random.Random(config.seed)

    alpha_points = 0.0
    mcts_points = 0.0
    draws = 0
    alpha_wins = 0
    mcts_wins = 0
    details: list[GameResult] = []

    for game_idx in range(games):
        alpha_agent = alpha_beta_factory()
        mcts_agent = mcts_factory()

        if game_idx % 2 == 0:
            white_agent, black_agent = alpha_agent, mcts_agent
            alpha_is_white = True
        else:
            white_agent, black_agent = mcts_agent, alpha_agent
            alpha_is_white = False

        result = play_game(white_agent, black_agent, config, rng)
        result.game_index = game_idx + 1
        details.append(result)

        if result.winner == "draw":
            draws += 1
            alpha_points += 0.5
            mcts_points += 0.5
        elif result.winner == "white":
            if alpha_is_white:
                alpha_wins += 1
                alpha_points += 1.0
            else:
                mcts_wins += 1
                mcts_points += 1.0
        elif result.winner == "black":
            if alpha_is_white:
                mcts_wins += 1
                mcts_points += 1.0
            else:
                alpha_wins += 1
                alpha_points += 1.0

    return ComparisonSummary(
        games=games,
        alpha_beta_points=alpha_points,
        mcts_points=mcts_points,
        draws=draws,
        alpha_beta_wins=alpha_wins,
        mcts_wins=mcts_wins,
        game_results=details,
    )


def summary_to_dict(summary: ComparisonSummary) -> dict:
    data = asdict(summary)
    data["game_results"] = [asdict(game) for game in summary.game_results]
    return data
