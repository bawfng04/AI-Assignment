from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import os
import random
from typing import Callable

import chess

from .agents.alphabeta import AlphaBetaAgent
from .config import MatchConfig
from .config import AlphaBetaConfig, MCTSConfig
from .agents.base import BaseAgent
from .agents.mcts import MCTSAgent


@dataclass
class GameResult:
    game_index: int
    white_agent: str
    black_agent: str
    result: str
    winner: str
    plies: int
    final_fen: str
    white_avg_move_time_ms: float = 0.0
    black_avg_move_time_ms: float = 0.0
    white_avg_nodes_visited: float = 0.0
    black_avg_nodes_visited: float = 0.0
    white_move_count: int = 0
    black_move_count: int = 0


@dataclass
class ComparisonSummary:
    games: int
    alpha_beta_points: float
    mcts_points: float
    draws: int
    alpha_beta_wins: int
    mcts_wins: int
    alpha_beta_win_rate: float
    mcts_win_rate: float
    alpha_beta_avg_move_time_ms: float
    mcts_avg_move_time_ms: float
    alpha_beta_avg_nodes_visited: float
    mcts_avg_nodes_visited: float
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
    white_time_ms_total = 0.0
    black_time_ms_total = 0.0
    white_nodes_total = 0
    black_nodes_total = 0
    white_moves = 0
    black_moves = 0

    while not board.is_game_over(claim_draw=True) and plies < config.max_plies:
        current = white_agent if board.turn == chess.WHITE else black_agent
        move = current.choose_move(board)
        stats = getattr(current, "last_search_stats", None)
        move_time_ms = getattr(stats, "move_time_ms", 0.0)
        nodes_visited = getattr(stats, "nodes_visited", 0)

        if board.turn == chess.WHITE:
            white_time_ms_total += move_time_ms
            white_nodes_total += int(nodes_visited)
            white_moves += 1
        else:
            black_time_ms_total += move_time_ms
            black_nodes_total += int(nodes_visited)
            black_moves += 1

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
                white_avg_move_time_ms=(
                    white_time_ms_total / white_moves if white_moves else 0.0
                ),
                black_avg_move_time_ms=(
                    black_time_ms_total / black_moves if black_moves else 0.0
                ),
                white_avg_nodes_visited=(
                    white_nodes_total / white_moves if white_moves else 0.0
                ),
                black_avg_nodes_visited=(
                    black_nodes_total / black_moves if black_moves else 0.0
                ),
                white_move_count=white_moves,
                black_move_count=black_moves,
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
        white_avg_move_time_ms=(white_time_ms_total / white_moves if white_moves else 0.0),
        black_avg_move_time_ms=(black_time_ms_total / black_moves if black_moves else 0.0),
        white_avg_nodes_visited=(white_nodes_total / white_moves if white_moves else 0.0),
        black_avg_nodes_visited=(black_nodes_total / black_moves if black_moves else 0.0),
        white_move_count=white_moves,
        black_move_count=black_moves,
    )


def compare_agents(
    games: int,
    alpha_beta_factory: Callable[[], BaseAgent],
    mcts_factory: Callable[[], BaseAgent],
    config: MatchConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ComparisonSummary:
    config = config or MatchConfig()
    rng = random.Random(config.seed)

    alpha_points = 0.0
    mcts_points = 0.0
    draws = 0
    alpha_wins = 0
    mcts_wins = 0
    alpha_time_ms_total = 0.0
    mcts_time_ms_total = 0.0
    alpha_nodes_total = 0.0
    mcts_nodes_total = 0.0
    alpha_moves_total = 0
    mcts_moves_total = 0
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

        if result.white_agent == "AlphaBeta":
            alpha_time_ms_total += result.white_avg_move_time_ms * result.white_move_count
            alpha_nodes_total += result.white_avg_nodes_visited * result.white_move_count
            alpha_moves_total += result.white_move_count
        if result.black_agent == "AlphaBeta":
            alpha_time_ms_total += result.black_avg_move_time_ms * result.black_move_count
            alpha_nodes_total += result.black_avg_nodes_visited * result.black_move_count
            alpha_moves_total += result.black_move_count
        if result.white_agent == "MCTS":
            mcts_time_ms_total += result.white_avg_move_time_ms * result.white_move_count
            mcts_nodes_total += result.white_avg_nodes_visited * result.white_move_count
            mcts_moves_total += result.white_move_count
        if result.black_agent == "MCTS":
            mcts_time_ms_total += result.black_avg_move_time_ms * result.black_move_count
            mcts_nodes_total += result.black_avg_nodes_visited * result.black_move_count
            mcts_moves_total += result.black_move_count

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

        if progress_callback:
            progress_callback(game_idx + 1, games)

    return ComparisonSummary(
        games=games,
        alpha_beta_points=alpha_points,
        mcts_points=mcts_points,
        draws=draws,
        alpha_beta_wins=alpha_wins,
        mcts_wins=mcts_wins,
        alpha_beta_win_rate=(alpha_wins / games if games else 0.0),
        mcts_win_rate=(mcts_wins / games if games else 0.0),
        alpha_beta_avg_move_time_ms=(
            alpha_time_ms_total / alpha_moves_total if alpha_moves_total else 0.0
        ),
        mcts_avg_move_time_ms=(
            mcts_time_ms_total / mcts_moves_total if mcts_moves_total else 0.0
        ),
        alpha_beta_avg_nodes_visited=(
            alpha_nodes_total / alpha_moves_total if alpha_moves_total else 0.0
        ),
        mcts_avg_nodes_visited=(
            mcts_nodes_total / mcts_moves_total if mcts_moves_total else 0.0
        ),
        game_results=details,
    )


def _play_indexed_game_from_configs(
    game_idx: int,
    alpha_beta_config: AlphaBetaConfig,
    mcts_config: MCTSConfig,
    match_config: MatchConfig,
) -> GameResult:
    seed_base = match_config.seed or 0
    rng = random.Random(seed_base + 1_000_003 * game_idx + 17)

    alpha_agent = AlphaBetaAgent(config=alpha_beta_config)
    mcts_agent = MCTSAgent(
        config=mcts_config,
        seed=seed_base + 2_000_003 * game_idx + 31,
    )

    if game_idx % 2 == 0:
        white_agent, black_agent = alpha_agent, mcts_agent
    else:
        white_agent, black_agent = mcts_agent, alpha_agent

    result = play_game(white_agent, black_agent, match_config, rng)
    result.game_index = game_idx + 1
    return result


def compare_agents_parallel(
    games: int,
    alpha_beta_config: AlphaBetaConfig,
    mcts_config: MCTSConfig,
    config: MatchConfig | None = None,
    workers: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ComparisonSummary:
    config = config or MatchConfig()
    if games <= 0:
        return ComparisonSummary(
            games=0,
            alpha_beta_points=0.0,
            mcts_points=0.0,
            draws=0,
            alpha_beta_wins=0,
            mcts_wins=0,
            alpha_beta_win_rate=0.0,
            mcts_win_rate=0.0,
            alpha_beta_avg_move_time_ms=0.0,
            mcts_avg_move_time_ms=0.0,
            alpha_beta_avg_nodes_visited=0.0,
            mcts_avg_nodes_visited=0.0,
            game_results=[],
        )

    max_workers = max(1, os.cpu_count() or 1)
    chosen_workers = workers if workers is not None else max_workers
    chosen_workers = max(1, min(chosen_workers, games, max_workers))

    if chosen_workers == 1:
        return compare_agents(
            games=games,
            alpha_beta_factory=lambda: AlphaBetaAgent(config=alpha_beta_config),
            mcts_factory=lambda: MCTSAgent(config=mcts_config, seed=config.seed),
            config=config,
            progress_callback=progress_callback,
        )

    details: list[GameResult] = []
    with ProcessPoolExecutor(max_workers=chosen_workers) as executor:
        futures = [
            executor.submit(
                _play_indexed_game_from_configs,
                game_idx,
                alpha_beta_config,
                mcts_config,
                config,
            )
            for game_idx in range(games)
        ]

        completed = 0
        for future in as_completed(futures):
            details.append(future.result())
            completed += 1
            if progress_callback:
                progress_callback(completed, games)

    details.sort(key=lambda r: r.game_index)

    alpha_points = 0.0
    mcts_points = 0.0
    draws = 0
    alpha_wins = 0
    mcts_wins = 0
    alpha_time_ms_total = 0.0
    mcts_time_ms_total = 0.0
    alpha_nodes_total = 0.0
    mcts_nodes_total = 0.0
    alpha_moves_total = 0
    mcts_moves_total = 0

    for result in details:
        if result.white_agent == "AlphaBeta":
            alpha_time_ms_total += result.white_avg_move_time_ms * result.white_move_count
            alpha_nodes_total += result.white_avg_nodes_visited * result.white_move_count
            alpha_moves_total += result.white_move_count
        if result.black_agent == "AlphaBeta":
            alpha_time_ms_total += result.black_avg_move_time_ms * result.black_move_count
            alpha_nodes_total += result.black_avg_nodes_visited * result.black_move_count
            alpha_moves_total += result.black_move_count
        if result.white_agent == "MCTS":
            mcts_time_ms_total += result.white_avg_move_time_ms * result.white_move_count
            mcts_nodes_total += result.white_avg_nodes_visited * result.white_move_count
            mcts_moves_total += result.white_move_count
        if result.black_agent == "MCTS":
            mcts_time_ms_total += result.black_avg_move_time_ms * result.black_move_count
            mcts_nodes_total += result.black_avg_nodes_visited * result.black_move_count
            mcts_moves_total += result.black_move_count

        alpha_is_white = (result.game_index - 1) % 2 == 0
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
        alpha_beta_win_rate=(alpha_wins / games if games else 0.0),
        mcts_win_rate=(mcts_wins / games if games else 0.0),
        alpha_beta_avg_move_time_ms=(
            alpha_time_ms_total / alpha_moves_total if alpha_moves_total else 0.0
        ),
        mcts_avg_move_time_ms=(
            mcts_time_ms_total / mcts_moves_total if mcts_moves_total else 0.0
        ),
        alpha_beta_avg_nodes_visited=(
            alpha_nodes_total / alpha_moves_total if alpha_moves_total else 0.0
        ),
        mcts_avg_nodes_visited=(
            mcts_nodes_total / mcts_moves_total if mcts_moves_total else 0.0
        ),
        game_results=details,
    )


def summary_to_dict(summary: ComparisonSummary) -> dict:
    data = asdict(summary)
    data["game_results"] = [asdict(game) for game in summary.game_results]
    return data
