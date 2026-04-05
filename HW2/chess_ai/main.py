from __future__ import annotations

import argparse
import json
import random
import os

import chess

from .agents.alphabeta import AlphaBetaAgent
from .agents.mcts import MCTSAgent
from .comparison import compare_agents, summary_to_dict
from .config import AlphaBetaConfig, MCTSConfig, MatchConfig


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self.name = "Random"
        self._rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        moves = list(board.legal_moves)
        return self._rng.choice(moves) if moves else None


def _create_agent(kind: str, args: argparse.Namespace, seed_offset: int = 0):
    if kind == "alphabeta":
        return AlphaBetaAgent(
            config=AlphaBetaConfig(
                max_depth=args.ab_depth, quiescence_depth=args.ab_quiescence_depth
            )
        )
    if kind == "mcts":
        return MCTSAgent(
            config=MCTSConfig(
                iterations=args.mcts_iterations,
                exploration_constant=args.mcts_exploration,
                rollout_depth=args.mcts_rollout_depth,
            ),
            seed=(args.seed + seed_offset) if args.seed is not None else None,
        )
    if kind == "random":
        return RandomAgent(
            seed=(args.seed + seed_offset) if args.seed is not None else None
        )
    raise ValueError(f"Unknown agent type: {kind}")


def run_compare(args: argparse.Namespace) -> None:
    config = MatchConfig(
        max_plies=args.max_plies,
        random_opening_plies=args.random_opening_plies,
        seed=args.seed,
    )

    summary = compare_agents(
        games=args.games,
        alpha_beta_factory=lambda: AlphaBetaAgent(
            config=AlphaBetaConfig(
                max_depth=args.ab_depth, quiescence_depth=args.ab_quiescence_depth
            )
        ),
        mcts_factory=lambda: MCTSAgent(
            config=MCTSConfig(
                iterations=args.mcts_iterations,
                exploration_constant=args.mcts_exploration,
                rollout_depth=args.mcts_rollout_depth,
            ),
            seed=args.seed,
        ),
        config=config,
    )

    print("=== Comparison Summary ===")
    print(f"Games: {summary.games}")
    print(f"Alpha-Beta points: {summary.alpha_beta_points}")
    print(f"MCTS points: {summary.mcts_points}")
    print(f"Alpha-Beta wins: {summary.alpha_beta_wins}")
    print(f"MCTS wins: {summary.mcts_wins}")
    print(f"Draws: {summary.draws}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary_to_dict(summary), f, indent=2)
        print(f"Saved detailed report to: {args.output_json}")


def run_play(args: argparse.Namespace) -> None:
    board = chess.Board()
    white_agent = (
        None
        if args.white == "human"
        else _create_agent(args.white, args, seed_offset=1)
    )
    black_agent = (
        None
        if args.black == "human"
        else _create_agent(args.black, args, seed_offset=2)
    )

    while not board.is_game_over(claim_draw=True):
        print("\n" + str(board))
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

        is_human_turn = (board.turn == chess.WHITE and args.white == "human") or (
            board.turn == chess.BLACK and args.black == "human"
        )

        if is_human_turn:
            move = _get_human_move(board)
        else:
            agent = white_agent if board.turn == chess.WHITE else black_agent
            assert agent is not None
            move = agent.choose_move(board)
            if move is None:
                print("No legal move available for AI.")
                break
            print(f"AI move ({agent.name}): {move.uci()}")

        board.push(move)

    print("\nFinal board:")
    print(board)
    print(f"Result: {board.result(claim_draw=True)}")


def run_play_ui(args: argparse.Namespace) -> None:
    from .ui import launch_unified_ui

    print(
        "play-ui now launches the unified GUI. "
        "Configure mode/parameters from the app controls."
    )
    launch_unified_ui()


def run_compare_plots(args: argparse.Namespace) -> None:
    from .visualization import show_comparison_window, save_comparison_plot

    config = MatchConfig(
        max_plies=args.max_plies,
        random_opening_plies=args.random_opening_plies,
        seed=args.seed,
    )

    print(f"Running {args.games} games...")
    summary = compare_agents(
        games=args.games,
        alpha_beta_factory=lambda: AlphaBetaAgent(
            config=AlphaBetaConfig(
                max_depth=args.ab_depth, quiescence_depth=args.ab_quiescence_depth
            )
        ),
        mcts_factory=lambda: MCTSAgent(
            config=MCTSConfig(
                iterations=args.mcts_iterations,
                exploration_constant=args.mcts_exploration,
                rollout_depth=args.mcts_rollout_depth,
            ),
            seed=args.seed,
        ),
        config=config,
    )

    print("\n=== Comparison Summary ===")
    print(f"Games: {summary.games}")
    print(f"Alpha-Beta points: {summary.alpha_beta_points}")
    print(f"MCTS points: {summary.mcts_points}")
    print(f"Alpha-Beta wins: {summary.alpha_beta_wins}")
    print(f"MCTS wins: {summary.mcts_wins}")
    print(f"Draws: {summary.draws}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary_to_dict(summary), f, indent=2)
        print(f"Saved detailed report to: {args.output_json}")

    if args.output_plot:
        save_comparison_plot(summary, args.output_plot)
        print(f"Saved plot to: {args.output_plot}")
    else:
        print("\nDisplaying comparison visualization...")
        show_comparison_window(summary)


def _get_human_move(board: chess.Board) -> chess.Move:
    while True:
        raw = input("Enter your move (UCI, e.g., e2e4): ").strip()
        try:
            move = chess.Move.from_uci(raw)
        except ValueError:
            print("Invalid format. Try again.")
            continue

        if move in board.legal_moves:
            return move

        print("Illegal move. Try again.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HW2 Chess AI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compare command (text output)
    compare_parser = subparsers.add_parser(
        "compare", help="Run Alpha-Beta vs MCTS matches (text output)"
    )
    compare_parser.add_argument("--games", type=int, default=10)
    compare_parser.add_argument("--ab-depth", type=int, default=4)
    compare_parser.add_argument("--ab-quiescence-depth", type=int, default=3)
    compare_parser.add_argument("--mcts-iterations", type=int, default=1500)
    compare_parser.add_argument("--mcts-exploration", type=float, default=1.41421356237)
    compare_parser.add_argument("--mcts-rollout-depth", type=int, default=40)
    compare_parser.add_argument("--max-plies", type=int, default=300)
    compare_parser.add_argument("--random-opening-plies", type=int, default=2)
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument("--output-json", type=str, default="")
    compare_parser.set_defaults(func=run_compare)

    # Compare with plots command
    compare_plots_parser = subparsers.add_parser(
        "compare-plots", help="Run comparison with visualization plots"
    )
    compare_plots_parser.add_argument("--games", type=int, default=10)
    compare_plots_parser.add_argument("--ab-depth", type=int, default=4)
    compare_plots_parser.add_argument("--ab-quiescence-depth", type=int, default=3)
    compare_plots_parser.add_argument("--mcts-iterations", type=int, default=1500)
    compare_plots_parser.add_argument(
        "--mcts-exploration", type=float, default=1.41421356237
    )
    compare_plots_parser.add_argument("--mcts-rollout-depth", type=int, default=40)
    compare_plots_parser.add_argument("--max-plies", type=int, default=300)
    compare_plots_parser.add_argument("--random-opening-plies", type=int, default=2)
    compare_plots_parser.add_argument("--seed", type=int, default=42)
    compare_plots_parser.add_argument("--output-json", type=str, default="")
    compare_plots_parser.add_argument("--output-plot", type=str, default="")
    compare_plots_parser.set_defaults(func=run_compare_plots)

    # Play command (text-based)
    play_parser = subparsers.add_parser(
        "play", help="Play or watch a game (text-based)"
    )
    play_parser.add_argument(
        "--white", choices=["human", "alphabeta", "mcts", "random"], default="human"
    )
    play_parser.add_argument(
        "--black", choices=["human", "alphabeta", "mcts", "random"], default="alphabeta"
    )
    play_parser.add_argument("--ab-depth", type=int, default=4)
    play_parser.add_argument("--ab-quiescence-depth", type=int, default=3)
    play_parser.add_argument("--mcts-iterations", type=int, default=1500)
    play_parser.add_argument("--mcts-exploration", type=float, default=1.41421356237)
    play_parser.add_argument("--mcts-rollout-depth", type=int, default=40)
    play_parser.add_argument("--seed", type=int, default=42)
    play_parser.set_defaults(func=run_play)

    # Play UI command (graphical)
    play_ui_parser = subparsers.add_parser(
        "play-ui", help="Launch unified graphical UI (legacy-compatible command)"
    )
    play_ui_parser.add_argument(
        "--white", choices=["human", "alphabeta", "mcts", "random"], default="human"
    )
    play_ui_parser.add_argument(
        "--black", choices=["human", "alphabeta", "mcts", "random"], default="alphabeta"
    )
    play_ui_parser.add_argument("--ab-depth", type=int, default=4)
    play_ui_parser.add_argument("--ab-quiescence-depth", type=int, default=3)
    play_ui_parser.add_argument("--mcts-iterations", type=int, default=1500)
    play_ui_parser.add_argument("--mcts-exploration", type=float, default=1.41421356237)
    play_ui_parser.add_argument("--mcts-rollout-depth", type=int, default=40)
    play_ui_parser.add_argument("--seed", type=int, default=42)
    play_ui_parser.set_defaults(func=run_play_ui)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
