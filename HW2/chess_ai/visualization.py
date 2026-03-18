from __future__ import annotations

from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import numpy as np

from .comparison import ComparisonSummary, GameResult


def create_comparison_figure(summary: ComparisonSummary) -> Figure:
    """Generate a comprehensive comparison visualization."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Chess AI Comparison: {summary.games} Games", fontsize=16, fontweight="bold"
    )

    # 1. Win/Draw distribution (pie chart)
    ax1 = plt.subplot(2, 3, 1)
    ab_total = summary.alpha_beta_points
    mcts_total = summary.mcts_points
    ax1.bar(
        ["Alpha-Beta", "MCTS"],
        [ab_total, mcts_total],
        color=["#1f77b4", "#ff7f0e"],
        alpha=0.7,
    )
    ax1.set_ylabel("Points (out of {})".format(summary.games))
    ax1.set_title("Total Score")
    ax1.set_ylim(0, summary.games)
    for i, (label, val) in enumerate(
        zip(["Alpha-Beta", "MCTS"], [ab_total, mcts_total])
    ):
        ax1.text(i, val + 0.1, f"{val:.1f}", ha="center", fontweight="bold")

    # 2. Win breakdown
    ax2 = plt.subplot(2, 3, 2)
    outcomes = ["Wins", "Draws", "Losses"]
    ab_outcomes = [
        summary.alpha_beta_wins,
        summary.draws,
        summary.mcts_wins,
    ]
    mcts_outcomes = [
        summary.mcts_wins,
        summary.draws,
        summary.alpha_beta_wins,
    ]
    x = np.arange(len(outcomes))
    width = 0.35
    ax2.bar(
        x - width / 2,
        ab_outcomes,
        width,
        label="Alpha-Beta",
        color="#1f77b4",
        alpha=0.7,
    )
    ax2.bar(
        x + width / 2, mcts_outcomes, width, label="MCTS", color="#ff7f0e", alpha=0.7
    )
    ax2.set_ylabel("Count")
    ax2.set_title("Outcome Distribution")
    ax2.set_xticks(x)
    ax2.set_xticklabels(outcomes)
    ax2.legend()

    # 3. Points by game
    ax3 = plt.subplot(2, 3, 3)
    ab_points_cumulative = []
    mcts_points_cumulative = []
    ab_cum = 0.0
    mcts_cum = 0.0
    for result in summary.game_results:
        if result.winner == "draw":
            ab_cum += 0.5
            mcts_cum += 0.5
        elif (result.winner == "white" and result.white_agent == "AlphaBeta") or (
            result.winner == "black" and result.black_agent == "AlphaBeta"
        ):
            ab_cum += 1.0
        else:
            mcts_cum += 1.0

        ab_points_cumulative.append(ab_cum)
        mcts_points_cumulative.append(mcts_cum)

    games_x = range(1, len(summary.game_results) + 1)
    ax3.plot(
        games_x,
        ab_points_cumulative,
        marker="o",
        label="Alpha-Beta",
        color="#1f77b4",
        linewidth=2,
    )
    ax3.plot(
        games_x,
        mcts_points_cumulative,
        marker="s",
        label="MCTS",
        color="#ff7f0e",
        linewidth=2,
    )
    ax3.set_xlabel("Game")
    ax3.set_ylabel("Cumulative Points")
    ax3.set_title("Score Progression")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Game length distribution
    ax4 = plt.subplot(2, 3, 4)
    plies = [result.plies for result in summary.game_results]
    ax4.hist(plies, bins=10, color="#2ca02c", alpha=0.7, edgecolor="black")
    ax4.set_xlabel("Plies (half-moves)")
    ax4.set_ylabel("Frequency")
    ax4.set_title(f"Game Length Distribution (avg: {np.mean(plies):.1f})")

    # 5. Win rate by color
    ax5 = plt.subplot(2, 3, 5)
    ab_white_wins = sum(
        1
        for r in summary.game_results
        if r.white_agent == "AlphaBeta" and r.winner == "white"
    )
    ab_white_total = sum(
        1 for r in summary.game_results if r.white_agent == "AlphaBeta"
    )
    ab_black_wins = sum(
        1
        for r in summary.game_results
        if r.black_agent == "AlphaBeta" and r.winner == "black"
    )
    ab_black_total = sum(
        1 for r in summary.game_results if r.black_agent == "AlphaBeta"
    )

    mcts_white_wins = sum(
        1
        for r in summary.game_results
        if r.white_agent == "MCTS" and r.winner == "white"
    )
    mcts_white_total = sum(1 for r in summary.game_results if r.white_agent == "MCTS")
    mcts_black_wins = sum(
        1
        for r in summary.game_results
        if r.black_agent == "MCTS" and r.winner == "black"
    )
    mcts_black_total = sum(1 for r in summary.game_results if r.black_agent == "MCTS")

    ab_white_rate = ab_white_wins / ab_white_total if ab_white_total > 0 else 0
    ab_black_rate = ab_black_wins / ab_black_total if ab_black_total > 0 else 0
    mcts_white_rate = mcts_white_wins / mcts_white_total if mcts_white_total > 0 else 0
    mcts_black_rate = mcts_black_wins / mcts_black_total if mcts_black_total > 0 else 0

    x = np.arange(2)
    width = 0.35
    ax5.bar(
        x - width / 2,
        [ab_white_rate, ab_black_rate],
        width,
        label="Alpha-Beta",
        color="#1f77b4",
        alpha=0.7,
    )
    ax5.bar(
        x + width / 2,
        [mcts_white_rate, mcts_black_rate],
        width,
        label="MCTS",
        color="#ff7f0e",
        alpha=0.7,
    )
    ax5.set_ylabel("Win Rate")
    ax5.set_title("Win Rate by Color")
    ax5.set_xticks(x)
    ax5.set_xticklabels(["White", "Black"])
    ax5.set_ylim(0, 1)
    ax5.legend()

    # 6. Statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    stats_text = f"""
Alpha-Beta vs MCTS Comparison

Total Points: {ab_total:.1f} - {mcts_total:.1f}
Wins: {summary.alpha_beta_wins} - {summary.mcts_wins}
Draws: {summary.draws}

Average Game Length: {np.mean(plies):.1f} plies
Min Game Length: {min(plies)} plies
Max Game Length: {max(plies)} plies

Win Rate (White): {ab_white_rate:.1%} - {mcts_white_rate:.1%}
Win Rate (Black): {ab_black_rate:.1%} - {mcts_black_rate:.1%}

Total Games: {summary.games}
"""

    ax6.text(
        0.1,
        0.5,
        stats_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


def show_comparison_window(summary: ComparisonSummary) -> None:
    """Display comparison visualization in a new window."""
    fig = create_comparison_figure(summary)
    plt.show()


def save_comparison_plot(summary: ComparisonSummary, filepath: str) -> None:
    """Save comparison visualization to file."""
    fig = create_comparison_figure(summary)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
