from __future__ import annotations

import os

import matplotlib.pyplot as plt

from .models import GenerationMetrics


def save_fitness_plot(
    metrics: list[GenerationMetrics],
    output_dir: str,
    file_name: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, file_name)

    generations = [item.generation for item in metrics]
    best_fitness = [item.best_fitness for item in metrics]
    avg_fitness = [item.avg_fitness for item in metrics]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(generations, best_fitness, label="Best Fitness", linewidth=2.2, color="#0B7285")
    ax1.plot(generations, avg_fitness, label="Average Fitness", linewidth=1.8, color="#D9480F")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_title("GA Fitness Evolution")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    hard_violations = [item.best_hard_violations for item in metrics]
    ax2.plot(
        generations,
        hard_violations,
        label="Best Hard Violations",
        linestyle="--",
        linewidth=1.2,
        color="#5F3DC4",
        alpha=0.75,
    )
    ax2.set_ylabel("Hard Violations")

    lines_a, labels_a = ax1.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax1.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path
