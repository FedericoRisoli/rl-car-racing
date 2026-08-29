"""
Genera i grafici finali del progetto PPO vs DQN su CarRacing-v2.

Il programma legge esclusivamente gli output gia' aggregati prodotti da
aggregate_results.py.

Input principali
----------------
- aggregated_results/learning_curve_algorithm.csv
- aggregated_results/test_per_run_summary.csv
- aggregated_results/test_algorithm_summary.csv
- aggregated_results/test_per_track_summary.csv

Output
------
- learning_curve.png
- final_test_reward_by_seed.png
- final_test_track_completion_by_seed.png
- final_test_reward_per_track.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


ALGORITHMS = ("dqn", "ppo")
DISPLAY_NAME = {"dqn": "DQN", "ppo": "PPO"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aggregated PPO/DQN CarRacing results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("aggregated_results"),
        help="Cartella prodotta da aggregate_results.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Cartella in cui salvare i grafici.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Risoluzione dei PNG prodotti.",
    )
    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi deve essere maggiore di zero.")

    return args


def read_csv(path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"File non trovato: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        available = set(reader.fieldnames or [])
        missing = sorted(set(required_columns) - available)

        if missing:
            raise ValueError(
                f"CSV {path} privo delle colonne richieste: {missing}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV vuoto: {path}")

    return rows


def as_int(value: Any) -> int:
    return int(float(value))


def as_float(value: Any) -> float:
    return float(value)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {path}")


def plot_learning_curve(input_dir: Path, output_dir: Path, dpi: int) -> None:
    """Learning curve media sui 4 training seed con fascia +/- 1 std."""

    path = input_dir / "learning_curve_algorithm.csv"
    rows = read_csv(
        path,
        {
            "algorithm",
            "timesteps",
            "mean_validation_reward_across_training_seeds",
            "std_validation_reward_between_training_seeds",
        },
    )

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        algorithm = row["algorithm"].strip().lower()
        if algorithm in ALGORITHMS:
            grouped[algorithm].append(row)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for algorithm in ALGORITHMS:
        algorithm_rows = sorted(
            grouped.get(algorithm, []),
            key=lambda row: as_int(row["timesteps"]),
        )

        if not algorithm_rows:
            raise ValueError(f"Nessun dato learning curve per {algorithm.upper()}.")

        timesteps = np.asarray(
            [as_int(row["timesteps"]) for row in algorithm_rows],
            dtype=np.int64,
        )
        means = np.asarray(
            [
                as_float(row["mean_validation_reward_across_training_seeds"])
                for row in algorithm_rows
            ],
            dtype=np.float64,
        )
        stds = np.asarray(
            [
                as_float(row["std_validation_reward_between_training_seeds"])
                for row in algorithm_rows
            ],
            dtype=np.float64,
        )

        line = ax.plot(
            timesteps,
            means,
            linewidth=2,
            label=DISPLAY_NAME[algorithm],
        )[0]
        ax.fill_between(
            timesteps,
            means - stds,
            means + stds,
            alpha=0.18,
            color=line.get_color(),
        )

    ax.set_title("Learning curve: PPO vs DQN")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Mean validation reward")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Algoritmo")

    save_figure(fig, output_dir / "learning_curve.png", dpi)


def load_test_summaries(
    input_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    per_run = read_csv(
        input_dir / "test_per_run_summary.csv",
        {
            "algorithm",
            "training_seed",
            "mean_reward",
            "mean_track_completion",
        },
    )
    algorithm_rows = read_csv(
        input_dir / "test_algorithm_summary.csv",
        {
            "algorithm",
            "mean_reward_across_training_seeds",
            "std_reward_between_training_seeds",
            "mean_track_completion_across_training_seeds",
            "std_track_completion_between_training_seeds",
        },
    )

    by_algorithm = {
        row["algorithm"].strip().lower(): row
        for row in algorithm_rows
        if row["algorithm"].strip().lower() in ALGORITHMS
    }

    return per_run, by_algorithm


def plot_final_reward_by_seed(input_dir: Path, output_dir: Path, dpi: int) -> None:
    """Confronta il reward medio finale dei quattro training seed."""

    per_run, algorithm_summary = load_test_summaries(input_dir)

    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in per_run:
        algorithm = row["algorithm"].strip().lower()
        if algorithm in ALGORITHMS:
            grouped[algorithm][as_int(row["training_seed"])] = as_float(
                row["mean_reward"]
            )

    seeds = sorted(
        set(grouped["dqn"]) | set(grouped["ppo"])
    )
    if not seeds:
        raise ValueError("Nessun training seed disponibile per il final test.")

    x = np.arange(len(seeds), dtype=np.float64)
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))

    dqn_values = [grouped["dqn"][seed] for seed in seeds]
    ppo_values = [grouped["ppo"][seed] for seed in seeds]

    dqn_mean = as_float(
        algorithm_summary["dqn"]["mean_reward_across_training_seeds"]
    )
    ppo_mean = as_float(
        algorithm_summary["ppo"]["mean_reward_across_training_seeds"]
    )

    ax.bar(
        x - width / 2,
        dqn_values,
        width,
        label=f"DQN (media = {dqn_mean:.1f})",
    )
    ax.bar(
        x + width / 2,
        ppo_values,
        width,
        label=f"PPO (media = {ppo_mean:.1f})",
    )

    ax.set_title("Final test: reward medio per training seed")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Mean test reward")
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Algoritmo")

    save_figure(fig, output_dir / "final_test_reward_by_seed.png", dpi)


def plot_final_track_completion_by_seed(
    input_dir: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    """Confronta la percentuale media di pista percorsa per training seed."""

    per_run, algorithm_summary = load_test_summaries(input_dir)

    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in per_run:
        algorithm = row["algorithm"].strip().lower()
        if algorithm in ALGORITHMS:
            grouped[algorithm][as_int(row["training_seed"])] = as_float(
                row["mean_track_completion"]
            )

    seeds = sorted(set(grouped["dqn"]) | set(grouped["ppo"]))
    if not seeds:
        raise ValueError("Nessun training seed disponibile per il final test.")

    x = np.arange(len(seeds), dtype=np.float64)
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))

    dqn_values = [grouped["dqn"][seed] for seed in seeds]
    ppo_values = [grouped["ppo"][seed] for seed in seeds]

    dqn_mean = as_float(
        algorithm_summary["dqn"]["mean_track_completion_across_training_seeds"]
    )
    ppo_mean = as_float(
        algorithm_summary["ppo"]["mean_track_completion_across_training_seeds"]
    )

    ax.bar(
        x - width / 2,
        dqn_values,
        width,
        label=f"DQN (media = {100.0 * dqn_mean:.1f}%)",
    )
    ax.bar(
        x + width / 2,
        ppo_values,
        width,
        label=f"PPO (media = {100.0 * ppo_mean:.1f}%)",
    )

    ax.set_title("Final test: track completion per training seed")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Mean track completion")
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Algoritmo")

    save_figure(
        fig,
        output_dir / "final_test_track_completion_by_seed.png",
        dpi,
    )


def plot_final_reward_per_track(input_dir: Path, output_dir: Path, dpi: int) -> None:
    """Confronto pista-per-pista del reward medio sui quattro training seed."""

    rows = read_csv(
        input_dir / "test_per_track_summary.csv",
        {
            "algorithm",
            "evaluation_seed",
            "mean_reward",
        },
    )

    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        algorithm = row["algorithm"].strip().lower()
        if algorithm in ALGORITHMS:
            grouped[algorithm].append(
                (
                    as_int(row["evaluation_seed"]),
                    as_float(row["mean_reward"]),
                )
            )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for algorithm in ALGORITHMS:
        values = sorted(grouped.get(algorithm, []))
        if not values:
            raise ValueError(f"Nessun risultato per-pista per {algorithm.upper()}.")

        seeds = [seed for seed, _ in values]
        rewards = [reward for _, reward in values]
        ax.plot(
            seeds,
            rewards,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            label=DISPLAY_NAME[algorithm],
        )

    ax.set_title("Final test: reward medio sulle 30 piste")
    ax.set_xlabel("Test seed")
    ax.set_ylabel("Mean reward across training seeds")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Algoritmo")

    save_figure(fig, output_dir / "final_test_reward_per_track.png", dpi)


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(
            f"Cartella aggregated_results non trovata: {args.input_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curve(args.input_dir, args.output_dir, args.dpi)
    plot_final_reward_by_seed(args.input_dir, args.output_dir, args.dpi)
    plot_final_track_completion_by_seed(args.input_dir, args.output_dir, args.dpi)
    plot_final_reward_per_track(args.input_dir, args.output_dir, args.dpi)

    print()
    print("Grafici generati correttamente.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()