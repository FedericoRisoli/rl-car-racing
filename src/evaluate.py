import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN, PPO

from env import make_env


ALGORITHMS = {
    "ppo": PPO,
    "dqn": DQN,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO/DQN agent on fixed CarRacing tracks."
    )

    parser.add_argument(
        "--algo",
        choices=["ppo", "dqn"],
        required=True,
        help="Algorithm used by the saved model.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to the saved model. If omitted, uses "
            "models/<algo>_smoke_seed_<train-seed>.zip."
        ),
    )

    parser.add_argument(
        "--train-seed",
        type=int,
        default=0,
        help="Seed used to train the model (used for naming/metadata).",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes when --eval-seeds is not supplied.",
    )

    parser.add_argument(
        "--eval-seed-start",
        type=int,
        default=100,
        help="First evaluation seed. Seeds are consecutive from this value.",
    )

    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit evaluation seeds; overrides --episodes/--eval-seed-start.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory in which CSV files are saved.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="SB3 device used for inference (default: cpu).",
    )

    visual = parser.add_mutually_exclusive_group()

    visual.add_argument(
        "--render",
        action="store_true",
        help="Render the evaluation live in a window.",
    )

    visual.add_argument(
        "--record-video",
        action="store_true",
        help="Record the first evaluation episodes as MP4 files.",
    )

    parser.add_argument(
        "--video-episodes",
        type=int,
        default=1,
        help="How many evaluation episodes to record when --record-video is used.",
    )

    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos",
        help="Directory in which evaluation videos are saved.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic evaluation.",
    )

    return parser.parse_args()


def resolve_model_path(args) -> Path:
    """
    Finds the model to evaluate.

    If --model is not provided, it uses the naming convention
    currently used by train.py.
    """
    if args.model is None:
        candidate = (
            Path("models")
            / f"{args.algo}_smoke_seed_{args.train_seed}.zip"
        )
    else:
        candidate = Path(args.model)

    if candidate.exists():
        return candidate

    # Allow the user to specify a path without ".zip"
    if candidate.suffix != ".zip":
        zipped = candidate.with_suffix(".zip")

        if zipped.exists():
            return zipped

    raise FileNotFoundError(
        f"Model not found: {candidate}"
    )


def infer_training_seed(
    model_path: Path,
    fallback: int,
) -> int:
    """
    Tries to extract the training seed from the model filename.

    Example:
        ppo_smoke_seed_2.zip -> 2
    """
    match = re.search(
        r"seed[_-]?(\d+)",
        model_path.stem,
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return fallback


def make_evaluation_env(
    args,
    run_name: str,
):
    """
    Creates exactly the same CarRacing environment used for training.

    Optional rendering/video recording is added only for evaluation.
    """
    if args.record_video:
        render_mode = "rgb_array"

    elif args.render:
        render_mode = "human"

    else:
        render_mode = None

    env = make_env(
        render_mode=render_mode
    )

    if args.record_video:
        from gymnasium.wrappers import RecordVideo

        video_dir = (
            Path(args.video_dir)
            / run_name
        )

        video_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        env = RecordVideo(
            env,
            video_folder=str(video_dir),

            # Record only the first N evaluation episodes,
            # otherwise 20-50 evaluation videos would be excessive.
            episode_trigger=lambda episode_id:
                episode_id < args.video_episodes,

            name_prefix=run_name,
        )

    return env


def get_track_stats(env) -> dict[str, Any]:
    """
    Reads CarRacing-specific information directly from
    the unwrapped Gymnasium environment.
    """
    base_env = env.unwrapped

    visited_tiles = int(
        getattr(
            base_env,
            "tile_visited_count",
            0,
        )
    )

    track = getattr(
        base_env,
        "track",
        None,
    )

    total_tiles = (
        len(track)
        if track is not None
        else 0
    )

    if total_tiles > 0:
        completion_fraction = (
            visited_tiles
            / total_tiles
        )
    else:
        completion_fraction = 0.0

    lap_complete_percent = float(
        getattr(
            base_env,
            "lap_complete_percent",
            1.0,
        )
    )

    new_lap = bool(
        getattr(
            base_env,
            "new_lap",
            False,
        )
    )

    # Important:
    # We do NOT simply use terminated=True as "completed".
    #
    # CarRacing can end because:
    # - the lap was completed
    # - the car left the playfield
    # - the TimeLimit was reached
    #
    # Therefore completion is determined using
    # CarRacing's internal track information.
    completed = (
        new_lap
        or (
            total_tiles > 0
            and visited_tiles == total_tiles
        )
    )

    return {
        "visited_tiles": visited_tiles,
        "total_tiles": total_tiles,
        "track_completion": completion_fraction,
        "lap_complete_percent": lap_complete_percent,
        "completed": completed,
    }


def termination_reason(
    completed: bool,
    terminated: bool,
    truncated: bool,
) -> str:
    """
    Gives a human-readable reason for the episode ending.
    """
    if completed:
        return "lap_completed"

    if terminated:
        return "out_of_bounds"

    if truncated:
        return "time_limit"

    return "unknown"


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """
    Saves a list of dictionaries as CSV.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Computes aggregate statistics over all evaluation tracks.
    """
    rewards = np.asarray(
        [
            row["episode_reward"]
            for row in rows
        ],
        dtype=np.float64,
    )

    lengths = np.asarray(
        [
            row["episode_length"]
            for row in rows
        ],
        dtype=np.float64,
    )

    completions = np.asarray(
        [
            row["completed"]
            for row in rows
        ],
        dtype=np.float64,
    )

    track_completion = np.asarray(
        [
            row["track_completion"]
            for row in rows
        ],
        dtype=np.float64,
    )

    return {
        "algorithm":
            rows[0]["algorithm"],

        "training_seed":
            rows[0]["training_seed"],

        "episodes":
            len(rows),

        "mean_reward":
            float(
                np.mean(rewards)
            ),

        "std_reward":
            (
                float(
                    np.std(
                        rewards,
                        ddof=1,
                    )
                )
                if len(rewards) > 1
                else 0.0
            ),

        "median_reward":
            float(
                np.median(rewards)
            ),

        "min_reward":
            float(
                np.min(rewards)
            ),

        "max_reward":
            float(
                np.max(rewards)
            ),

        "completion_rate":
            float(
                np.mean(completions)
            ),

        "mean_episode_length":
            float(
                np.mean(lengths)
            ),

        "std_episode_length":
            (
                float(
                    np.std(
                        lengths,
                        ddof=1,
                    )
                )
                if len(lengths) > 1
                else 0.0
            ),

        "mean_track_completion":
            float(
                np.mean(
                    track_completion
                )
            ),

        "deterministic":
            rows[0]["deterministic"],
    }


def main():
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be greater than zero"
        )

    if args.video_episodes <= 0:
        raise ValueError(
            "--video-episodes must be greater than zero"
        )

    # ------------------------------------------------
    # MODEL
    # ------------------------------------------------

    model_path = resolve_model_path(
        args
    )

    training_seed = infer_training_seed(
        model_path,
        args.train_seed,
    )

    run_name = (
        f"{args.algo}_seed_{training_seed}"
    )

    # ------------------------------------------------
    # EVALUATION SEEDS
    # ------------------------------------------------

    if args.eval_seeds is not None:
        eval_seeds = args.eval_seeds

    else:
        eval_seeds = list(
            range(
                args.eval_seed_start,
                args.eval_seed_start
                + args.episodes,
            )
        )

    # ------------------------------------------------
    # OUTPUT PATHS
    # ------------------------------------------------

    results_dir = Path(
        args.results_dir
    )

    raw_csv_path = (
        results_dir
        / f"{run_name}_episodes.csv"
    )

    summary_csv_path = (
        results_dir
        / f"{run_name}_summary.csv"
    )

    # ------------------------------------------------
    # ENVIRONMENT
    # ------------------------------------------------

    env = make_evaluation_env(
        args,
        run_name,
    )

    # ------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------

    model_class = ALGORITHMS[
        args.algo
    ]

    model = model_class.load(
        model_path,
        device=args.device,
    )

    deterministic = (
        not args.stochastic
    )

    rows: list[
        dict[str, Any]
    ] = []

    print()
    print(
        f"Algorithm:       "
        f"{args.algo.upper()}"
    )
    print(
        f"Model:           "
        f"{model_path}"
    )
    print(
        f"Training seed:   "
        f"{training_seed}"
    )
    print(
        f"Evaluation runs: "
        f"{len(eval_seeds)}"
    )
    print(
        f"Deterministic:   "
        f"{deterministic}"
    )
    print()

    # ------------------------------------------------
    # EVALUATION LOOP
    # ------------------------------------------------

    try:

        for episode_index, eval_seed in enumerate(
            eval_seeds,
            start=1,
        ):

            # IMPORTANT:
            # reset(seed=...) generates a reproducible track.
            #
            # PPO and DQN evaluated with the same eval_seed
            # therefore see the same track.
            obs, _ = env.reset(
                seed=eval_seed
            )

            episode_reward = 0.0
            episode_length = 0

            terminated = False
            truncated = False

            while not (
                terminated
                or truncated
            ):

                action, _ = model.predict(
                    obs,
                    deterministic=deterministic,
                )

                # We are using a non-vectorized
                # Discrete Gymnasium environment.
                if np.asarray(action).size == 1:
                    action = int(
                        np.asarray(
                            action
                        ).item()
                    )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(
                    action
                )

                episode_reward += float(
                    reward
                )

                episode_length += 1

            # ------------------------------------------------
            # CAR RACING SPECIFIC STATISTICS
            # ------------------------------------------------

            stats = get_track_stats(
                env
            )

            reason = termination_reason(
                completed=
                    stats["completed"],

                terminated=
                    terminated,

                truncated=
                    truncated,
            )

            # ------------------------------------------------
            # SAVE EPISODE DATA
            # ------------------------------------------------

            row = {
                "algorithm":
                    args.algo,

                "model_path":
                    str(model_path),

                "training_seed":
                    training_seed,

                "evaluation_seed":
                    eval_seed,

                "episode":
                    episode_index,

                "episode_reward":
                    episode_reward,

                "episode_length":
                    episode_length,

                "completed":
                    int(
                        stats["completed"]
                    ),

                "termination_reason":
                    reason,

                "visited_tiles":
                    stats[
                        "visited_tiles"
                    ],

                "total_tiles":
                    stats[
                        "total_tiles"
                    ],

                "track_completion":
                    stats[
                        "track_completion"
                    ],

                "lap_complete_percent":
                    stats[
                        "lap_complete_percent"
                    ],

                "terminated":
                    int(terminated),

                "truncated":
                    int(truncated),

                "deterministic":
                    int(deterministic),
            }

            rows.append(
                row
            )

            # ------------------------------------------------
            # PRINT EPISODE
            # ------------------------------------------------

            print(
                f"["
                f"{episode_index:02d}"
                f"/"
                f"{len(eval_seeds):02d}"
                f"] "
                f"eval_seed="
                f"{eval_seed:<4d} "
                f"reward="
                f"{episode_reward:8.2f}  "
                f"steps="
                f"{episode_length:4d}  "
                f"track="
                f"{100.0 * stats['track_completion']:6.2f}%  "
                f"status="
                f"{reason}"
            )

    finally:
        env.close()

    # ------------------------------------------------
    # SAVE RAW EPISODES
    # ------------------------------------------------

    save_csv(
        raw_csv_path,
        rows,
    )

    # ------------------------------------------------
    # BUILD + SAVE SUMMARY
    # ------------------------------------------------

    summary = build_summary(
        rows
    )

    save_csv(
        summary_csv_path,
        [summary],
    )

    # ------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------

    print()
    print(
        "========== EVALUATION SUMMARY =========="
    )

    print(
        f"Episodes:              "
        f"{summary['episodes']}"
    )

    print(
        f"Mean reward:           "
        f"{summary['mean_reward']:.2f}"
    )

    print(
        f"Std reward:            "
        f"{summary['std_reward']:.2f}"
    )

    print(
        f"Median reward:         "
        f"{summary['median_reward']:.2f}"
    )

    print(
        f"Min / max reward:      "
        f"{summary['min_reward']:.2f}"
        f" / "
        f"{summary['max_reward']:.2f}"
    )

    print(
        f"Completion rate:       "
        f"{100.0 * summary['completion_rate']:.1f}%"
    )

    print(
        f"Mean episode length:   "
        f"{summary['mean_episode_length']:.1f}"
    )

    print(
        f"Mean track completion: "
        f"{100.0 * summary['mean_track_completion']:.1f}%"
    )

    print(
        "========================================"
    )

    print(
        f"Per-episode CSV: "
        f"{raw_csv_path}"
    )

    print(
        f"Summary CSV:     "
        f"{summary_csv_path}"
    )

    if args.record_video:
        print(
            f"Videos:          "
            f"{Path(args.video_dir) / run_name}"
        )


if __name__ == "__main__":
    main()