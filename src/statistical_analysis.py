"""
statistical_analysis.py

Analisi statistica semplice dei risultati finali PPO vs DQN.

Scopo
-----
Questo script legge esclusivamente:

    aggregated_results/combined_test_episodes.csv

e produce statistiche descrittive facili da spiegare e studiare.

NON usa:
- Cohen's d;
- test t;
- test di Wilcoxon;
- permutation test;
- bootstrap;
- p-value.

L'obiettivo è mantenere l'analisi coerente con il livello del progetto e
rendere molto chiaro cosa significano i numeri.

Struttura dei dati finali
-------------------------
Nel protocollo attuale:

    2 algoritmi × 4 training seed × 30 piste = 240 episodi

Per evitare di trattare tutti i 120 episodi di un algoritmo come se fossero
120 training indipendenti, lo script distingue:

1. statistiche complessive sugli episodi;
2. statistiche per training seed;
3. statistiche per test track;
4. confronti appaiati PPO vs DQN.

Output prodotti
---------------
statistical_analysis/
├── statistical_summary.csv
├── per_training_seed.csv
├── per_test_track.csv
├── paired_training_seed_comparison.csv
├── paired_test_track_comparison.csv
└── statistical_report.json

Esecuzione
----------
Dalla root della repository:

    python src/statistical_analysis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path("aggregated_results")
DEFAULT_OUTPUT_DIR = Path("statistical_analysis")

ALGORITHMS = ("ppo", "dqn")

# Nel CSV consolidato la reward episodio-per-episodio si chiama
# "episode_reward".
METRICS = {
    "reward": "episode_reward",
    "track_completion": "track_completion",
}


# ---------------------------------------------------------------------------
# Argomenti da riga di comando
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Legge gli argomenti da riga di comando.

    Normalmente non serve specificare nulla:

        python src/statistical_analysis.py
    """

    parser = argparse.ArgumentParser(
        description="Analisi statistica descrittiva PPO vs DQN."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Cartella contenente combined_test_episodes.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella in cui salvare i risultati.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Lettura e controlli
# ---------------------------------------------------------------------------

def load_data(input_dir: Path) -> pd.DataFrame:
    """
    Carica il CSV consolidato creato da aggregate_results.py.
    """

    path = input_dir / "combined_test_episodes.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path}\n"
            "Eseguire prima: python src/aggregate_results.py"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("combined_test_episodes.csv è vuoto.")

    return df


def validate_data(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """
    Controlla che il dataset sia completo e coerente.

    Verifica:
    - colonne necessarie;
    - presenza di PPO e DQN;
    - nessun duplicato;
    - stessi training seed per PPO e DQN;
    - stessi test seed per PPO e DQN;
    - stessa quantità di test per ogni modello.
    """

    required = {
        "algorithm",
        "training_seed",
        "evaluation_seed",
        "episode_reward",
        "track_completion",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Mancano colonne necessarie: "
            + ", ".join(sorted(missing))
        )

    df["algorithm"] = df["algorithm"].astype(str).str.lower().str.strip()

    algorithms_found = set(df["algorithm"].unique())

    for algorithm in ALGORITHMS:
        if algorithm not in algorithms_found:
            raise ValueError(
                f"Algoritmo '{algorithm}' non trovato nel dataset."
            )

    # Controllo duplicati.
    key = ["algorithm", "training_seed", "evaluation_seed"]

    duplicated = df.duplicated(key, keep=False)

    if duplicated.any():
        raise ValueError(
            "Sono presenti episodi duplicati per "
            "algorithm/training_seed/evaluation_seed."
        )

    # Conversione esplicita delle colonne numeriche.
    numeric_columns = [
        "training_seed",
        "evaluation_seed",
        "episode_reward",
        "track_completion",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Sono presenti valori mancanti o non numerici "
            "nelle colonne principali."
        )

    # track_completion deve essere una frazione 0..1.
    if (
        (df["track_completion"] < 0)
        | (df["track_completion"] > 1)
    ).any():
        raise ValueError(
            "track_completion contiene valori fuori dall'intervallo [0, 1]."
        )

    seeds_ppo = sorted(
        df.loc[df["algorithm"] == "ppo", "training_seed"]
        .astype(int)
        .unique()
        .tolist()
    )

    seeds_dqn = sorted(
        df.loc[df["algorithm"] == "dqn", "training_seed"]
        .astype(int)
        .unique()
        .tolist()
    )

    if seeds_ppo != seeds_dqn:
        raise ValueError(
            "PPO e DQN non hanno gli stessi training seed."
        )

    tracks_ppo = sorted(
        df.loc[df["algorithm"] == "ppo", "evaluation_seed"]
        .astype(int)
        .unique()
        .tolist()
    )

    tracks_dqn = sorted(
        df.loc[df["algorithm"] == "dqn", "evaluation_seed"]
        .astype(int)
        .unique()
        .tolist()
    )

    if tracks_ppo != tracks_dqn:
        raise ValueError(
            "PPO e DQN non sono stati valutati sulle stesse piste."
        )

    expected_rows = len(ALGORITHMS) * len(seeds_ppo) * len(tracks_ppo)

    if len(df) != expected_rows:
        raise ValueError(
            f"Numero di righe inatteso: attese {expected_rows}, "
            f"trovate {len(df)}."
        )

    return seeds_ppo, tracks_ppo


# ---------------------------------------------------------------------------
# Funzioni statistiche semplici
# ---------------------------------------------------------------------------

def sample_std(series: pd.Series) -> float:
    """
    Calcola la deviazione standard campionaria.

    La deviazione standard misura quanto i valori sono dispersi
    rispetto alla media.

    Più è piccola, più i risultati sono simili tra loro.
    """

    if len(series) < 2:
        return float("nan")

    return float(series.std(ddof=1))


def basic_stats(series: pd.Series) -> dict[str, float]:
    """
    Restituisce le statistiche descrittive principali.

    - mean   = media;
    - median = mediana;
    - std    = deviazione standard;
    - min    = minimo;
    - max    = massimo.
    """

    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": sample_std(series),
        "min": float(series.min()),
        "max": float(series.max()),
    }


# ---------------------------------------------------------------------------
# Statistiche complessive
# ---------------------------------------------------------------------------

def build_algorithm_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce una riga di riepilogo per PPO e una per DQN.

    IMPORTANTE:
    la deviazione standard "between_training_seeds" viene calcolata
    sulle quattro medie dei modelli, non sui 120 episodi individuali.
    """

    rows = []

    for algorithm in ALGORITHMS:
        subset = df[df["algorithm"] == algorithm]

        # Media di ciascun modello sulle 30 piste.
        per_seed = (
            subset.groupby("training_seed", as_index=False)
            .agg(
                mean_reward=("episode_reward", "mean"),
                mean_track_completion=("track_completion", "mean"),
            )
        )

        pooled_reward = basic_stats(subset["episode_reward"])
        pooled_completion = basic_stats(subset["track_completion"])

        row = {
            "algorithm": algorithm,
            "n_training_seeds": int(per_seed["training_seed"].nunique()),
            "n_test_episodes": int(len(subset)),

            # Reward medio finale.
            "mean_reward": float(per_seed["mean_reward"].mean()),
            "median_run_mean_reward": float(
                per_seed["mean_reward"].median()
            ),
            "std_reward_between_training_seeds": sample_std(
                per_seed["mean_reward"]
            ),
            "min_run_mean_reward": float(
                per_seed["mean_reward"].min()
            ),
            "max_run_mean_reward": float(
                per_seed["mean_reward"].max()
            ),

            # Track completion.
            "mean_track_completion": float(
                per_seed["mean_track_completion"].mean()
            ),
            "std_track_completion_between_training_seeds": sample_std(
                per_seed["mean_track_completion"]
            ),

            # Statistiche sui 120 episodi individuali.
            # Sono utili come descrizione, ma non rappresentano 120
            # training indipendenti.
            "pooled_episode_reward_median": pooled_reward["median"],
            "pooled_episode_reward_std": pooled_reward["std"],
            "pooled_track_completion_median": pooled_completion["median"],
            "pooled_track_completion_std": pooled_completion["std"],
        }

        # Completion rate se la colonna è disponibile.
        if "completed" in subset.columns:
            completed_numeric = normalize_boolean_column(
                subset["completed"]
            )

            row["completed_episodes"] = int(completed_numeric.sum())
            row["completion_rate"] = float(completed_numeric.mean())

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Conversione della colonna completed
# ---------------------------------------------------------------------------

def normalize_boolean_column(series: pd.Series) -> pd.Series:
    """
    Converte la colonna completed in valori booleani.

    Gestisce:
    - True/False;
    - 1/0;
    - stringhe "true"/"false".
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        return numeric != 0

    text = series.astype(str).str.lower().str.strip()

    true_values = {"true", "1", "yes", "si", "sì"}
    false_values = {"false", "0", "no"}

    unknown = ~text.isin(true_values | false_values)

    if unknown.any():
        raise ValueError(
            "La colonna completed contiene valori non riconosciuti."
        )

    return text.isin(true_values)


# ---------------------------------------------------------------------------
# Statistiche per training seed
# ---------------------------------------------------------------------------

def build_per_training_seed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le statistiche di ciascun modello addestrato.

    Ogni riga rappresenta un training indipendente valutato sulle 30 piste.
    """

    aggregations = {
        "mean_reward": ("episode_reward", "mean"),
        "median_reward": ("episode_reward", "median"),
        "std_reward_across_tracks": ("episode_reward", "std"),
        "mean_track_completion": ("track_completion", "mean"),
        "std_track_completion_across_tracks": (
            "track_completion",
            "std",
        ),
        "n_test_tracks": ("evaluation_seed", "count"),
    }

    result = (
        df.groupby(
            ["algorithm", "training_seed"],
            as_index=False,
        )
        .agg(**aggregations)
        .sort_values(["algorithm", "training_seed"])
    )

    # Completion rate per modello, se disponibile.
    if "completed" in df.columns:
        temp = df.copy()
        temp["_completed"] = normalize_boolean_column(
            temp["completed"]
        ).astype(int)

        completion = (
            temp.groupby(
                ["algorithm", "training_seed"],
                as_index=False,
            )
            .agg(
                completed_episodes=("_completed", "sum"),
                completion_rate=("_completed", "mean"),
            )
        )

        result = result.merge(
            completion,
            on=["algorithm", "training_seed"],
            how="left",
        )

    return result


# ---------------------------------------------------------------------------
# Confronto appaiato per training seed
# ---------------------------------------------------------------------------

def build_paired_training_seed_comparison(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Confronta PPO e DQN usando lo stesso training seed.

    Esempio:

        seed 10:
            PPO reward - DQN reward

    La differenza positiva indica un vantaggio di PPO.
    """

    reward_table = per_seed.pivot(
        index="training_seed",
        columns="algorithm",
        values="mean_reward",
    )

    completion_table = per_seed.pivot(
        index="training_seed",
        columns="algorithm",
        values="mean_track_completion",
    )

    rows = []

    for seed in reward_table.index:
        reward_diff = float(
            reward_table.loc[seed, "ppo"]
            - reward_table.loc[seed, "dqn"]
        )

        completion_diff = float(
            completion_table.loc[seed, "ppo"]
            - completion_table.loc[seed, "dqn"]
        )

        rows.append(
            {
                "training_seed": int(seed),

                "ppo_mean_reward": float(
                    reward_table.loc[seed, "ppo"]
                ),
                "dqn_mean_reward": float(
                    reward_table.loc[seed, "dqn"]
                ),
                "reward_difference_ppo_minus_dqn": reward_diff,
                "reward_winner": (
                    "ppo"
                    if reward_diff > 0
                    else "dqn"
                    if reward_diff < 0
                    else "tie"
                ),

                "ppo_mean_track_completion": float(
                    completion_table.loc[seed, "ppo"]
                ),
                "dqn_mean_track_completion": float(
                    completion_table.loc[seed, "dqn"]
                ),
                "track_completion_difference_ppo_minus_dqn": (
                    completion_diff
                ),
                "track_completion_winner": (
                    "ppo"
                    if completion_diff > 0
                    else "dqn"
                    if completion_diff < 0
                    else "tie"
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistiche per test track
# ---------------------------------------------------------------------------

def build_per_test_track(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni test track calcola la media sui quattro training seed.

    Questo permette di capire se il vantaggio di un algoritmo è diffuso
    sulle piste oppure concentrato solo su pochi circuiti.
    """

    return (
        df.groupby(
            ["algorithm", "evaluation_seed"],
            as_index=False,
        )
        .agg(
            mean_reward=("episode_reward", "mean"),
            median_reward=("episode_reward", "median"),
            std_reward_between_models=("episode_reward", "std"),
            mean_track_completion=("track_completion", "mean"),
            std_track_completion_between_models=(
                "track_completion",
                "std",
            ),
            n_models=("training_seed", "count"),
        )
        .sort_values(["evaluation_seed", "algorithm"])
    )


# ---------------------------------------------------------------------------
# Confronto appaiato per test track
# ---------------------------------------------------------------------------

def build_paired_track_comparison(
    per_track: pd.DataFrame,
) -> pd.DataFrame:
    """
    Confronta PPO e DQN sulla stessa pista.

    Le prestazioni di ciascun algoritmo sono prima mediate sui quattro
    training seed.
    """

    reward_table = per_track.pivot(
        index="evaluation_seed",
        columns="algorithm",
        values="mean_reward",
    )

    completion_table = per_track.pivot(
        index="evaluation_seed",
        columns="algorithm",
        values="mean_track_completion",
    )

    rows = []

    for seed in reward_table.index:
        reward_diff = float(
            reward_table.loc[seed, "ppo"]
            - reward_table.loc[seed, "dqn"]
        )

        completion_diff = float(
            completion_table.loc[seed, "ppo"]
            - completion_table.loc[seed, "dqn"]
        )

        rows.append(
            {
                "evaluation_seed": int(seed),

                "ppo_mean_reward": float(
                    reward_table.loc[seed, "ppo"]
                ),
                "dqn_mean_reward": float(
                    reward_table.loc[seed, "dqn"]
                ),
                "reward_difference_ppo_minus_dqn": reward_diff,
                "reward_winner": (
                    "ppo"
                    if reward_diff > 0
                    else "dqn"
                    if reward_diff < 0
                    else "tie"
                ),

                "ppo_mean_track_completion": float(
                    completion_table.loc[seed, "ppo"]
                ),
                "dqn_mean_track_completion": float(
                    completion_table.loc[seed, "dqn"]
                ),
                "track_completion_difference_ppo_minus_dqn": (
                    completion_diff
                ),
                "track_completion_winner": (
                    "ppo"
                    if completion_diff > 0
                    else "dqn"
                    if completion_diff < 0
                    else "tie"
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Confronto dei 120 casi training seed × test track
# ---------------------------------------------------------------------------

def count_episode_level_wins(
    df: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """
    Conta quante volte PPO e DQN vincono negli stessi 120 casi:

        training_seed × evaluation_seed

    ATTENZIONE:
    questo conteggio è descrittivo.
    Non viene usato per fingere di avere 120 training indipendenti.
    """

    result = {}

    for label, column in METRICS.items():
        table = df.pivot(
            index=["training_seed", "evaluation_seed"],
            columns="algorithm",
            values=column,
        )

        diff = table["ppo"] - table["dqn"]

        result[label] = {
            "ppo_wins": int((diff > 0).sum()),
            "dqn_wins": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
            "total_comparisons": int(len(diff)),
        }

    return result


# ---------------------------------------------------------------------------
# Report finale
# ---------------------------------------------------------------------------

def build_report(
    summary: pd.DataFrame,
    paired_seed: pd.DataFrame,
    paired_track: pd.DataFrame,
    episode_wins: dict,
    training_seeds: list[int],
    test_tracks: list[int],
) -> dict:
    """
    Costruisce un JSON leggibile con i risultati più importanti.
    """

    ppo = summary[summary["algorithm"] == "ppo"].iloc[0]
    dqn = summary[summary["algorithm"] == "dqn"].iloc[0]

    reward_difference = float(
        ppo["mean_reward"] - dqn["mean_reward"]
    )

    track_difference = float(
        ppo["mean_track_completion"]
        - dqn["mean_track_completion"]
    )

    # Miglioramento relativo del reward rispetto a DQN.
    relative_reward_improvement = (
        100.0 * reward_difference / abs(float(dqn["mean_reward"]))
        if float(dqn["mean_reward"]) != 0
        else None
    )

    return {
        "analysis_version": 2,

        "protocol": {
            "training_seeds": training_seeds,
            "test_tracks": test_tracks,
            "n_training_seeds": len(training_seeds),
            "n_test_tracks": len(test_tracks),
            "n_total_episodes": int(
                len(ALGORITHMS)
                * len(training_seeds)
                * len(test_tracks)
            ),
        },

        "main_results": {
            "ppo_mean_reward": float(ppo["mean_reward"]),
            "dqn_mean_reward": float(dqn["mean_reward"]),
            "reward_difference_ppo_minus_dqn": reward_difference,
            "reward_improvement_vs_dqn_percent": (
                relative_reward_improvement
            ),

            "ppo_mean_track_completion": float(
                ppo["mean_track_completion"]
            ),
            "dqn_mean_track_completion": float(
                dqn["mean_track_completion"]
            ),
            "track_completion_difference_percentage_points": (
                100.0 * track_difference
            ),

            "ppo_std_reward_between_training_seeds": float(
                ppo["std_reward_between_training_seeds"]
            ),
            "dqn_std_reward_between_training_seeds": float(
                dqn["std_reward_between_training_seeds"]
            ),
        },

        "training_seed_comparison": {
            "ppo_reward_wins": int(
                (paired_seed["reward_winner"] == "ppo").sum()
            ),
            "dqn_reward_wins": int(
                (paired_seed["reward_winner"] == "dqn").sum()
            ),
            "ties": int(
                (paired_seed["reward_winner"] == "tie").sum()
            ),
        },

        "test_track_comparison": {
            "ppo_reward_wins": int(
                (paired_track["reward_winner"] == "ppo").sum()
            ),
            "dqn_reward_wins": int(
                (paired_track["reward_winner"] == "dqn").sum()
            ),
            "ties": int(
                (paired_track["reward_winner"] == "tie").sum()
            ),
        },

        "individual_seed_track_comparison": episode_wins,

        "interpretation": {
            "important_note": (
                "Le statistiche principali vengono interpretate soprattutto "
                "tra i training seed, perché ogni training seed rappresenta "
                "un processo di apprendimento indipendente."
            ),
            "std_note": (
                "La deviazione standard tra training seed misura quanto "
                "cambiano le prestazioni medie dei modelli al cambiare "
                "del seed di training."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Salvataggio
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    summary: pd.DataFrame,
    per_seed: pd.DataFrame,
    per_track: pd.DataFrame,
    paired_seed: pd.DataFrame,
    paired_track: pd.DataFrame,
    report: dict,
) -> None:
    """
    Crea la cartella di output e salva tutti i risultati.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output_dir / "statistical_summary.csv",
        index=False,
    )

    per_seed.to_csv(
        output_dir / "per_training_seed.csv",
        index=False,
    )

    per_track.to_csv(
        output_dir / "per_test_track.csv",
        index=False,
    )

    paired_seed.to_csv(
        output_dir / "paired_training_seed_comparison.csv",
        index=False,
    )

    paired_track.to_csv(
        output_dir / "paired_test_track_comparison.csv",
        index=False,
    )

    with (
        output_dir / "statistical_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Output nel terminale
# ---------------------------------------------------------------------------

def print_summary(
    summary: pd.DataFrame,
    paired_seed: pd.DataFrame,
    paired_track: pd.DataFrame,
    episode_wins: dict,
) -> None:
    """
    Stampa nel terminale soltanto i numeri più importanti.
    """

    ppo = summary[summary["algorithm"] == "ppo"].iloc[0]
    dqn = summary[summary["algorithm"] == "dqn"].iloc[0]

    reward_difference = (
        float(ppo["mean_reward"])
        - float(dqn["mean_reward"])
    )

    completion_difference = (
        float(ppo["mean_track_completion"])
        - float(dqn["mean_track_completion"])
    )

    print("\nAnalisi statistica descrittiva completata.")

    print("\n=== REWARD FINALE ===")
    print(f"PPO: {ppo['mean_reward']:.2f}")
    print(f"DQN: {dqn['mean_reward']:.2f}")
    print(f"Differenza PPO - DQN: {reward_difference:.2f}")

    if float(dqn["mean_reward"]) != 0:
        relative = (
            100.0
            * reward_difference
            / abs(float(dqn["mean_reward"]))
        )
        print(
            f"Vantaggio relativo rispetto a DQN: {relative:.2f}%"
        )

    print("\nVariabilità tra training seed:")
    print(
        "PPO std: "
        f"{ppo['std_reward_between_training_seeds']:.2f}"
    )
    print(
        "DQN std: "
        f"{dqn['std_reward_between_training_seeds']:.2f}"
    )

    print("\n=== TRACK COMPLETION ===")
    print(
        f"PPO: "
        f"{100.0 * ppo['mean_track_completion']:.2f}%"
    )
    print(
        f"DQN: "
        f"{100.0 * dqn['mean_track_completion']:.2f}%"
    )
    print(
        "Differenza: "
        f"{100.0 * completion_difference:.2f} punti percentuali"
    )

    print("\n=== CONFRONTO TRA TRAINING SEED ===")
    print(
        "PPO migliore: "
        f"{(paired_seed['reward_winner'] == 'ppo').sum()}"
        f"/{len(paired_seed)}"
    )
    print(
        "DQN migliore: "
        f"{(paired_seed['reward_winner'] == 'dqn').sum()}"
        f"/{len(paired_seed)}"
    )

    print("\n=== CONFRONTO TRA LE 30 PISTE ===")
    print(
        "PPO migliore: "
        f"{(paired_track['reward_winner'] == 'ppo').sum()}"
        f"/{len(paired_track)}"
    )
    print(
        "DQN migliore: "
        f"{(paired_track['reward_winner'] == 'dqn').sum()}"
        f"/{len(paired_track)}"
    )

    print("\n=== CONFRONTI TRAINING SEED × PISTA ===")
    reward_wins = episode_wins["reward"]

    print(
        f"PPO: {reward_wins['ppo_wins']} / "
        f"{reward_wins['total_comparisons']}"
    )
    print(
        f"DQN: {reward_wins['dqn_wins']} / "
        f"{reward_wins['total_comparisons']}"
    )
    print(
        f"Pareggi: {reward_wins['ties']}"
    )

    if (
        "completed_episodes" in summary.columns
        and "completion_rate" in summary.columns
    ):
        print("\n=== GIRI COMPLETATI ===")

        print(
            f"PPO: {int(ppo['completed_episodes'])}/"
            f"{int(ppo['n_test_episodes'])} "
            f"({100.0 * ppo['completion_rate']:.2f}%)"
        )

        print(
            f"DQN: {int(dqn['completed_episodes'])}/"
            f"{int(dqn['n_test_episodes'])} "
            f"({100.0 * dqn['completion_rate']:.2f}%)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Coordina l'intera analisi.
    """

    args = parse_args()

    df = load_data(args.input_dir)

    training_seeds, test_tracks = validate_data(df)

    summary = build_algorithm_summary(df)

    per_seed = build_per_training_seed(df)

    paired_seed = build_paired_training_seed_comparison(
        per_seed
    )

    per_track = build_per_test_track(df)

    paired_track = build_paired_track_comparison(
        per_track
    )

    episode_wins = count_episode_level_wins(df)

    report = build_report(
        summary=summary,
        paired_seed=paired_seed,
        paired_track=paired_track,
        episode_wins=episode_wins,
        training_seeds=training_seeds,
        test_tracks=test_tracks,
    )

    save_outputs(
        output_dir=args.output_dir,
        summary=summary,
        per_seed=per_seed,
        per_track=per_track,
        paired_seed=paired_seed,
        paired_track=paired_track,
        report=report,
    )

    print_summary(
        summary=summary,
        paired_seed=paired_seed,
        paired_track=paired_track,
        episode_wins=episode_wins,
    )

    print("\nFile prodotti:")
    print(
        f"  {args.output_dir / 'statistical_summary.csv'}"
    )
    print(
        f"  {args.output_dir / 'per_training_seed.csv'}"
    )
    print(
        f"  {args.output_dir / 'per_test_track.csv'}"
    )
    print(
        f"  {args.output_dir / 'paired_training_seed_comparison.csv'}"
    )
    print(
        f"  {args.output_dir / 'paired_test_track_comparison.csv'}"
    )
    print(
        f"  {args.output_dir / 'statistical_report.json'}"
    )

    print(
        "\nNota: questa versione usa soltanto statistiche descrittive "
        "semplici (media, mediana, deviazione standard, differenze e "
        "conteggio delle vittorie)."
    )


if __name__ == "__main__":
    main()
