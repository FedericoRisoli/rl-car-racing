"""
Aggrega i risultati finali del progetto PPO vs DQN su CarRacing-v2.

Il file lavora su DUE sorgenti diverse, che hanno ruoli metodologici diversi:

1) logs/<run_final>/evaluations.csv
   Sono le evaluation PERIODICHE fatte durante il training sulle piste di
   validation (nel protocollo attuale: seed 200-204). Servono per costruire
   le learning curve e per documentare il checkpoint scelto come best model.

2) results/<algo>_seed_<training_seed>_episodes.csv
   Sono le evaluation FINALI prodotte da evaluate.py sui test seed mai usati
   durante tuning/validation (nel protocollo attuale: 1000-1029). Servono per
   il confronto finale PPO vs DQN.

Principio statistico importante
-------------------------------
Le piste di test NON vengono trattate come se fossero training indipendenti.
Per il confronto tra algoritmi, l'unita' sperimentale principale e' il
TRAINING SEED. Per questo il file produce prima statistiche per singolo run e
poi aggrega tali statistiche tra i training seed.

Output principali
-----------------
- combined_test_episodes.csv
- test_per_run_summary.csv
- test_algorithm_summary.csv
- test_per_track_summary.csv
- test_paired_seed_comparison.csv
- training_run_summary.csv
- learning_curve_per_run.csv
- learning_curve_algorithm.csv
- aggregation_report.json

Il programma usa solo Python standard library + NumPy, gia' dipendenza del
progetto. Non richiede pandas o scipy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# COSTANTI DEL PROTOCOLLO FINALE
# ---------------------------------------------------------------------------

ALGORITHMS = ("ppo", "dqn")

# Questi default corrispondono al protocollo deciso per gli esperimenti finali.
DEFAULT_TRAINING_SEEDS = (10, 11, 12, 13)
DEFAULT_VALIDATION_SEEDS = (200, 201, 202, 203, 204)
DEFAULT_TEST_SEED_START = 1000
DEFAULT_TEST_EPISODES = 30

# Colonne che evaluate.py salva nel CSV episodio-per-episodio.
REQUIRED_TEST_COLUMNS = {
    "algorithm",
    "model_path",
    "training_seed",
    "evaluation_seed",
    "episode",
    "episode_reward",
    "episode_length",
    "completed",
    "termination_reason",
    "visited_tiles",
    "total_tiles",
    "track_completion",
    "lap_complete_percent",
    "terminated",
    "truncated",
    "deterministic",
}

# Colonne che train.py salva durante le evaluation periodiche.
REQUIRED_LEARNING_CURVE_COLUMNS = {
    "timesteps",
    "eval_seed",
    "episode_reward",
    "episode_length",
}


# ---------------------------------------------------------------------------
# ARGOMENTI DA RIGA DI COMANDO
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Legge e valida gli argomenti principali passati da PowerShell."""

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate final PPO/DQN training validation curves and final "
            "test results."
        )
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Cartella contenente i CSV prodotti da evaluate.py.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Cartella contenente i log prodotti da train.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("aggregated_results"),
        help="Cartella in cui scrivere i CSV aggregati.",
    )

    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAINING_SEEDS),
        help="Training seed ufficiali attesi.",
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
        help="Validation seed usati durante i training finali.",
    )
    parser.add_argument(
        "--test-seed-start",
        type=int,
        default=DEFAULT_TEST_SEED_START,
        help="Primo seed del test finale.",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=DEFAULT_TEST_EPISODES,
        help="Numero di piste/episodi del test finale.",
    )

    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Permette di aggregare dati parziali mostrando warning. "
            "Senza questa opzione, dati mancanti o incoerenti bloccano "
            "l'aggregazione finale."
        ),
    )
    parser.add_argument(
        "--allow-non-best-model",
        action="store_true",
        help=(
            "Permette CSV di test ottenuti da modelli il cui nome non "
            "contiene '_best'. Normalmente il test finale deve usare il "
            "best model scelto sulla validation."
        ),
    )
    parser.add_argument(
        "--allow-stochastic",
        action="store_true",
        help=(
            "Permette test finali stochastic. Il protocollo ufficiale "
            "prevede invece deterministic=True."
        ),
    )

    args = parser.parse_args()

    # Controlli immediati: meglio fallire con un messaggio chiaro che produrre
    # statistiche apparentemente corrette su parametri sbagliati.
    if args.test_episodes <= 0:
        parser.error("--test-episodes deve essere maggiore di 0.")

    if not args.training_seeds:
        parser.error("Deve essere specificato almeno un training seed.")

    if not args.validation_seeds:
        parser.error("Deve essere specificato almeno un validation seed.")

    if len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error("--training-seeds contiene duplicati.")

    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        parser.error("--validation-seeds contiene duplicati.")

    return args


# ---------------------------------------------------------------------------
# FUNZIONI GENERICHE DI SUPPORTO
# ---------------------------------------------------------------------------


def expected_test_seeds(args: argparse.Namespace) -> list[int]:
    """Costruisce la lista dei test seed consecutivi attesi."""

    return list(
        range(
            args.test_seed_start,
            args.test_seed_start + args.test_episodes,
        )
    )


def fail_or_warn(
    condition: bool,
    message: str,
    *,
    allow_incomplete: bool,
    warnings: list[str],
) -> None:
    """
    Applica la modalita' strict del programma.

    - Se condition e' vera, non fa nulla.
    - Se e' falsa e allow_incomplete=False, solleva ValueError.
    - Se e' falsa e allow_incomplete=True, registra un warning e continua.

    Questo evita che l'analisi finale ignori silenziosamente dati mancanti.
    """

    if condition:
        return

    if allow_incomplete:
        warnings.append(message)
        print(f"WARNING: {message}", file=sys.stderr)
        return

    raise ValueError(message)


def canonical_json(value: Any) -> str:
    """Serializza strutture JSON in modo stabile per confrontarle."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def require_columns(path: Path, fieldnames: Iterable[str] | None, required: set[str]) -> None:
    """Verifica che un CSV contenga tutte le colonne necessarie."""

    available = set(fieldnames or [])
    missing = sorted(required - available)

    if missing:
        raise ValueError(
            f"CSV {path} privo delle colonne richieste: {missing}"
        )


def read_dict_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    """Legge un CSV come lista di dizionari e ne controlla lo schema."""

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        require_columns(path, reader.fieldnames, required_columns)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV vuoto: {path}")

    return rows


def read_json(path: Path) -> dict[str, Any]:
    """Carica un file JSON assicurandosi che contenga un oggetto/dizionario."""

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(f"Il JSON deve contenere un oggetto: {path}")

    return value


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Scrive un CSV con ordine delle colonne deterministico."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Scrive JSON leggibile e stabile, utile anche da versionare su Git."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=4, ensure_ascii=False, sort_keys=True)


def parse_int(value: Any, *, field: str, path: Path) -> int:
    """Converte un valore CSV in intero producendo errori esplicativi."""

    try:
        # int(float(...)) rende robusta la lettura anche di valori come "1.0".
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Valore non numerico per '{field}' in {path}: {value!r}"
        ) from error

    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(
            f"Valore non intero per '{field}' in {path}: {value!r}"
        )

    return int(number)


def parse_float(value: Any, *, field: str, path: Path) -> float:
    """Converte un valore CSV in float e rifiuta NaN/inf."""

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Valore non numerico per '{field}' in {path}: {value!r}"
        ) from error

    if not math.isfinite(number):
        raise ValueError(
            f"Valore non finito per '{field}' in {path}: {value!r}"
        )

    return number


def parse_binary_flag(value: Any, *, field: str, path: Path) -> int:
    """Converte 0/1, true/false, yes/no in un flag intero 0 o 1."""

    text = str(value).strip().lower()

    if text in {"1", "true", "yes"}:
        return 1
    if text in {"0", "false", "no"}:
        return 0

    raise ValueError(
        f"Flag non valido per '{field}' in {path}: {value!r}"
    )


def sample_std(values: Iterable[float]) -> float:
    """Deviazione standard campionaria (ddof=1), 0 se c'e' un solo valore."""

    array = np.asarray(list(values), dtype=np.float64)

    if array.size <= 1:
        return 0.0

    return float(np.std(array, ddof=1))


def population_std(values: Iterable[float]) -> float:
    """
    Deviazione standard descrittiva della lista completa (ddof=0).

    Viene usata per le 5 piste fisse di validation per restare coerenti con
    train.py, che usa np.std senza specificare ddof.
    """

    array = np.asarray(list(values), dtype=np.float64)

    if array.size == 0:
        return 0.0

    return float(np.std(array, ddof=0))


def mean(values: Iterable[float]) -> float:
    """Media aritmetica convertita esplicitamente in float Python."""

    array = np.asarray(list(values), dtype=np.float64)

    if array.size == 0:
        raise ValueError("Impossibile calcolare la media di una lista vuota.")

    return float(np.mean(array))


def median(values: Iterable[float]) -> float:
    """Mediana convertita esplicitamente in float Python."""

    array = np.asarray(list(values), dtype=np.float64)

    if array.size == 0:
        raise ValueError("Impossibile calcolare la mediana di una lista vuota.")

    return float(np.median(array))


# ---------------------------------------------------------------------------
# TEST FINALE: LETTURA E VALIDAZIONE DEGLI EPISODI
# ---------------------------------------------------------------------------


def normalize_test_row(raw: dict[str, str], path: Path) -> dict[str, Any]:
    """Converte una riga di evaluate.py in tipi Python affidabili."""

    algorithm = raw["algorithm"].strip().lower()

    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Algoritmo non riconosciuto in {path}: {algorithm!r}"
        )

    row = {
        "algorithm": algorithm,
        "model_path": raw["model_path"].strip(),
        "training_seed": parse_int(
            raw["training_seed"], field="training_seed", path=path
        ),
        "evaluation_seed": parse_int(
            raw["evaluation_seed"], field="evaluation_seed", path=path
        ),
        "episode": parse_int(raw["episode"], field="episode", path=path),
        "episode_reward": parse_float(
            raw["episode_reward"], field="episode_reward", path=path
        ),
        "episode_length": parse_int(
            raw["episode_length"], field="episode_length", path=path
        ),
        "completed": parse_binary_flag(
            raw["completed"], field="completed", path=path
        ),
        "termination_reason": raw["termination_reason"].strip(),
        "visited_tiles": parse_int(
            raw["visited_tiles"], field="visited_tiles", path=path
        ),
        "total_tiles": parse_int(
            raw["total_tiles"], field="total_tiles", path=path
        ),
        "track_completion": parse_float(
            raw["track_completion"], field="track_completion", path=path
        ),
        "lap_complete_percent": parse_float(
            raw["lap_complete_percent"],
            field="lap_complete_percent",
            path=path,
        ),
        "terminated": parse_binary_flag(
            raw["terminated"], field="terminated", path=path
        ),
        "truncated": parse_binary_flag(
            raw["truncated"], field="truncated", path=path
        ),
        "deterministic": parse_binary_flag(
            raw["deterministic"], field="deterministic", path=path
        ),
        "source_file": str(path),
    }

    # Controlli di dominio: intercettano CSV corrotti o metriche impossibili.
    if row["episode_length"] <= 0:
        raise ValueError(f"episode_length <= 0 in {path}: {row}")

    if row["visited_tiles"] < 0 or row["total_tiles"] < 0:
        raise ValueError(f"Numero di tile negativo in {path}: {row}")

    if row["total_tiles"] > 0 and row["visited_tiles"] > row["total_tiles"]:
        raise ValueError(f"visited_tiles > total_tiles in {path}: {row}")

    if not (0.0 <= row["track_completion"] <= 1.0 + 1e-9):
        raise ValueError(f"track_completion fuori da [0,1] in {path}: {row}")

    if not (0.0 < row["lap_complete_percent"] <= 1.0):
        raise ValueError(f"lap_complete_percent non valido in {path}: {row}")

    # evaluate.py assegna "lap_completed" esattamente quando completed=True.
    if row["completed"] == 1 and row["termination_reason"] != "lap_completed":
        raise ValueError(
            f"completed=1 ma termination_reason non e' lap_completed in {path}: {row}"
        )

    if row["termination_reason"] == "lap_completed" and row["completed"] != 1:
        raise ValueError(
            f"termination_reason=lap_completed ma completed=0 in {path}: {row}"
        )

    return row


def load_final_test_episodes(
    args: argparse.Namespace,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """
    Carica tutti i CSV episodio-per-episodio del test finale.

    I file estranei (pilot, vecchi seed, altri esperimenti) vengono ignorati
    se non contengono una combinazione algoritmo/training_seed ufficiale.
    """

    if not args.results_dir.exists():
        fail_or_warn(
            False,
            f"Cartella results non trovata: {args.results_dir}",
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )
        return []

    official_training_seeds = set(args.training_seeds)
    episodes: list[dict[str, Any]] = []

    # evaluate.py salva file con suffisso _episodes.csv. rglob rende il codice
    # robusto anche se in futuro i risultati vengono organizzati in sottocartelle.
    for path in sorted(args.results_dir.rglob("*_episodes.csv")):
        raw_rows = read_dict_csv(path, REQUIRED_TEST_COLUMNS)
        normalized = [normalize_test_row(row, path) for row in raw_rows]

        # Ogni file evaluate.py dovrebbe riferirsi a un unico algoritmo e seed.
        combos = {
            (row["algorithm"], row["training_seed"])
            for row in normalized
        }
        if len(combos) != 1:
            raise ValueError(
                f"Il file {path} contiene piu' run differenti: {sorted(combos)}"
            )

        algorithm, training_seed = next(iter(combos))

        # Ignora automaticamente vecchi smoke/pilot o seed non ufficiali.
        if algorithm not in ALGORITHMS or training_seed not in official_training_seeds:
            continue

        episodes.extend(normalized)

    expected_combos = {
        (algorithm, seed)
        for algorithm in ALGORITHMS
        for seed in args.training_seeds
    }
    found_combos = {
        (row["algorithm"], row["training_seed"])
        for row in episodes
    }

    fail_or_warn(
        found_combos == expected_combos,
        (
            "Combinazioni finali algoritmo/seed incomplete o inattese. "
            f"Attese={sorted(expected_combos)}, trovate={sorted(found_combos)}"
        ),
        allow_incomplete=args.allow_incomplete,
        warnings=warnings,
    )

    return episodes


def validate_final_test_episodes(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    warnings: list[str],
) -> None:
    """Esegue controlli incrociati sul test finale prima di aggregarlo."""

    if not rows:
        fail_or_warn(
            False,
            "Nessun episodio di test finale disponibile.",
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )
        return

    expected_eval_seeds = set(expected_test_seeds(args))

    # Non possono esistere due risultati per la stessa tripletta.
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (
            row["algorithm"],
            row["training_seed"],
            row["evaluation_seed"],
        )
        if key in seen:
            raise ValueError(
                "Episodio duplicato per algorithm/training_seed/evaluation_seed: "
                f"{key}"
            )
        seen.add(key)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["training_seed"])].append(row)

    for key, run_rows in sorted(grouped.items()):
        actual_eval_seeds = {row["evaluation_seed"] for row in run_rows}

        fail_or_warn(
            actual_eval_seeds == expected_eval_seeds,
            (
                f"Test seed errati/incompleti per {key}. "
                f"Attesi={sorted(expected_eval_seeds)}, "
                f"trovati={sorted(actual_eval_seeds)}"
            ),
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )

        model_paths = {row["model_path"] for row in run_rows}
        fail_or_warn(
            len(model_paths) == 1,
            f"Piu' model_path nello stesso run {key}: {sorted(model_paths)}",
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )

        # Il protocollo finale prevede il checkpoint scelto sulla validation.
        if model_paths and not args.allow_non_best_model:
            model_path = Path(next(iter(model_paths)))
            fail_or_warn(
                "_best" in model_path.stem.lower(),
                (
                    f"Il test {key} non sembra usare un best model: "
                    f"{model_path}. Usa --allow-non-best-model solo se e' "
                    "una scelta intenzionale."
                ),
                allow_incomplete=args.allow_incomplete,
                warnings=warnings,
            )

        # Il protocollo prevede evaluation deterministica.
        if not args.allow_stochastic:
            deterministic_values = {row["deterministic"] for row in run_rows}
            fail_or_warn(
                deterministic_values == {1},
                f"Evaluation non completamente deterministica per {key}.",
                allow_incomplete=args.allow_incomplete,
                warnings=warnings,
            )

    # Stesso evaluation seed => stessa pista => stesso numero di tile.
    # E' un controllo molto utile per verificare che PPO e DQN abbiano visto
    # davvero gli stessi circuiti di test.
    by_eval_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_eval_seed[row["evaluation_seed"]].append(row)

    for eval_seed, seed_rows in sorted(by_eval_seed.items()):
        total_tiles = {
            row["total_tiles"]
            for row in seed_rows
            if row["total_tiles"] > 0
        }

        fail_or_warn(
            len(total_tiles) <= 1,
            (
                f"La pista test seed={eval_seed} ha total_tiles differenti "
                f"tra i run: {sorted(total_tiles)}"
            ),
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )

    # Anche la soglia di completamento deve essere identica in tutto il test.
    thresholds = {row["lap_complete_percent"] for row in rows}
    fail_or_warn(
        len(thresholds) == 1,
        f"lap_complete_percent non coerente tra i test: {sorted(thresholds)}",
        allow_incomplete=args.allow_incomplete,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# TEST FINALE: AGGREGAZIONI
# ---------------------------------------------------------------------------


def summarize_test_run(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcola le statistiche di UN singolo training seed sul test finale."""

    first = run_rows[0]
    rewards = [row["episode_reward"] for row in run_rows]
    lengths = [float(row["episode_length"]) for row in run_rows]
    completions = [float(row["completed"]) for row in run_rows]
    track_completion = [row["track_completion"] for row in run_rows]

    n = len(run_rows)

    reason_counts: dict[str, int] = defaultdict(int)
    for row in run_rows:
        reason_counts[row["termination_reason"]] += 1

    return {
        "algorithm": first["algorithm"],
        "training_seed": first["training_seed"],
        "episodes": n,
        "mean_reward": mean(rewards),
        "std_reward": sample_std(rewards),
        "median_reward": median(rewards),
        "min_reward": float(min(rewards)),
        "max_reward": float(max(rewards)),
        "completion_rate": mean(completions),
        "mean_track_completion": mean(track_completion),
        "std_track_completion": sample_std(track_completion),
        "mean_episode_length": mean(lengths),
        "std_episode_length": sample_std(lengths),
        "lap_completed_rate": reason_counts["lap_completed"] / n,
        "out_of_bounds_rate": reason_counts["out_of_bounds"] / n,
        "time_limit_rate": reason_counts["time_limit"] / n,
        "unknown_termination_rate": reason_counts["unknown"] / n,
    }


def build_test_per_run_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Crea una riga di statistiche per ogni algoritmo e training seed."""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["training_seed"])].append(row)

    summaries = [
        summarize_test_run(sorted(group, key=lambda item: item["evaluation_seed"]))
        for _, group in sorted(grouped.items())
    ]

    return summaries


def build_test_algorithm_summary(
    per_run_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Aggrega le statistiche TRA training seed.

    Le colonne "mean_*_across_training_seeds" sono quelle da privilegiare nel
    confronto scientifico, perche' ogni training seed conta come replica.

    Le colonne "pooled_*" sono descrittive: uniscono tutti gli episodi e NON
    devono essere confuse con il numero di repliche indipendenti.
    """

    per_run_by_algo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    episodes_by_algo: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in per_run_rows:
        per_run_by_algo[row["algorithm"]].append(row)
    for row in episode_rows:
        episodes_by_algo[row["algorithm"]].append(row)

    output: list[dict[str, Any]] = []

    for algorithm in ALGORITHMS:
        runs = per_run_by_algo.get(algorithm, [])
        episodes = episodes_by_algo.get(algorithm, [])

        if not runs:
            continue

        run_mean_rewards = [row["mean_reward"] for row in runs]
        run_completion_rates = [row["completion_rate"] for row in runs]
        run_track_completion = [row["mean_track_completion"] for row in runs]
        run_lengths = [row["mean_episode_length"] for row in runs]

        pooled_rewards = [row["episode_reward"] for row in episodes]
        pooled_completions = [float(row["completed"]) for row in episodes]
        pooled_track_completion = [row["track_completion"] for row in episodes]

        output.append(
            {
                "algorithm": algorithm,
                "n_training_seeds": len(runs),
                "n_test_episodes_total": len(episodes),
                "mean_reward_across_training_seeds": mean(run_mean_rewards),
                "std_reward_between_training_seeds": sample_std(run_mean_rewards),
                "median_reward_across_training_seeds": median(run_mean_rewards),
                "min_run_mean_reward": float(min(run_mean_rewards)),
                "max_run_mean_reward": float(max(run_mean_rewards)),
                "mean_completion_rate_across_training_seeds": mean(
                    run_completion_rates
                ),
                "std_completion_rate_between_training_seeds": sample_std(
                    run_completion_rates
                ),
                "mean_track_completion_across_training_seeds": mean(
                    run_track_completion
                ),
                "std_track_completion_between_training_seeds": sample_std(
                    run_track_completion
                ),
                "mean_episode_length_across_training_seeds": mean(run_lengths),
                "std_episode_length_between_training_seeds": sample_std(
                    run_lengths
                ),
                # Le seguenti metriche pooled sono utili per descrivere tutti
                # gli episodi, ma non sostituiscono la variabilita' tra seed.
                "pooled_episode_mean_reward": mean(pooled_rewards),
                "pooled_episode_std_reward": sample_std(pooled_rewards),
                "pooled_completion_rate": mean(pooled_completions),
                "pooled_mean_track_completion": mean(pooled_track_completion),
            }
        )

    return output


def build_test_per_track_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Riassume ogni pista di test attraverso i diversi training seed.

    Questo output aiuta a individuare piste sistematicamente facili/difficili
    e a scegliere esempi rappresentativi per video e discussione.
    """

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["evaluation_seed"])].append(row)

    output: list[dict[str, Any]] = []

    for (algorithm, evaluation_seed), group in sorted(grouped.items()):
        rewards = [row["episode_reward"] for row in group]
        completions = [float(row["completed"]) for row in group]
        track_completion = [row["track_completion"] for row in group]

        output.append(
            {
                "algorithm": algorithm,
                "evaluation_seed": evaluation_seed,
                "n_training_seeds": len(group),
                "mean_reward": mean(rewards),
                "std_reward_between_training_seeds": sample_std(rewards),
                "median_reward": median(rewards),
                "completion_rate": mean(completions),
                "mean_track_completion": mean(track_completion),
                "std_track_completion_between_training_seeds": sample_std(
                    track_completion
                ),
            }
        )

    return output


def build_paired_seed_comparison(
    per_run_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """
    Confronta PPO e DQN usando lo STESSO training seed.

    Le differenze sono definite come PPO - DQN. Un valore positivo del reward
    significa quindi che PPO ha avuto reward medio maggiore su quel seed.
    """

    lookup = {
        (row["algorithm"], row["training_seed"]): row
        for row in per_run_rows
    }

    output: list[dict[str, Any]] = []

    for seed in args.training_seeds:
        ppo = lookup.get(("ppo", seed))
        dqn = lookup.get(("dqn", seed))

        if ppo is None or dqn is None:
            fail_or_warn(
                False,
                f"Confronto appaiato impossibile per training seed {seed}.",
                allow_incomplete=args.allow_incomplete,
                warnings=warnings,
            )
            continue

        output.append(
            {
                "training_seed": seed,
                "ppo_mean_reward": ppo["mean_reward"],
                "dqn_mean_reward": dqn["mean_reward"],
                "ppo_minus_dqn_mean_reward": (
                    ppo["mean_reward"] - dqn["mean_reward"]
                ),
                "ppo_completion_rate": ppo["completion_rate"],
                "dqn_completion_rate": dqn["completion_rate"],
                "ppo_minus_dqn_completion_rate": (
                    ppo["completion_rate"] - dqn["completion_rate"]
                ),
                "ppo_mean_track_completion": ppo["mean_track_completion"],
                "dqn_mean_track_completion": dqn["mean_track_completion"],
                "ppo_minus_dqn_track_completion": (
                    ppo["mean_track_completion"] - dqn["mean_track_completion"]
                ),
            }
        )

    return output


# ---------------------------------------------------------------------------
# TRAINING FINALE: DISCOVERY DEI RUN E CONTROLLI DI CONFIGURAZIONE
# ---------------------------------------------------------------------------


def discover_final_training_runs(
    args: argparse.Namespace,
    warnings: list[str],
) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    """
    Trova i run finali leggendo i config.json, non affidandosi solo ai nomi.

    Questo e' piu' robusto di una regex sul nome della cartella e permette di
    verificare direttamente run_type, algoritmo, seed e protocollo validation.
    """

    if not args.logs_dir.exists():
        fail_or_warn(
            False,
            f"Cartella logs non trovata: {args.logs_dir}",
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )
        return {}

    official_seeds = set(args.training_seeds)
    found: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}

    # I config principali sono direttamente dentro logs/<run_name>/config.json.
    for config_path in sorted(args.logs_dir.glob("*/config.json")):
        config = read_json(config_path)

        if config.get("run_type") != "final":
            continue

        algorithm = str(config.get("algo", "")).lower()
        seed = config.get("seed")

        if algorithm not in ALGORITHMS:
            continue

        if not isinstance(seed, int) or seed not in official_seeds:
            continue

        key = (algorithm, seed)

        if key in found:
            previous = found[key][0]
            raise ValueError(
                f"Due run finali per {key}: {previous} e {config_path.parent}. "
                "L'aggregatore non puo' scegliere arbitrariamente quale usare."
            )

        found[key] = (config_path.parent, config)

    expected = {
        (algorithm, seed)
        for algorithm in ALGORITHMS
        for seed in args.training_seeds
    }

    fail_or_warn(
        set(found) == expected,
        (
            "Run finali nei log incompleti. "
            f"Attesi={sorted(expected)}, trovati={sorted(found)}"
        ),
        allow_incomplete=args.allow_incomplete,
        warnings=warnings,
    )

    return found


def validate_final_training_configs(
    runs: dict[tuple[str, int], tuple[Path, dict[str, Any]]],
    args: argparse.Namespace,
    warnings: list[str],
) -> None:
    """Verifica che i run ufficiali siano realmente confrontabili."""

    if not runs:
        return

    expected_validation = list(args.validation_seeds)

    # Elementi che DEVONO essere identici per tutti i run, PPO e DQN inclusi.
    environment_configs: set[str] = set()
    software_configs: set[str] = set()
    target_timesteps: set[int] = set()
    eval_frequencies: set[int] = set()

    # Gli iperparametri devono essere identici tra seed DELLO STESSO algoritmo.
    model_configs_by_algo: dict[str, set[str]] = defaultdict(set)

    for (algorithm, seed), (run_dir, config) in sorted(runs.items()):
        evaluation = config.get("evaluation", {})

        actual_validation = evaluation.get("eval_seeds")
        fail_or_warn(
            actual_validation == expected_validation,
            (
                f"Validation seed non coerenti in {run_dir}: "
                f"attesi={expected_validation}, trovati={actual_validation}"
            ),
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )

        deterministic = evaluation.get("deterministic")
        fail_or_warn(
            deterministic is True,
            f"Validation non deterministica nel run {run_dir}.",
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )

        target = config.get("target_timesteps")
        if not isinstance(target, int):
            raise ValueError(f"target_timesteps non valido in {run_dir}: {target!r}")
        target_timesteps.add(target)

        eval_freq = evaluation.get("eval_freq")
        if not isinstance(eval_freq, int):
            raise ValueError(f"eval_freq non valido in {run_dir}: {eval_freq!r}")
        eval_frequencies.add(eval_freq)

        environment_configs.add(canonical_json(config.get("environment")))
        software_configs.add(canonical_json(config.get("software")))
        model_configs_by_algo[algorithm].add(canonical_json(config.get("model")))

    fail_or_warn(
        len(target_timesteps) == 1,
        f"Budget target differenti tra run finali: {sorted(target_timesteps)}",
        allow_incomplete=args.allow_incomplete,
        warnings=warnings,
    )
    fail_or_warn(
        len(eval_frequencies) == 1,
        f"eval_freq differenti tra run finali: {sorted(eval_frequencies)}",
        allow_incomplete=args.allow_incomplete,
        warnings=warnings,
    )
    fail_or_warn(
        len(environment_configs) == 1,
        "Configurazione environment diversa tra run finali.",
        allow_incomplete=args.allow_incomplete,
        warnings=warnings,
    )
    fail_or_warn(
        len(software_configs) == 1,
        "Versioni software diverse tra run finali.",
        allow_incomplete=args.allow_incomplete,
        warnings=warnings,
    )

    for algorithm, configs in model_configs_by_algo.items():
        fail_or_warn(
            len(configs) == 1,
            f"Iperparametri {algorithm.upper()} diversi tra training seed finali.",
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# TRAINING FINALE: LEARNING CURVE E BEST MODEL
# ---------------------------------------------------------------------------


def normalize_learning_curve_row(
    raw: dict[str, str],
    path: Path,
) -> dict[str, Any]:
    """Converte una riga evaluations.csv di train.py in tipi numerici."""

    return {
        "timesteps": parse_int(raw["timesteps"], field="timesteps", path=path),
        "eval_seed": parse_int(raw["eval_seed"], field="eval_seed", path=path),
        "episode_reward": parse_float(
            raw["episode_reward"], field="episode_reward", path=path
        ),
        "episode_length": parse_int(
            raw["episode_length"], field="episode_length", path=path
        ),
        "source_file": str(path),
    }


def parse_compact_timesteps_label(label: str) -> int:
    """
    Converte le etichette create da format_timesteps() in numeri interi.

    Esempi:
        "400k" -> 400000
        "1M"   -> 1000000
        "750"  -> 750

    Serve solo come fallback se una cartella resume non contiene ancora un
    run_summary.json da cui leggere start_timesteps.
    """

    text = label.strip()

    if text.lower().endswith("k"):
        return int(text[:-1]) * 1_000

    if text.lower().endswith("m"):
        return int(text[:-1]) * 1_000_000

    return int(text)


def get_segment_start_timestep(segment_dir: Path, run_dir: Path) -> int:
    """
    Determina da quale timestep parte un segmento di training/log.

    - Il log principale del run parte da 0.
    - Un log resume_<N> parte dal valore start_timesteps scritto nel suo
      run_summary.json.
    - Se il summary non e' disponibile, usa come fallback il nome resume_<N>.
    """

    if segment_dir == run_dir:
        return 0

    summary_path = segment_dir / "run_summary.json"

    if summary_path.is_file():
        summary = read_json(summary_path)
        if "start_timesteps" in summary:
            return int(summary["start_timesteps"])

    prefix = "resume_"
    if segment_dir.name.startswith(prefix):
        label = segment_dir.name[len(prefix):]
        try:
            return parse_compact_timesteps_label(label)
        except ValueError as error:
            raise ValueError(
                f"Impossibile ricavare il timestep di resume da {segment_dir}"
            ) from error

    raise ValueError(
        f"Cartella evaluations.csv non riconosciuta come segmento: {segment_dir}"
    )


def load_run_learning_curve(
    run_dir: Path,
    expected_validation_seeds_set: set[int],
    *,
    allow_incomplete: bool,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Legge le evaluation del training e ricostruisce la traiettoria effettiva.

    Perche' serve una logica speciale per i resume?
    ------------------------------------------------
    Immaginiamo:
        training originale -> arriva a 470k
        ultimo recovery     -> 400k
        crash
        resume              -> riparte da 400k

    Il vecchio evaluations.csv puo' contenere punti a 425k e 450k che
    appartengono al ramo di training abbandonato dopo il crash. Il nuovo ramo
    produrra' nuove evaluation a 425k e 450k. Non dobbiamo mediare i due rami.

    Regola usata:
    - quando entra un segmento resume da S, vengono scartati dalla learning
      curve corrente tutti i punti con timestep > S;
    - vengono poi inseriti i punti del nuovo segmento;
    - eventuali resume successivi applicano la stessa regola.

    La funzione restituisce DUE liste:
    1) stitched_rows: traiettoria finale ricostruita, usata per i grafici;
    2) all_observed_rows: tutte le evaluation osservate, inclusi rami poi
       abbandonati. Servono per verificare correttamente best_model_info,
       perche' un best model salvato prima del crash puo' essere ancora valido.
    """

    paths = sorted(run_dir.rglob("evaluations.csv"))

    fail_or_warn(
        bool(paths),
        f"Nessun evaluations.csv trovato nel run {run_dir}.",
        allow_incomplete=allow_incomplete,
        warnings=warnings,
    )

    # Ogni evaluations.csv viene trattato come un segmento indipendente.
    segments: list[tuple[int, Path, list[dict[str, Any]]]] = []
    all_observed_rows: list[dict[str, Any]] = []

    for path in paths:
        segment_dir = path.parent
        segment_start = get_segment_start_timestep(segment_dir, run_dir)
        raw_rows = read_dict_csv(path, REQUIRED_LEARNING_CURVE_COLUMNS)
        segment_rows: list[dict[str, Any]] = []

        for raw in raw_rows:
            row = normalize_learning_curve_row(raw, path)
            row["segment_start_timesteps"] = segment_start
            row["segment_name"] = (
                "initial" if segment_dir == run_dir else segment_dir.name
            )

            if row["eval_seed"] not in expected_validation_seeds_set:
                fail_or_warn(
                    False,
                    (
                        f"Validation seed inatteso {row['eval_seed']} "
                        f"in {path}."
                    ),
                    allow_incomplete=allow_incomplete,
                    warnings=warnings,
                )
                continue

            segment_rows.append(row)
            all_observed_rows.append(row.copy())

        segments.append((segment_start, path, segment_rows))

    # Ordina cronologicamente i segmenti: iniziale, poi resume progressivi.
    segments.sort(key=lambda item: (item[0], str(item[1])))

    stitched: dict[tuple[int, int], dict[str, Any]] = {}

    for segment_start, path, segment_rows in segments:
        if segment_start > 0:
            # Tutto cio' che era stato osservato DOPO il punto di recovery
            # appartiene a un ramo abbandonato e non va nella curva finale.
            obsolete_keys = [
                key
                for key in stitched
                if key[0] > segment_start
            ]
            for key in obsolete_keys:
                del stitched[key]

        # All'interno dello stesso segmento una coppia timestep/eval_seed non
        # deve comparire due volte.
        segment_seen: set[tuple[int, int]] = set()

        for row in segment_rows:
            key = (row["timesteps"], row["eval_seed"])

            if key in segment_seen:
                raise ValueError(
                    f"Evaluation duplicata nello stesso file/segmento {path}: {key}"
                )
            segment_seen.add(key)

            # Se il nuovo segmento contiene anche un punto esattamente alla
            # soglia di resume, il ramo nuovo e' quello autorevole.
            stitched[key] = row

    return (
        [stitched[key] for key in sorted(stitched)],
        all_observed_rows,
    )


def best_info_matches_observed_evaluation(
    best: dict[str, Any],
    observed_rows: list[dict[str, Any]],
    expected_validation_seeds_set: set[int],
) -> bool:
    """
    Controlla se best_model_info corrisponde a una evaluation realmente vista.

    Le righe vengono raggruppate per segmento+timestep, non soltanto timestep:
    dopo un resume due rami diversi possono avere entrambi una evaluation a
    425k ma con reward differenti.
    """

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for row in observed_rows:
        grouped[(row["segment_name"], row["timesteps"])].append(row)

    for (_, timestep), group in grouped.items():
        if timestep != best["timesteps"]:
            continue

        seeds = {row["eval_seed"] for row in group}
        if seeds != expected_validation_seeds_set:
            continue

        candidate_mean = mean(row["episode_reward"] for row in group)

        if math.isclose(
            candidate_mean,
            best["mean_reward"],
            rel_tol=1e-8,
            abs_tol=1e-5,
        ):
            return True

    return False


def build_run_learning_curve(
    algorithm: str,
    training_seed: int,
    raw_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Riduce le 5 piste di validation a una riga per timestep e run."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[row["timesteps"]].append(row)

    expected_validation = set(args.validation_seeds)
    output: list[dict[str, Any]] = []

    for timestep, group in sorted(grouped.items()):
        actual_validation = {row["eval_seed"] for row in group}

        fail_or_warn(
            actual_validation == expected_validation,
            (
                f"Validation incompleta per {algorithm} seed={training_seed} "
                f"a timestep={timestep}. Attesi={sorted(expected_validation)}, "
                f"trovati={sorted(actual_validation)}"
            ),
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )

        rewards = [row["episode_reward"] for row in group]
        lengths = [float(row["episode_length"]) for row in group]

        output.append(
            {
                "algorithm": algorithm,
                "training_seed": training_seed,
                "timesteps": timestep,
                "n_validation_tracks": len(group),
                "mean_validation_reward": mean(rewards),
                # ddof=0 per coerenza con il callback di train.py.
                "std_validation_reward_across_tracks": population_std(rewards),
                "median_validation_reward": median(rewards),
                "min_validation_reward": float(min(rewards)),
                "max_validation_reward": float(max(rewards)),
                "mean_validation_episode_length": mean(lengths),
            }
        )

    return output


def load_best_model_info(run_dir: Path) -> dict[str, Any] | None:
    """
    Recupera il best_model_info piu' forte tra run principale e resume.

    Durante un resume train.py puo' scrivere un nuovo best_model_info.json in
    una sottocartella. Scegliere il massimo mean_reward ricostruisce il best
    globale senza dipendere dalla cartella best_models/ (che e' gitignored).
    """

    candidates: list[dict[str, Any]] = []

    for path in sorted(run_dir.rglob("best_model_info.json")):
        info = read_json(path)

        try:
            candidates.append(
                {
                    "timesteps": int(info["timesteps"]),
                    "mean_reward": float(info["mean_reward"]),
                    "std_reward": float(info["std_reward"]),
                    "eval_seeds": list(info["eval_seeds"]),
                    "source_file": str(path),
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"best_model_info non valido: {path}") from error

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["mean_reward"])


def load_run_summaries(run_dir: Path) -> list[dict[str, Any]]:
    """Legge tutti i run_summary.json, inclusi quelli delle sessioni resume."""

    summaries: list[dict[str, Any]] = []

    for path in sorted(run_dir.rglob("run_summary.json")):
        summary = read_json(path)
        summary["source_file"] = str(path)
        summaries.append(summary)

    return summaries


def build_training_outputs(
    runs: dict[tuple[str, int], tuple[Path, dict[str, Any]]],
    args: argparse.Namespace,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Costruisce:
    - training_run_summary.csv
    - learning_curve_per_run.csv
    """

    training_summaries: list[dict[str, Any]] = []
    all_curve_rows: list[dict[str, Any]] = []
    expected_validation = set(args.validation_seeds)

    for (algorithm, training_seed), (run_dir, config) in sorted(runs.items()):
        raw_curve, all_observed_curve_rows = load_run_learning_curve(
            run_dir,
            expected_validation,
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )

        curve = build_run_learning_curve(
            algorithm,
            training_seed,
            raw_curve,
            args,
            warnings,
        )
        all_curve_rows.extend(curve)

        best = load_best_model_info(run_dir)
        fail_or_warn(
            best is not None,
            f"best_model_info.json mancante per {run_dir}.",
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )

        if best is not None:
            fail_or_warn(
                best["eval_seeds"] == list(args.validation_seeds),
                (
                    f"Best model {run_dir} selezionato su validation seed "
                    f"diversi: {best['eval_seeds']}"
                ),
                allow_incomplete=args.allow_incomplete,
                warnings=warnings,
            )

            # Il best puo' provenire anche da un ramo osservato prima di un
            # crash e poi abbandonato dal resume. Verifichiamo quindi che
            # corrisponda a QUALCHE evaluation realmente osservata, senza
            # imporre che sia per forza nella learning curve finale stitched.
            matches_observed = best_info_matches_observed_evaluation(
                best,
                all_observed_curve_rows,
                expected_validation,
            )

            fail_or_warn(
                matches_observed,
                (
                    f"best_model_info non corrisponde a nessuna evaluation "
                    f"osservata per {run_dir}: "
                    f"({best['timesteps']}, {best['mean_reward']})"
                ),
                allow_incomplete=args.allow_incomplete,
                warnings=warnings,
            )

        session_summaries = load_run_summaries(run_dir)

        # Conta le sessioni reali anche attraverso le cartelle di resume.
        resume_dirs = [
            path
            for path in run_dir.iterdir()
            if path.is_dir()
            and path.name.startswith("resume_")
        ]

        # Ogni run ha una sessione iniziale, più una per ogni resume.
        training_sessions = 1 + len(resume_dirs)

        # Numero di sessioni per cui possediamo effettivamente un run_summary.json.
        session_summaries_found = len(
            session_summaries
        )

        # Il tempo totale è completo solo se possediamo
        # il summary di ogni sessione.
        elapsed_time_complete = (
            session_summaries_found
            == training_sessions
        )

        # Sommare i tempi delle sessioni permette di ricostruire il costo totale
        # anche quando un training e' stato interrotto e poi ripreso.
        elapsed_total = 0.0
        actual_timesteps_values: list[int] = []
        statuses: list[str] = []

        for summary in session_summaries:
            elapsed_value = summary.get(
                "elapsed_seconds_this_session",
                summary.get("elapsed_seconds", 0.0),
            )
            try:
                elapsed_total += float(elapsed_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Tempo non valido in {summary.get('source_file')}"
                ) from error

            if "actual_timesteps" in summary:
                actual_timesteps_values.append(int(summary["actual_timesteps"]))

            if "status" in summary:
                statuses.append(str(summary["status"]))

        target_timesteps = int(config["target_timesteps"])
        actual_timesteps = (
            max(actual_timesteps_values)
            if actual_timesteps_values
            else None
        )

        # Al termine di un esperimento ufficiale deve esistere almeno una
        # sessione completata. Un resume puo' lasciare anche summary "interrupted".
        if session_summaries:
            fail_or_warn(
                "completed" in statuses,
                f"Nessuna sessione completed nel run finale {run_dir}.",
                allow_incomplete=args.allow_incomplete,
                warnings=warnings,
            )

        training_summaries.append(
            {
                "algorithm": algorithm,
                "training_seed": training_seed,
                "run_name": run_dir.name,
                "target_timesteps": target_timesteps,
                "actual_timesteps": actual_timesteps,
                "evaluation_frequency": config["evaluation"]["eval_freq"],
                "n_validation_tracks": len(args.validation_seeds),
                "n_evaluation_points": len(curve),
                "last_evaluation_timestep": (
                    max(row["timesteps"] for row in curve) if curve else None
                ),
                "best_timestep": best["timesteps"] if best else None,
                "best_mean_validation_reward": best["mean_reward"] if best else None,
                "best_std_validation_reward": best["std_reward"] if best else None,
                "training_sessions": training_sessions,
                "session_summaries_found": session_summaries_found,
                "elapsed_seconds_total": (
                    elapsed_total
                    if elapsed_time_complete
                    else None
                ),
                "elapsed_time_complete": int(
                    elapsed_time_complete
                ),
            }
        )

    return training_summaries, all_curve_rows


def build_learning_curve_algorithm(
    per_run_curve: list[dict[str, Any]],
    args: argparse.Namespace,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """
    Aggrega la learning curve tra training seed.

    Prima ogni run e' gia' stato ridotto alla media sulle piste 200-204.
    Solo DOPO queste medie vengono aggregate tra seed. Questo mantiene chiara
    la distinzione tra variabilita' delle piste e variabilita' dei training.
    """

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_run_curve:
        grouped[(row["algorithm"], row["timesteps"])].append(row)

    expected_training_seed_set = set(args.training_seeds)
    output: list[dict[str, Any]] = []

    for (algorithm, timestep), group in sorted(grouped.items()):
        present_seeds = {row["training_seed"] for row in group}

        fail_or_warn(
            present_seeds == expected_training_seed_set,
            (
                f"Learning curve {algorithm} a timestep={timestep} non contiene "
                f"tutti i training seed. Attesi={sorted(expected_training_seed_set)}, "
                f"trovati={sorted(present_seeds)}"
            ),
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )

        run_means = [row["mean_validation_reward"] for row in group]

        output.append(
            {
                "algorithm": algorithm,
                "timesteps": timestep,
                "n_training_seeds": len(group),
                "mean_validation_reward_across_training_seeds": mean(run_means),
                "std_validation_reward_between_training_seeds": sample_std(run_means),
                "median_validation_reward_across_training_seeds": median(run_means),
                "min_validation_reward_across_training_seeds": float(min(run_means)),
                "max_validation_reward_across_training_seeds": float(max(run_means)),
            }
        )

    # Verifica che PPO e DQN abbiano una griglia temporale confrontabile.
    timesteps_by_algo: dict[str, set[int]] = defaultdict(set)
    for row in output:
        timesteps_by_algo[row["algorithm"]].add(row["timesteps"])

    if all(algorithm in timesteps_by_algo for algorithm in ALGORITHMS):
        fail_or_warn(
            timesteps_by_algo["ppo"] == timesteps_by_algo["dqn"],
            (
                "PPO e DQN non hanno la stessa griglia di timestep nelle "
                "learning curve finali."
            ),
            allow_incomplete=args.allow_incomplete,
            warnings=warnings,
        )

    return output


# ---------------------------------------------------------------------------
# SCRITTURA DEGLI OUTPUT
# ---------------------------------------------------------------------------


def save_all_outputs(
    args: argparse.Namespace,
    warnings: list[str],
    test_episodes: list[dict[str, Any]],
    per_run_test: list[dict[str, Any]],
    algorithm_test: list[dict[str, Any]],
    per_track_test: list[dict[str, Any]],
    paired_test: list[dict[str, Any]],
    training_summary: list[dict[str, Any]],
    per_run_curve: list[dict[str, Any]],
    algorithm_curve: list[dict[str, Any]],
) -> dict[str, str]:
    """Salva tutti i CSV in una cartella unica e restituisce i loro percorsi."""

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Episodi finali combinati
    # ------------------------------------------------------------------
    combined_path = output_dir / "combined_test_episodes.csv"
    combined_rows = sorted(
        test_episodes,
        key=lambda row: (
            row["algorithm"],
            row["training_seed"],
            row["evaluation_seed"],
        ),
    )
    write_csv(
        combined_path,
        combined_rows,
        [
            "algorithm",
            "training_seed",
            "evaluation_seed",
            "episode",
            "episode_reward",
            "episode_length",
            "completed",
            "termination_reason",
            "visited_tiles",
            "total_tiles",
            "track_completion",
            "lap_complete_percent",
            "terminated",
            "truncated",
            "deterministic",
            "model_path",
            "source_file",
        ],
    )
    outputs["combined_test_episodes"] = str(combined_path)

    # ------------------------------------------------------------------
    # Test per run
    # ------------------------------------------------------------------
    per_run_path = output_dir / "test_per_run_summary.csv"
    write_csv(
        per_run_path,
        per_run_test,
        [
            "algorithm",
            "training_seed",
            "episodes",
            "mean_reward",
            "std_reward",
            "median_reward",
            "min_reward",
            "max_reward",
            "completion_rate",
            "mean_track_completion",
            "std_track_completion",
            "mean_episode_length",
            "std_episode_length",
            "lap_completed_rate",
            "out_of_bounds_rate",
            "time_limit_rate",
            "unknown_termination_rate",
        ],
    )
    outputs["test_per_run_summary"] = str(per_run_path)

    # ------------------------------------------------------------------
    # Test aggregato tra training seed
    # ------------------------------------------------------------------
    algorithm_path = output_dir / "test_algorithm_summary.csv"
    write_csv(
        algorithm_path,
        algorithm_test,
        [
            "algorithm",
            "n_training_seeds",
            "n_test_episodes_total",
            "mean_reward_across_training_seeds",
            "std_reward_between_training_seeds",
            "median_reward_across_training_seeds",
            "min_run_mean_reward",
            "max_run_mean_reward",
            "mean_completion_rate_across_training_seeds",
            "std_completion_rate_between_training_seeds",
            "mean_track_completion_across_training_seeds",
            "std_track_completion_between_training_seeds",
            "mean_episode_length_across_training_seeds",
            "std_episode_length_between_training_seeds",
            "pooled_episode_mean_reward",
            "pooled_episode_std_reward",
            "pooled_completion_rate",
            "pooled_mean_track_completion",
        ],
    )
    outputs["test_algorithm_summary"] = str(algorithm_path)

    # ------------------------------------------------------------------
    # Test per pista
    # ------------------------------------------------------------------
    per_track_path = output_dir / "test_per_track_summary.csv"
    write_csv(
        per_track_path,
        per_track_test,
        [
            "algorithm",
            "evaluation_seed",
            "n_training_seeds",
            "mean_reward",
            "std_reward_between_training_seeds",
            "median_reward",
            "completion_rate",
            "mean_track_completion",
            "std_track_completion_between_training_seeds",
        ],
    )
    outputs["test_per_track_summary"] = str(per_track_path)

    # ------------------------------------------------------------------
    # Confronto PPO-DQN appaiato per training seed
    # ------------------------------------------------------------------
    paired_path = output_dir / "test_paired_seed_comparison.csv"
    write_csv(
        paired_path,
        paired_test,
        [
            "training_seed",
            "ppo_mean_reward",
            "dqn_mean_reward",
            "ppo_minus_dqn_mean_reward",
            "ppo_completion_rate",
            "dqn_completion_rate",
            "ppo_minus_dqn_completion_rate",
            "ppo_mean_track_completion",
            "dqn_mean_track_completion",
            "ppo_minus_dqn_track_completion",
        ],
    )
    outputs["test_paired_seed_comparison"] = str(paired_path)

    # ------------------------------------------------------------------
    # Riepilogo dei training finali
    # ------------------------------------------------------------------
    training_path = output_dir / "training_run_summary.csv"
    write_csv(
        training_path,
        training_summary,
        [
            "algorithm",
            "training_seed",
            "run_name",
            "target_timesteps",
            "actual_timesteps",
            "evaluation_frequency",
            "n_validation_tracks",
            "n_evaluation_points",
            "last_evaluation_timestep",
            "best_timestep",
            "best_mean_validation_reward",
            "best_std_validation_reward",
            "training_sessions",
            "session_summaries_found",
            "elapsed_seconds_total",
            "elapsed_time_complete",
        ],
    )
    outputs["training_run_summary"] = str(training_path)

    # ------------------------------------------------------------------
    # Learning curve: un valore per run/timestep
    # ------------------------------------------------------------------
    per_run_curve_path = output_dir / "learning_curve_per_run.csv"
    write_csv(
        per_run_curve_path,
        per_run_curve,
        [
            "algorithm",
            "training_seed",
            "timesteps",
            "n_validation_tracks",
            "mean_validation_reward",
            "std_validation_reward_across_tracks",
            "median_validation_reward",
            "min_validation_reward",
            "max_validation_reward",
            "mean_validation_episode_length",
        ],
    )
    outputs["learning_curve_per_run"] = str(per_run_curve_path)

    # ------------------------------------------------------------------
    # Learning curve aggregata tra seed: input ideale per plot_results.py
    # ------------------------------------------------------------------
    algorithm_curve_path = output_dir / "learning_curve_algorithm.csv"
    write_csv(
        algorithm_curve_path,
        algorithm_curve,
        [
            "algorithm",
            "timesteps",
            "n_training_seeds",
            "mean_validation_reward_across_training_seeds",
            "std_validation_reward_between_training_seeds",
            "median_validation_reward_across_training_seeds",
            "min_validation_reward_across_training_seeds",
            "max_validation_reward_across_training_seeds",
        ],
    )
    outputs["learning_curve_algorithm"] = str(algorithm_curve_path)

    return outputs


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Esegue l'intera pipeline di aggregazione.

    Ordine intenzionale:
    1. legge e valida i test finali;
    2. calcola statistiche finali;
    3. legge e valida i training ufficiali;
    4. ricostruisce le learning curve;
    5. salva output e report di integrita'.
    """

    args = parse_args()
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1) TEST FINALE
    # ------------------------------------------------------------------
    test_episodes = load_final_test_episodes(args, warnings)
    validate_final_test_episodes(test_episodes, args, warnings)

    per_run_test = build_test_per_run_summary(test_episodes) if test_episodes else []
    algorithm_test = (
        build_test_algorithm_summary(per_run_test, test_episodes)
        if per_run_test
        else []
    )
    per_track_test = build_test_per_track_summary(test_episodes) if test_episodes else []
    paired_test = (
        build_paired_seed_comparison(per_run_test, args, warnings)
        if per_run_test
        else []
    )

    # ------------------------------------------------------------------
    # 2) TRAINING / VALIDATION
    # ------------------------------------------------------------------
    runs = discover_final_training_runs(args, warnings)
    validate_final_training_configs(runs, args, warnings)

    training_summary, per_run_curve = build_training_outputs(
        runs,
        args,
        warnings,
    )

    algorithm_curve = (
        build_learning_curve_algorithm(per_run_curve, args, warnings)
        if per_run_curve
        else []
    )

    # ------------------------------------------------------------------
    # 3) OUTPUT
    # ------------------------------------------------------------------
    outputs = save_all_outputs(
        args,
        warnings,
        test_episodes,
        per_run_test,
        algorithm_test,
        per_track_test,
        paired_test,
        training_summary,
        per_run_curve,
        algorithm_curve,
    )

    report = {
        "aggregation_version": 1,
        "strict_mode": not args.allow_incomplete,
        "protocol": {
            "algorithms": list(ALGORITHMS),
            "training_seeds": list(args.training_seeds),
            "validation_seeds": list(args.validation_seeds),
            "test_seeds": expected_test_seeds(args),
            "require_best_model_for_test": not args.allow_non_best_model,
            "require_deterministic_test": not args.allow_stochastic,
        },
        "counts": {
            "final_training_runs": len(runs),
            "test_episodes": len(test_episodes),
            "test_run_summaries": len(per_run_test),
            "learning_curve_run_points": len(per_run_curve),
            "learning_curve_algorithm_points": len(algorithm_curve),
        },
        "warnings": warnings,
        "outputs": outputs,
    }

    report_path = args.output_dir / "aggregation_report.json"
    write_json(report_path, report)

    # Messaggio finale breve ma utile: se siamo in strict mode e siamo arrivati
    # qui, tutte le verifiche bloccanti sono state superate.
    print()
    print("Aggregazione completata.")
    print(f"Output: {args.output_dir}")
    print(f"Training finali trovati: {len(runs)}")
    print(f"Episodi di test trovati: {len(test_episodes)}")
    print(f"Warning: {len(warnings)}")

    if warnings:
        print(
            "ATTENZIONE: l'aggregazione contiene warning. "
            "Leggere aggregation_report.json prima di usare i risultati."
        )
    else:
        print("Controlli di integrita' superati senza warning.")


if __name__ == "__main__":
    main()
