import argparse
import csv
import os
import json
import platform
import time
from importlib import metadata
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from env import (
    make_env,
    ENV_ID,
    CONTINUOUS,
    LAP_COMPLETE_PERCENT,
    DOMAIN_RANDOMIZE,
    MAX_EPISODE_STEPS,
)

POLICY = "CnnPolicy"
DEVICE = "cpu"
DQN_BUFFER_SIZE = 10_000
DQN_LEARNING_STARTS = 500
DQN_EXPLORATION_FRACTION = 0.1

# Funzione che converte il numero di timesteps in un'etichetta compatta
def format_timesteps(timesteps: int) -> str:
    if timesteps % 1_000_000 == 0:
        return f"{timesteps // 1_000_000}M"

    if timesteps % 1_000 == 0:
        return f"{timesteps // 1_000}k"

    return str(timesteps)

# Restituisce le versioni software utilizzate nell'esperimento
def get_software_versions():
    return {
        "python": platform.python_version(),
        "numpy": metadata.version("numpy"),
        "gymnasium": metadata.version("gymnasium"),
        "stable_baselines3": metadata.version(
            "stable-baselines3"
        ),
        "torch": metadata.version("torch"),
    }

# Restituisce gli iperparametri effettivamente utilizzati dal modello
def get_model_hyperparameters(model, algo):

    if algo == "ppo":
        return {
            "learning_rate": float(model.learning_rate),
            "n_steps": int(model.n_steps),
            "batch_size": int(model.batch_size),
            "n_epochs": int(model.n_epochs),
            "gamma": float(model.gamma),
            "gae_lambda": float(model.gae_lambda),
            "clip_range": float(model.clip_range(1.0)),
            "clip_range_vf": (
                None
                if model.clip_range_vf is None
                else float(model.clip_range_vf(1.0))
            ),
            "normalize_advantage": bool(
                model.normalize_advantage
            ),
            "ent_coef": float(model.ent_coef),
            "vf_coef": float(model.vf_coef),
            "max_grad_norm": float(model.max_grad_norm),
            "use_sde": bool(model.use_sde),
            "sde_sample_freq": int(model.sde_sample_freq),
            "target_kl": (
                None
                if model.target_kl is None
                else float(model.target_kl)
            ),
        }

    return {
        "learning_rate": float(model.learning_rate),
        "buffer_size": int(model.buffer_size),
        "learning_starts": int(model.learning_starts),
        "batch_size": int(model.batch_size),
        "tau": float(model.tau),
        "gamma": float(model.gamma),
        "train_freq": {
            "frequency": int(model.train_freq.frequency),
            "unit": model.train_freq.unit.value,
        },
        "gradient_steps": int(model.gradient_steps),
        "optimize_memory_usage": bool(
            model.optimize_memory_usage
        ),
        "target_update_interval": int(
            model.target_update_interval
        ),
        "exploration_fraction": float(
            model.exploration_fraction
        ),
        "exploration_initial_eps": float(
            model.exploration_initial_eps
        ),
        "exploration_final_eps": float(
            model.exploration_final_eps
        ),
        "max_grad_norm": float(model.max_grad_norm),
    }

# Callback di evaluation con seed fissi
class FixedSeedEvalCallback(BaseCallback):

    def __init__(
        self,
        eval_env,
        eval_freq,
        eval_seeds,
        log_path,
        best_model_save_path,
        best_model_name,
        deterministic=True,
        verbose=1,
    ):
        super().__init__(verbose)

        # Parametri necessari per l'evaluation periodica
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_seeds = eval_seeds
        self.log_path = log_path
        self.deterministic = deterministic

        # Cartella e nome utilizzati per salvare il miglior modello
        self.best_model_save_path = best_model_save_path
        self.best_model_name = best_model_name

        # Miglior reward medio osservato durante le evaluation
        self.best_mean_reward = -np.inf

        # File contenente le informazioni sul miglior modello
        self.best_info_path = os.path.join(
            best_model_save_path,
            "best_model_info.json",
        )

        self.best_log_info_path = os.path.join(
            log_path,
            "best_model_info.json",
        )

        # File in cui salvare i risultati delle evaluation
        self.csv_path = os.path.join(
            log_path,
            "evaluations.csv",
        )

    def _init_callback(self):
        # Crea la cartella dei log, se necessario
        os.makedirs(
            self.log_path,
            exist_ok=True,
        )

        # Crea la cartella del best model, se necessario
        os.makedirs(
            self.best_model_save_path,
            exist_ok=True,
        )

        # Se il training viene ripreso, recupera il miglior reward precedente
        if os.path.isfile(self.best_info_path):
            with open(
                self.best_info_path,
                "r",
                encoding="utf-8",
            ) as file:
                best_info = json.load(file)

            self.best_mean_reward = float(
                best_info["mean_reward"]
            )

        # Crea il CSV con la relativa intestazione
        with open(
            self.csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "timesteps",
                "eval_seed",
                "episode_reward",
                "episode_length",
            ])

    def _run_episode(self, seed):
        # Imposta il seed della pista per questo episodio
        self.eval_env.seed(seed)
        obs = self.eval_env.reset()

        episode_reward = 0.0
        episode_length = 0
        done = False
        final_info = None

        while not done:
            # Il modello sceglie l'azione da eseguire
            action, _ = self.model.predict(
                obs,
                deterministic=self.deterministic,
            )

            obs, rewards, dones, infos = self.eval_env.step(action)

            episode_reward += float(rewards[0])
            episode_length += 1
            done = bool(dones[0])

            if done:
                final_info = infos[0]

        # Usa le statistiche registrate dal Monitor, se disponibili
        if final_info is not None and "episode" in final_info:
            episode_reward = float(
                final_info["episode"]["r"]
            )
            episode_length = int(
                final_info["episode"]["l"]
            )

        return episode_reward, episode_length

    def _on_step(self):
        # Esegue l'evaluation solo alla frequenza stabilita
        if self.num_timesteps % self.eval_freq != 0:
            return True

        episode_rewards = []
        episode_lengths = []
        rows = []

        # Ripete sempre gli stessi seed ad ogni evaluation
        for seed in self.eval_seeds:
            reward, length = self._run_episode(seed)

            episode_rewards.append(reward)
            episode_lengths.append(length)

            rows.append([
                self.num_timesteps,
                seed,
                reward,
                length,
            ])

        # Salva i risultati dei singoli episodi
        with open(
            self.csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        # Calcola le statistiche dell'evaluation
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)

        # Salva il modello se ottiene la migliore evaluation vista finora
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = float(mean_reward)

            best_model_path = os.path.join(
                self.best_model_save_path,
                f"{self.best_model_name}_best",
            )

            self.model.save(
                best_model_path
            )

            best_info = {
                "timesteps": int(self.num_timesteps),
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "eval_seeds": self.eval_seeds,
            }

            with open(
                self.best_info_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                best_info,
                file,
                indent=4,
            )

            with open(
                self.best_log_info_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    best_info,
                    file,
                    indent=4,
                )

            if self.verbose >= 1:
                print(
                    f"Nuovo best model a {self.num_timesteps} timesteps: "
                    f"mean_reward={mean_reward:.2f}"
                )

        if self.verbose >= 1:
            print(
                f"Eval num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )

            print(
                f"Episode length: "
                f"{mean_length:.2f} +/- {std_length:.2f}"
            )

            print(
                f"Evaluation seeds: {self.eval_seeds}"
            )

        return True

# Salva checkpoint intermedi in base ai timesteps globali del training
class AnalysisCheckpointCallback(BaseCallback):

    def __init__(
        self,
        save_freq,
        save_path,
        name_prefix,
        verbose=1,
    ):
        super().__init__(verbose)

        # Frequenza con cui salvare i checkpoint
        self.save_freq = save_freq

        # Cartella in cui salvare i checkpoint
        self.save_path = save_path

        # Prefisso utilizzato per il nome dei file
        self.name_prefix = name_prefix

    def _init_callback(self):
        # Crea la cartella dei checkpoint, se necessario
        os.makedirs(
            self.save_path,
            exist_ok=True,
        )

    def _on_step(self):
        # Salva solo ai multipli globali della frequenza stabilita
        if self.num_timesteps % self.save_freq != 0:
            return True

        steps_label = format_timesteps(
            self.num_timesteps
        )

        checkpoint_path = os.path.join(
            self.save_path,
            f"{self.name_prefix}_checkpoint_{steps_label}",
        )

        # Salva solamente il modello, senza replay buffer
        self.model.save(
            checkpoint_path
        )

        if self.verbose >= 1:
            print(
                f"Checkpoint salvato a "
                f"{self.num_timesteps} timesteps"
            )

        return True

# Salva lo stato necessario per recuperare un training DQN interrotto
class RecoveryCallback(BaseCallback):

    def __init__(
        self,
        save_freq,
        save_path,
        verbose=1,
    ):
        super().__init__(verbose)

        # Frequenza con cui aggiornare il recovery
        self.save_freq = save_freq

        # Cartella in cui salvare modello e replay buffer
        self.save_path = save_path

    def _init_callback(self):
        # Crea la cartella di recovery, se necessario
        os.makedirs(
            self.save_path,
            exist_ok=True,
        )

    def _on_step(self):
        # Salva il recovery solo alla frequenza stabilita
        if self.num_timesteps % self.save_freq != 0:
            return True

        model_path = os.path.join(
            self.save_path,
            "recovery_model",
        )

        replay_buffer_path = os.path.join(
            self.save_path,
            "recovery_replay_buffer.pkl",
        )

        # Sovrascrive il precedente modello di recovery
        self.model.save(model_path)

        # Sovrascrive il precedente replay buffer
        self.model.save_replay_buffer(
            replay_buffer_path
        )

        if self.verbose >= 1:
            print(
                f"Recovery DQN salvato a "
                f"{self.num_timesteps} timesteps"
            )

        return True


def main():
    parser = argparse.ArgumentParser()

    # Parametri

    # Algoritmo di RL da utilizzare
    parser.add_argument(
        "--algo",
        choices=["ppo", "dqn"],
        required=True,
    )

    # Numero di environment steps richiesti per il training
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2000,
    )

    # Seed utilizzato per rendere il training riproducibile
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    # Tipo di training
    parser.add_argument(
        "--run-type",
        choices=["smoke", "pilot", "final"],
        required=True,
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Etichetta opzionale per distinguere varianti dello stesso esperimento.",
    )

    # Ogni quanti timesteps effettuare la valutazione del modello
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
    )

    # Seed delle piste utilizzate ad ogni evaluation
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=[100, 101, 102, 103, 104],
    )

    # Frequenza dei checkpoint; se omessa viene scelta in base al tipo di training
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
    )

    # Frequenza del salvataggio di recovery per DQN
    parser.add_argument(
        "--recovery-freq",
        type=int,
        default=None,
    )

    # Cartella di recovery da cui riprendere un training DQN
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # Verifica che i parametri numerici siano validi
    if args.timesteps <= 0:
        raise ValueError(
            "--timesteps deve essere maggiore di 0."
    )

    if args.eval_freq <= 0:
        raise ValueError(
            "--eval-freq deve essere maggiore di 0."
    )

    if args.checkpoint_freq is not None and args.checkpoint_freq <= 0:
        raise ValueError(
            "--checkpoint-freq deve essere maggiore di 0."
    )

    if args.recovery_freq is not None and args.recovery_freq <= 0:
        raise ValueError(
            "--recovery-freq deve essere maggiore di 0."
        )

    # Determina la frequenza del recovery DQN
    if args.recovery_freq is not None:
        recovery_freq = args.recovery_freq
    elif args.algo == "dqn" and args.run_type == "pilot":
        recovery_freq = max(1, args.timesteps // 4)
    elif args.algo == "dqn" and args.run_type == "final":
        recovery_freq = max(1, args.timesteps // 10)
    else:
        recovery_freq = None

    # Determina automaticamente la frequenza dei checkpoint
    if args.checkpoint_freq is not None:
        checkpoint_freq = args.checkpoint_freq
    elif args.run_type == "smoke":
        checkpoint_freq = None
    elif args.run_type == "pilot":
        checkpoint_freq = max(1, args.timesteps // 4)
    else:
        checkpoint_freq = max(1, args.timesteps // 10)

    # Creo un'etichetta compatta per il numero di timesteps e costruisco un identificatore univoco del training
    timesteps_label = format_timesteps(args.timesteps)
    run_name = f"{args.algo}_{args.run_type}_{timesteps_label}_seed_{args.seed}"

    if args.tag is not None:
        run_name = f"{run_name}_{args.tag}"

    # Indica se il training deve essere ripreso da un recovery
    is_resume = args.resume_from is not None

    # Configurazione completa dell'esperimento
    run_config = {
        "config_version": 1,

        "algo": args.algo,
        "seed": args.seed,
        "run_type": args.run_type,
        "tag": args.tag,
        "target_timesteps": args.timesteps,

        "evaluation": {
            "eval_freq": args.eval_freq,
            "eval_seeds": args.eval_seeds,
            "deterministic": True,
        },

        "best_model": {
            "metric": "mean_eval_reward",
            "enabled": True,
        },

        "checkpoint": {
            "checkpoint_freq": checkpoint_freq,
        },

        "recovery": {
            "recovery_freq": recovery_freq,
        },

        "environment": {
            "env_id": ENV_ID,
            "continuous": CONTINUOUS,
            "lap_complete_percent": LAP_COMPLETE_PERCENT,
            "domain_randomize": DOMAIN_RANDOMIZE,
            "max_episode_steps": MAX_EPISODE_STEPS,
        },

        "model": {
            "policy": POLICY,
            "device": DEVICE,
        },

        "software": get_software_versions(),
}

    env = make_env()

    # Creo un ambiente separato solo per valutare il modello
    eval_env = DummyVecEnv([
        lambda: Monitor(make_env())
    ])

    eval_env = VecTransposeImage(eval_env)

    os.makedirs("models", exist_ok=True)

    # Crea un nuovo modello oppure riprende un training DQN
    if is_resume:
        if args.algo != "dqn":
            raise ValueError(
                "Il resume da recovery è supportato solo per DQN."
            )

        # Configurazione associata al recovery
        recovery_config_path = os.path.join(
            args.resume_from,
            "config.json",
        )

        if not os.path.isfile(recovery_config_path):
            raise FileNotFoundError(
                f"Config del recovery non trovato: {recovery_config_path}"
            )

        # Carica la configurazione originale del training
        with open(
            recovery_config_path,
            "r",
            encoding="utf-8",
        ) as file:
            recovery_config = json.load(file)

        # Verifica che il comando di resume sia coerente con il training originale
        config_keys = [
            "config_version",
            "algo",
            "seed",
            "tag",
            "best_model",
            "run_type",
            "target_timesteps",
            "evaluation",
            "checkpoint",
            "recovery",
            "environment",
            "software",
        ]

        for key in config_keys:
            if recovery_config.get(key) != run_config.get(key):
                raise ValueError(
                    f"Configurazione non coerente per '{key}': "
                    f"recovery={recovery_config.get(key)}, "
                    f"comando attuale={run_config.get(key)}"
                )

        recovery_model_path = os.path.join(
            args.resume_from,
            "recovery_model.zip",
        )

        recovery_buffer_path = os.path.join(
            args.resume_from,
            "recovery_replay_buffer.pkl",
        )

        # Verifica che entrambi i file di recovery esistano
        if not os.path.isfile(recovery_model_path):
            raise FileNotFoundError(
                f"Modello di recovery non trovato: {recovery_model_path}"
        )

        if not os.path.isfile(recovery_buffer_path):
            raise FileNotFoundError(
                f"Replay buffer non trovato: {recovery_buffer_path}"
        )

        # Carica il modello DQN salvato
        model = DQN.load(
            recovery_model_path,
            env=env,
            device=DEVICE,
        )

        # Ripristina anche il replay buffer
        model.load_replay_buffer(
            recovery_buffer_path
        )

        resumed_timesteps = model.num_timesteps


        # Evita di riprendere un training che ha già raggiunto il target
        if resumed_timesteps >= args.timesteps:
            raise ValueError(
                f"Il recovery contiene già {resumed_timesteps} timesteps, "
                f"mentre il target richiesto è {args.timesteps}."
            )

        # Calcola quanti step mancano per raggiungere il target
        training_timesteps = args.timesteps - resumed_timesteps
        reset_num_timesteps = False

        print(
            f"Recovery DQN caricato da {args.resume_from}"
        )

        print(
            f"Timesteps già completati: {resumed_timesteps}"
        )

        print(
            f"Timesteps rimanenti: {training_timesteps}"
        )

    else:
        resumed_timesteps = 0
        training_timesteps = args.timesteps
        reset_num_timesteps = True

        if args.algo == "ppo":
            model = PPO(
                POLICY,
                env,
                seed=args.seed,
                device=DEVICE,
                verbose=1,
            )
        else:
            model = DQN(
                POLICY,
                env,
                seed=args.seed,
                device=DEVICE,
                verbose=1,
                buffer_size=DQN_BUFFER_SIZE,
                learning_starts=DQN_LEARNING_STARTS,
                exploration_fraction=DQN_EXPLORATION_FRACTION,
            )

    # Registra gli iperparametri effettivamente utilizzati dal modello
    run_config["model"]["hyperparameters"] = (
        get_model_hyperparameters(
        model,
        args.algo,
        )
    )

    # Durante un resume verifica anche la configurazione effettiva del modello
    if is_resume:
        if recovery_config.get("model") != run_config.get("model"):
            raise ValueError(
                "Configurazione del modello non coerente: "
                f"recovery={recovery_config.get('model')}, "
                f"modello caricato={run_config.get('model')}"
            )

    # Cartella principale dei log del training
    base_log_dir = os.path.join(
        "logs",
        run_name,
    )

    # Crea la cartella principale dei log
    os.makedirs(
        base_log_dir,
        exist_ok=True,
    )

    # Salva la configurazione completa del run
    config_path = os.path.join(
        base_log_dir,
        "config.json",
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
        run_config,
        file,
        indent=4,
    )

    # Durante un resume salva i nuovi log in una sottocartella separata
    if is_resume:
        log_dir = os.path.join(
            base_log_dir,
            f"resume_{format_timesteps(resumed_timesteps)}",
    )

    else:
        log_dir = base_log_dir

    # Mostra i log nel terminale e li salva anche in formato CSV
    logger = configure(
        log_dir,
        ["stdout", "csv"],
    )

    # Uso del logger per registrare il training
    model.set_logger(logger)

    # Cartella in cui salvare il miglior modello trovato durante la validation
    best_model_dir = os.path.join(
        "best_models",
        run_name,
    )

    # Valuta periodicamente il modello sulle stesse piste
    eval_callback = FixedSeedEvalCallback(
        eval_env=eval_env,               # Ambiente in cui viene effettuata la valutazione
        eval_freq=args.eval_freq,        # Ogni quanti step la valutazione viene effettuata
        eval_seeds=args.eval_seeds,
        log_path=log_dir,                # Cartella in cui vengono salvati i risultati delle evaluation
        best_model_save_path=best_model_dir,
        best_model_name=run_name,
        deterministic=True,              # La valutazione dell'agente avviene senza introdurre esplorazione casuale
        verbose=1,
    )     

    # Callback eseguiti durante il training
    callbacks = [
        eval_callback,
    ]

    # Aggiunge i checkpoint solo quando previsti
    if checkpoint_freq is not None:
        checkpoint_dir = os.path.join(
            "checkpoints",
            run_name,
        )

        os.makedirs(
            checkpoint_dir,
            exist_ok=True,
        )

        # Salva i checkpoint intermedi utilizzati per analisi e confronto
        checkpoint_callback = AnalysisCheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=run_name,
            verbose=1,
        )

        callbacks.append(checkpoint_callback)

    # Aggiunge il recovery solo per DQN quando previsto
    if args.algo == "dqn" and recovery_freq is not None:
        # Durante un resume continua ad utilizzare la stessa cartella di recovery
        if is_resume:
            recovery_dir = args.resume_from
        else:
            recovery_dir = os.path.join(
                "recovery",
                run_name,
            )

            os.makedirs(
                recovery_dir,
                exist_ok=True,
            )

            recovery_config_path = os.path.join(
                recovery_dir,
                "config.json",
            )

            with open(
                recovery_config_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    run_config,
                    file,
                    indent=4,
                )

        recovery_callback = RecoveryCallback(
            save_freq=recovery_freq,
            save_path=recovery_dir,
            verbose=1,
        )

        callbacks.append(recovery_callback)



    # Stato iniziale della sessione di training
    training_status = "completed"
    error_message = None

    # Avvia la misurazione del tempo
    training_start_time = time.perf_counter()

    try:
        # Avvia o riprende il training con i callback configurati
        model.learn(
            total_timesteps=training_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
        )

    except KeyboardInterrupt:
        # Interruzione volontaria, ad esempio tramite Ctrl+C
        training_status = "interrupted"

        print(
            "\nTraining interrotto dall'utente."
        )

    except Exception as error:
        # Registra eventuali errori inattesi
        training_status = "failed"
        error_message = (
            f"{type(error).__name__}: {error}"
        )

        raise

    finally:
        # Calcola il tempo trascorso durante questa sessione
        elapsed_seconds = (
            time.perf_counter()
            - training_start_time
        )

        # Numero effettivo di timestep raggiunti
        actual_timesteps = int(
            model.num_timesteps
        )

        # Salva il modello finale solo se il training è terminato normalmente
        if training_status == "completed":
            model.save(
                f"models/{run_name}"
            )

        # Riassunto di ciò che è realmente successo
        run_summary = {
            "status": training_status,
            "target_timesteps": args.timesteps,
            "start_timesteps": resumed_timesteps,
            "actual_timesteps": actual_timesteps,
            "timesteps_this_session": (
                actual_timesteps - resumed_timesteps
            ),
            "elapsed_seconds_this_session": elapsed_seconds,
            "resumed_from": args.resume_from,
            "hardware": {
                "processor": platform.processor(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
            },
        }

        if error_message is not None:
            run_summary["error"] = error_message

        summary_path = os.path.join(
            log_dir,
            "run_summary.json",
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                run_summary,
                file,
                indent=4,
            )

        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()