import argparse
import csv
import os

import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from env import make_env

# Funzione che converte il numero di timesteps in un'etichetta compatta
def format_timesteps(timesteps: int) -> str:
    if timesteps % 1_000_000 == 0:
        return f"{timesteps // 1_000_000}M"

    if timesteps % 1_000 == 0:
        return f"{timesteps // 1_000}k"

    return str(timesteps)

# Callback di evaluation con seed fissi
class FixedSeedEvalCallback(BaseCallback):

    def __init__(
        self,
        eval_env,
        eval_freq,
        eval_seeds,
        log_path,
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

    args = parser.parse_args()

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

    env = make_env()

    # Creo un ambiente separato solo per valutare il modello
    eval_env = DummyVecEnv([
        lambda: Monitor(make_env())
    ])

    eval_env = VecTransposeImage(eval_env)

    os.makedirs("models", exist_ok=True)

    # Cartella in cui vengono salvati i log del training e delle evaluation.
    log_dir = os.path.join("logs", run_name)

    # Mostra i log nel terminale e li salva anche in formato CSV
    logger = configure(log_dir, ["stdout", "csv"])

    if args.algo == "ppo":
        model = PPO(
            "CnnPolicy",
            env,
            seed=args.seed,
            device="cpu",
            verbose=1,
        )

    else:
        model = DQN(
    "CnnPolicy",
    env,
    seed=args.seed,
    device="cpu",
    verbose=1,
    buffer_size=10_000,
    learning_starts=500,
    )

    # Uso del logger per registrare il training
    model.set_logger(logger)

    # Valuta periodicamente il modello sulle stesse piste
    eval_callback = FixedSeedEvalCallback(
        eval_env=eval_env,               # Ambiente in cui viene effettuata la valutazione
        eval_freq=args.eval_freq,        # Ogni quanti step la valutazione viene effettuata
        eval_seeds=args.eval_seeds,
        log_path=log_dir,                # Cartella in cui vengono salvati i risultati delle evaluation
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

        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=run_name,
            save_replay_buffer=(args.algo == "dqn"),
            save_vecnormalize=False,
            verbose=2,
        )

        callbacks.append(checkpoint_callback)



    # Avvia il training con i callback configurati.
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
    )

    # Salva il modello al termine del training.
    model.save(
        f"models/{run_name}"
    )

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()