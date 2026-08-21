import argparse
import os

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
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
        required = True,
    )

    # Ogni quanti timesteps effettuare la valutazione del modello
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
    )

    # Numero di episodi utilizati per ogni valutazione periodica
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
    )

    # Seed usato per inizializzare l'ambiente di evaluation
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=100
    )

    args = parser.parse_args()

    # Creo un'etichetta compatta per il numero di timesteps e costruisco un identificatore univoco del training
    timesteps_label = format_timesteps(args.timesteps)
    run_name = (f"{args.algo}_{args.run_type}_{timesteps_label}_seed_{args.seed}")

    env = make_env()

    # Creo un ambiente separato solo per valutare il modello
    eval_env = DummyVecEnv([
        lambda: Monitor(make_env())
    ])

    eval_env = VecTransposeImage(eval_env)

    
    eval_env.seed(args.eval_seed)

    os.makedirs("models", exist_ok=True)

    # Cartella in cui vengono salvati i log del training e delle evaluation.
    log_dir = os.path.join("logs", run_name)

    # Mostra i log nel terminale e li salva anche in formato CSV
    logger = configure(log_dir, ["stdout","csv"])

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

    # Valuta periodicamente il modello durante il training
    eval_callback = EvalCallback(
        eval_env,                               # Ambiente in cui viene effettuata la valutazione
        n_eval_episodes=args.eval_episodes,     # Numero di episodi eseguiti per ogni evaluation
        eval_freq=args.eval_freq,               # Ogni quanti step la valutazione viene effettuata
        log_path=log_dir,                       # Cartella in cui vengono salvati i risultati delle evaluation
        deterministic=True,                     # La valutazione dell'agente avviene senza introdurre esplorazione casuale
        render=False,                           # Non viene aperta alcuna finestra grafica
        verbose=1,                              
    )       

    # Avvia il training ed esegue periodicamente l'evaluation.
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
    )

    # Salva il modello al termine del training.
    model.save(
        f"models/{run_name}"
    )

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()