import argparse
import os

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.logger import configure

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

    args = parser.parse_args()

    # Creo un'etichetta compatta per il numero di timesteps e costruisco un identificatore univoco del training
    timesteps_label = format_timesteps(args.timesteps)
    run_name = (f"{args.algo}_{args.run_type}_{timesteps_label}_seed_{args.seed}")

    env = make_env()

    os.makedirs("models", exist_ok=True)

    # Cartella in cui vengno salvati i dati di training
    log_dir = os.path.join("logs", run_name);

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

    model.learn(total_timesteps=args.timesteps)

    timesteps_label = format_timesteps(args.timesteps)

    model.save(
    f"models/{args.algo}_{args.run_type}_{timesteps_label}_seed_{args.seed}"
    )

    env.close()


if __name__ == "__main__":
    main()