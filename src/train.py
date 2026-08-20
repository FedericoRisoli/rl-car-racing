import argparse
import os

from stable_baselines3 import PPO, DQN

from env import make_env

# Funzione utilizzata per attribuire nomi standard ai modelli
def format_timesteps(timesteps: int) -> str:
    if timesteps % 1_000_000 == 0:
        return f"{timesteps // 1_000_000}M"

    if timesteps % 1_000 == 0:
        return f"{timesteps // 1_000}k"

    return str(timesteps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=["ppo", "dqn"],
        required=True,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--run-type",
        choices=["smoke", "pilot", "final"],
        default="smoke",
    )

    args = parser.parse_args()

    env = make_env()

    os.makedirs("models", exist_ok=True)

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
        

    model.learn(total_timesteps=args.timesteps)

    timesteps_label = format_timesteps(args.timesteps)

    model.save(
    f"models/{args.algo}_{args.run_type}_{timesteps_label}_seed_{args.seed}"
    )

    env.close()


if __name__ == "__main__":
    main()