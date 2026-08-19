import argparse
import os

from stable_baselines3 import PPO, DQN

from env import make_env


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

    model.save(
        f"models/{args.algo}_smoke_seed_{args.seed}"
    )

    env.close()


if __name__ == "__main__":
    main()