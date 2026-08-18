from env import make_env


def main():
    env = make_env(render_mode="human")

    obs, info = env.reset(seed=12)

    while True:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()


if __name__ == "__main__":
    main()