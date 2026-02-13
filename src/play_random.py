import gymnasium as gym

ENV_ID = "CarRacing-v2"  # cambia in v3 se serve

def main():
    env = gym.make(ENV_ID, render_mode="human")
    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    main()
