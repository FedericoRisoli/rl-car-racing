import gymnasium as gym

ENV_ID = "CarRacing-v2"


def make_env(render_mode=None):
    return gym.make(
        ENV_ID,
        continuous=False,
        render_mode=render_mode,
    )