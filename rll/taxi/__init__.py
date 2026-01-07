from typing import Literal
import multiprocessing as mp
import gymnasium as gym
import random as rd
import itertools
from rll.taxi.utils import episilon_decay_builder
from gymnasium.vector import VectorEnv
import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")
import matplotlib.pyplot as plt


from .agent import Agent


EPISILON = 0.02
NUMBER_OF_EPISODES = 10000
NUM_OF_ENVS = 10


hyperparameters: dict[Literal["alpha", "gamma"], list[float]] = {
    "alpha": [0.1, 1],
    "gamma": [0.1, 0.6, 0.9],
}


def run(model: str, eval_episilon: float = 0.05):
    env = gym.make("Taxi-v3", render_mode="human")
    agent = Agent(env)
    agent.load(f"rll/taxi/model/{model}")

    state, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        if rd.random() < eval_episilon:
            action = agent.act(state)
        else:
            action = agent.act(state)

        state, _, terminated, truncated, _ = env.step(action)


def train(agent: Agent, env: VectorEnv):
    penalties_per_episode: list[int] = []
    timesteps_per_episode: list[int] = []
    states, _ = env.reset()
    autoreset_mask = [False for _ in range(NUM_OF_ENVS)]
    penalties = np.zeros(NUM_OF_ENVS)
    timesteps = np.zeros(NUM_OF_ENVS)
    total_episodes = 0
    episilon_decay = episilon_decay_builder(decay_end=8000, start=0.5, end=0.01)

    while total_episodes < NUMBER_OF_EPISODES:
        episilon = episilon_decay(total_episodes)
        greedy_actions = agent.act_batch(states)
        random_actions = np.array(
            [env.single_action_space.sample() for _ in range(NUM_OF_ENVS)]
        )
        use_random = np.random.rand(NUM_OF_ENVS) < episilon
        actions = np.where(use_random, random_actions, greedy_actions)

        next_states, rewards, terminated, truncated, _ = env.step(actions)
        terminateds = terminated | truncated

        for i in range(NUM_OF_ENVS):
            if not autoreset_mask[i]:
                agent.update(states[i], actions[i], rewards[i], next_states[i])

                if rewards[i] == -10:
                    penalties[i] += 1

                timesteps[i] += 1

            if terminateds[i]:
                penalties_per_episode.append(penalties[i])
                timesteps_per_episode.append(timesteps[i])
                penalties[i] = 0
                timesteps[i] = 0
                total_episodes += 1
                if total_episodes >= NUMBER_OF_EPISODES:
                    break

        states = next_states
        autoreset_mask = terminateds

    agent.save("rll/taxi/model")
    return agent, timesteps_per_episode, penalties_per_episode


def train_with_hyperparameter(idx: int, alpha: float, gamma: float):
    print(f"Training with alpha: {alpha} and gamma: {gamma}")
    env = gym.make_vec("Taxi-v3", num_envs=NUM_OF_ENVS, vectorization_mode="sync")
    agent = Agent(env, alpha=alpha, gamma=gamma)
    _, timesteps, penalties = train(agent, env)

    return pd.DataFrame(
        {
            "run_number": idx,
            "hyperparameters": f"alpha={alpha}, gamma={gamma}",
            "timesteps": timesteps,
            "penalties": penalties,
        }
    )


def train_with_hyperparameters():
    results = pd.DataFrame()
    combinations = list(
        itertools.product(hyperparameters["alpha"], hyperparameters["gamma"])
    )

    with mp.Pool(len(combinations)) as pool:
        results = pd.concat(
            pool.starmap(
                train_with_hyperparameter,
                [(i, alpha, gamma) for i, (alpha, gamma) in enumerate(combinations)],
            )
        )
    results = results.reset_index().rename(columns={"index": "episode"})

    print(results)

    fig, ax = plt.subplots()
    sns.lineplot(
        data=results,
        x="episode",
        y="timesteps",
        units="run_number",
        estimator=None,
        hue="hyperparameters",
        ax=ax,
        dashes=True,
        errorbar=None,
    )
    ax.grid(False)
    fig.tight_layout()
    fig.show()
