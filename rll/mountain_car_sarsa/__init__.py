from typing import cast
import random as rd
from gymnasium.vector import SyncVectorEnv, VectorEnv
from gymnasium.envs.classic_control import MountainCarEnv
import gymnasium as gym
import numpy as np
import pandas as pd
from rll.mountain_car_sarsa.agent import BaseAgent, RandomAgent
from rll.mountain_car_sarsa.utils import Hyperparameters

import matplotlib as mpl

mpl.use("module://matplotlib-backend-kitty")

import matplotlib.pyplot as plt


def loop(agent: BaseAgent, hyperparameters: Hyperparameters):
    episilon_decay = hyperparameters.episilon_decay_builder(
        decay_end=hyperparameters.num_of_episodes
    )
    reward_per_episode: list[float] = []
    max_position_per_episode: list[float] = []

    for episode in range(hyperparameters.num_of_episodes):
        rewards = 0
        max_position = -99

        state = agent.env.reset()

        done = False
        while not done:
            epsilon = episilon_decay(episode)
            action = agent.get_action(cast(np.ndarray, state))

            next_state, reward, done, _, _ = agent.env.step(action)

            agent.update(
                cast(np.ndarray, state),
                action,
                cast(float, reward),
                cast(np.ndarray, next_state),
            )

            rewards += cast(float, reward)
            if next_state[0] > max_position:
                max_position = next_state[0]

            state = next_state

        reward_per_episode.append(rewards)
        max_position_per_episode.append(max_position)

    return reward_per_episode, max_position_per_episode


def train_with_hyperparameter(idx: int, hyperparameters: Hyperparameters):
    print(
        f"Training with alpha: {hyperparameters.alpha} and gamma: {hyperparameters.gamma}"
    )
    env = gym.make(
        "MountainCar-v0",
        # num_envs=hyperparameters.num_of_envs,
        # vectorization_mode="sync",
        # render_mode="rgb_array",
    )
    env._max_episode_steps = 1000
    # sync_env = cast(SyncVectorEnv, env)
    agent = RandomAgent(env, hyperparameters)
    reward_per_episode, max_positions = loop(agent, hyperparameters)
    n_completed = sum([1 if m > 0.5 else 0 for m in max_positions])
    print(f"{n_completed} success out of {hyperparameters.num_of_episodes} attempts")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Max position reached by the car")
    ax.set(xlim=(-0.5, 0.6), xticks=np.arange(-0.5, 0.6, 0.1))
    pd.Series(max_positions).plot(kind="hist", bins=100)
    plt.show()
    # env.reset()
    #
    # frame = env.render()
    #
    # fig, ax = plt.subplots()
    # if ax.axes and frame:
    #     # unwrapped = cast(MountainCarEnv, sync_env.envs[0].unwrapped)
    #     unwrapped = cast(MountainCarEnv, env.unwrapped)
    #     print(f"State space {unwrapped.observation_space}")
    #     print(f"Position space {unwrapped.min_position}, {unwrapped.max_position}")
    #     print(f"Velocity space {-unwrapped.max_speed}, {unwrapped.max_speed}")
    #     print(f"Action space {unwrapped.action_space}")
    #
    #     ax.axes.yaxis.set_visible(False)
    #     ax.imshow(
    #         frame[0],
    #         extent=(
    #             float(unwrapped.min_position),
    #             float(unwrapped.max_position),
    #             0.0,
    #             1.1,
    #         ),
    #     )
    #     fig.show()

    # agent = Agent(env, hyperparameters)
    # _, timesteps, penalties = train(agent, env)
    #
    #


def train_with_hyperparameters():
    train_with_hyperparameter(0, Hyperparameters(num_of_episodes=100))
