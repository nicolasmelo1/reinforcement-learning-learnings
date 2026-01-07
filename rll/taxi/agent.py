from gymnasium import Env, spaces
from gymnasium.vector import VectorEnv
import numpy as np

import os


class Agent:
    def __init__(self, env: Env | VectorEnv, alpha: float = 0.1, gamma: float = 0.9):
        self.env = env
        observation_space = getattr(
            env, "single_observation_space", env.observation_space
        )
        action_space = getattr(env, "single_action_space", env.action_space)
        assert isinstance(observation_space, spaces.Discrete)
        assert isinstance(action_space, spaces.Discrete)
        self.q_table = np.zeros((observation_space.n, action_space.n))

        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

    def act_batch(self, states: np.ndarray):
        action = np.argmax(self.q_table[states], axis=1)
        return action

    def act(self, states: np.ndarray):
        action = np.argmax(self.q_table[states])
        return action

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - old_value
        )

    def save(self, path: str):
        exists = os.path.exists(path)
        if not exists:
            os.makedirs(path)
        np.save(f"{path}/{str(self)}.npy", self.q_table)

    def load(self, path: str):
        self.q_table = np.load(path)

    def __str__(self):
        return f"model_alpha_{str(self.alpha).replace('.', '-')}_gamma_{str(self.gamma).replace('.', '-')}"
