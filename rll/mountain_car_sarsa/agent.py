import abc
import pickle
from gymnasium import Env
from gymnasium.vector import VectorEnv
import numpy as np
from rll.mountain_car_sarsa.utils import Hyperparameters


class BaseAgent(abc.ABC):
    env: Env | VectorEnv

    @abc.abstractmethod
    def get_action(self, state: np.ndarray) -> int:
        pass

    @abc.abstractmethod
    def update(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ):
        pass

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            self = pickle.load(f)
        return self


class RandomAgent(BaseAgent):
    def __init__(self, env: Env | VectorEnv, hyperparameters: Hyperparameters):
        self.env = env
        self.hyperparameters = hyperparameters

    def get_action(self, state: np.ndarray) -> int:
        return self.env.action_space.sample()

    def update(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ):
        pass
