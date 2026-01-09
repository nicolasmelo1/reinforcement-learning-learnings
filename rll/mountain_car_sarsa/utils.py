from dataclasses import dataclass, field


@dataclass
class Hyperparameters:
    alpha: list[float] = field(default_factory=lambda: [0.01, 0.1, 1])
    gamma: list[float] = field(default_factory=lambda: [0.1, 0.6, 0.9])
    num_of_envs: int = 4
    num_of_episodes: int = 1000

    def episilon_decay_builder(
        self, decay_end: float, start: float = 0.5, end: float = 0.01
    ):
        def episilon_decay(ep: float):
            if ep >= decay_end:
                return end
            return start + (end - start) * (ep / decay_end)

        return episilon_decay
