import gymnasium as gym
import numpy as np
from core.agent.py import *
from core.fun import *
from gymnasium.core import ActType

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class PolicyGradAgent(Agent):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        hypothesis_set: Type[FunctionApprox],
        *args,
        **kwargs,
    ) -> None:
        assert isinstance(env.action_space, Discrete)
        super().__init__(env)
        self.min_action = env.action_space.start
        self.max_action = self.min_action + env.action_space.n - 1
        self.qvalue = list()
        for _ in range(env.action_space.n):
            self.qvalue.append(hypothesis_set(env.observation_space, *args, **kwargs))
