import time
from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

import gymnasium as gym
import gymnasium.wrappers.monitoring.video_recorder as vid
import numpy as np
from function_approximation import *
from gymnasium.spaces import Discrete

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Agent(ABC, Generic[ObsType, ActType]):
    """The abstract class agent."""

    def __init__(self, env: gym.Env[ObsType, ActType]) -> None:
        """Init method for the Agent class.

        Initialize with the environment where the agent will play.
        """
        self.env = env
        super().__init__()

    @abstractmethod
    def act(self, state: ObsType) -> ActType:
        """Predict the value."""
        assert self.env.observation_space.contains(
            state
        ), f"{state!r} ({type(state)}) invalid"
        return self.env.action_space.sample()

    def play(self, max_steps: bool | int = True) -> float:
        """Record the agent play in a video.

        Parameters
        ----------
        max_steps : bool | int
            if max_steps =  False, play until the terminal state is reached.
            if max_steps =  True, play until the terminal state is reached or
                the environment step return truncated = True
            if max_steps = int,  play until the terminal state is reached or
                max_steps steps have been played.

        Returns
        -------
        score : float
            The score obtained by the agent at the end of the play.

        """
        steps = 0
        score = 0.0
        terminated = False
        truncated = False
        state = self.env.reset()[0]
        while not terminated and (
            not max_steps
            or (not truncated and isinstance(max_steps, bool) or steps < int(max_steps))
        ):
            state, reward, terminated, truncated, _ = self.env.step(self.act(state))
            score += float(reward)
            steps += 1
        return score

    def live_play(self, path: str, max_steps: bool | int = False) -> float:
        """Record the game played by the agent in a video.

        Parameters
        ----------
        path : str
            The relative path where the video will be recorded.
        max_steps : bool | int
            If max_steps =  False, play until the terminal state is reached.
            If max_steps =  True, play until the terminal state is reached or
                the environment step return truncated = True
            If max_steps = int,  play until the terminal state is reached or
                max_steps steps have been played.

        Returns
        -------
        score : float
            The score of the play

        """
        steps = 0
        score = 0.0
        terminated = False
        truncated = False
        state = self.env.reset()[0]
        rec = vid.VideoRecorder(self.env, path)
        while not terminated and (
            not max_steps
            or (not truncated and isinstance(max_steps, bool) or steps < int(max_steps))
        ):
            rec.capture_frame()
            state, reward, terminated, truncated, _ = self.env.step(self.act(state))
            score += float(reward)
            steps += 1
        rec.close()
        return score

    def mean_score(self, iter: int = 200) -> tuple[float, float]:
        """Compute the mean_score.

        Parameters
        ----------
        iter : int
            The number of episodes to run.

        Return

        mean_score: float
            The final score

        standard_dev: float

        """
        sum_1 = 0.0
        sum_2 = 0.0
        for _ in range(iter):
            score = self.play()
            sum_1 += score
            sum_2 += score**2
        return sum_1 / iter, np.sqrt(sum_2 - sum_1**2 / iter) / np.sqrt(iter - 1.5)


class QLearningAgent(Agent[ObsType, np.int64]):
    def __init__(
        self,
        env: gym.Env[ObsType, np.int64],
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

    def val_act(self, observation: ObsType) -> tuple[np.int64, np.float64]:
        super().act(observation)
        m = np.float64(np.NINF)
        res = self.min_action
        for i, q in enumerate(self.qvalue):
            v = q.value(observation)
            if m < v:
                m = v
                res = self.min_action + i
        return res, m

    def _tab(
        self, observation: ObsType
    ) -> tuple[dict[np.int64, np.float64], np.int64, np.float64]:
        super().act(observation)
        m = np.float64(np.NINF)
        res = self.min_action
        d = dict()
        for i, q in enumerate(self.qvalue):
            v = q.value(observation)
            d[self.min_action + i] = v
            if m < v:
                m = v
                res = self.min_action + i
        return d, res, m

    def act(self, state: ObsType) -> np.int64:
        return self.val_act(state)[0]

    def eps_greedy(
        self, observation: ObsType, epsilon: float = 0.1
    ) -> tuple[np.int64, np.float64]:
        super().act(observation)
        if np.random.binomial(1, epsilon):
            act = np.random.randint(self.min_action, high=self.max_action + 1)
            return act, self.qvalue[act - self.min_action].value(observation)
        else:
            return self.val_act(observation)

    def train(
        self,
        alpha: float,
        num_episodes: int,
        max_steps: bool | int = False,
        gamma: float = 1.0,
        policy: str = "on",
    ) -> None:
        """Train the model using on-policy or off-policy 1-step algorithm."""
        mscore = 0.0
        for i in range(1, num_episodes + 1):
            steps = 0
            score = 0.0
            terminated = False
            truncated = False
            eps = max(
                0.05, 1.0 / (1.0 + 0.6 * np.log(i))
            )  # Works good with Cliff Walking
            # eps = 0.3
            observation = self.env.reset()[0]
            action, qval = self.eps_greedy(observation, epsilon=eps)
            pos = action - self.min_action

            while not max_steps or (
                not truncated and isinstance(max_steps, bool) or steps < int(max_steps)
            ):
                nobservation, reward, terminated, truncated, _ = self.env.step(action)

                reward = np.float64(reward)
                score += reward
                if terminated:
                    self.qvalue[pos].param += (
                        alpha
                        * (reward - qval)
                        * self.qvalue[pos].derivative(observation)
                    )
                    break

                if policy == "off":
                    tab, best_action, v = self._tab(nobservation)
                    if np.random.binomial(1, eps):
                        naction = np.random.randint(
                            self.min_action, high=self.max_action + 1
                        )
                        nqval = tab.get(naction)
                    else:
                        naction = best_action
                        nqval = v
                else:
                    naction, nqval = self.eps_greedy(nobservation, epsilon=eps)
                    v = nqval

                npos = naction - self.min_action
                self.qvalue[pos].param += (
                    alpha
                    * (reward + gamma * v - qval)
                    * self.qvalue[pos].derivative(observation)
                )
                observation = nobservation
                qval = nqval
                action = naction
                pos = npos

                steps += 1
                # print(steps)

            mscore += score
            if i % 10_000 == 0:
                print(f"Episode {i} with mean score {mscore / 10_000}")
                mscore = 0.0
                time.sleep(1)
