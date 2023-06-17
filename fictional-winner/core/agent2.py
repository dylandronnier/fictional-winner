import time

import numpy as np
from core.agent import *
from core.fun import *
from core.q import *


class QLearningAgent2(Agent[ObsType, np.int64]):
    def __init__(
        self,
        env: gym.Env[ObsType, np.int64],
        hypothesis_set: Type[FunctionApprox],
    ) -> None:
        assert isinstance(env.action_space, Discrete)
        super().__init__(env)
        self.min_action = env.action_space.start
        self.max_action = self.min_action + env.action_space.n - 1
        self.qvalue = Q(env.action_space, env.observation_space, hypothesis_set)

    def val_act(self, observation: ObsType) -> tuple[np.int64, np.float64]:
        super().act(observation)
        m = np.float64(np.NINF)
        res = self.min_action
        for i in range(self.min_action, self.max_action + 1):
            v = self.qvalue.value((observation, i))
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
        for i in range(self.min_action, self.max_action + 1):
            v = self.qvalue.value((observation, i))
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
            return act, self.qvalue.value((observation, act))
        else:
            return self.val_act(observation)

    def train(
        self,
        alpha: float,
        num_episodes: int,
        max_steps: bool | int = False,
        gamma: float = 1.0,
        policy: str = "on",
        freq: int = 100,
    ) -> None:
        """Train the model using on-policy or off-policy 1-step algorithm."""
        mscore = 0.0
        for i in range(1, num_episodes + 1):
            steps = 0
            score = 0.0
            terminated = False
            truncated = False
            eps = max(
                0.05, 1.0 / (1.0 + 0.6 * np.sqrt(i))
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
                    self.qvalue.param[pos] += (
                        alpha
                        * (reward - qval)
                        * self.qvalue.derivative((observation, action))[pos]
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
                self.qvalue.param[pos] += (
                    alpha
                    * (reward + gamma * v - qval)
                    * self.qvalue.derivative((observation, action))[pos]
                )
                observation = nobservation
                qval = nqval
                action = naction
                pos = npos

                steps += 1
                # print(steps)

            mscore += score / freq
            if i % freq == 0:
                print(f"Episode {i} with mean score {mscore}")
                mscore = 0.0
                # time.sleep(1)

    def train_elligibility(
        self,
        alpha: float,
        num_episodes: int,
        max_steps: bool | int = False,
        gamma: float = 1.0,
        lambd: float = 0.0,
        freq: int = 100,
    ) -> None:
        """Train the model using on-policy or off-policy 1-step algorithm."""
        mscore = 0.0
        z = np.zeros(self.qvalue.param.shape)
        for i in range(1, num_episodes + 1):
            steps = 0
            score = 0.0
            terminated = False
            truncated = False
            eps = 0.0  # max(0.05, 0.3 / (1.0 + np.log(i)))  # Works good with Cliff Walking
            # eps = 0.3
            observation = self.env.reset()[0]
            action, qval = self.eps_greedy(observation, epsilon=eps)
            pos = action - self.min_action
            z[:] = 0.0

            while not max_steps or (
                not truncated and isinstance(max_steps, bool) or steps < int(max_steps)
            ):
                nobservation, reward, terminated, truncated, _ = self.env.step(action)

                reward = np.float64(reward)
                score += reward

                delta = reward - qval
                z[pos] += self.qvalue.derivative((observation, action))[pos]

                if terminated:
                    self.qvalue.param += alpha * delta * z
                    break

                # if policy == "off":
                #    tab, best_action, v = self._tab(nobservation)
                #    if np.random.binomial(1, eps):
                #        naction = np.random.randint(
                #            self.min_action, high=self.max_action + 1
                #        )
                #        nqval = tab.get(naction)
                #    else:
                #        naction = best_action
                #        nqval = v
                naction, nqval = self.eps_greedy(nobservation, epsilon=eps)
                delta += gamma * nqval

                npos = naction - self.min_action

                self.qvalue.param += alpha * delta * z
                z *= gamma * lambd

                observation = nobservation
                qval = nqval
                action = naction
                pos = npos

                steps += 1
                # print(steps)

            mscore += score / freq
            if i % freq == 0:
                print(f"Episodes from {i - freq} to {i}, mean score {mscore}")
                mscore = 0.0
                time.sleep(2)
