import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from ia import *
from numpy.random import gamma
from numpy.typing import NDArray


def plot_values(v: NDArray) -> None:
    v = np.reshape(v, (4, 4))
    # plot the state-value function
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    _ = ax.imshow(v, cmap="cool")
    for (j, i), label in np.ndenumerate(v):
        ax.text(i, j, np.round(label, 3), ha="center", va="center", fontsize=14)
    plt.tick_params(bottom="off", left="off", labelbottom="off", labelleft="off")
    plt.title("State-Value Function")
    plt.show()


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
    d = env.observation_space
    qa = QLearningAgent(env, Tabular, d=d)
    qa.train(0.01, max_steps=True, num_episodes=10000, gamma=0.9)
    v = np.empty(d.n)
    for i in range(d.start, d.start + d.n):
        print(i)
        _, v[i] = qa.val_act(np.int64(i))
    plot_values(v)
    qa.live_play(path="frozenlake.mp4", max_steps=True)
