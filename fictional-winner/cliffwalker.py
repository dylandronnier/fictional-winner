import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from core.agent2 import *
from core.fun import *
from numpy.typing import NDArray


def plot_values(v: NDArray) -> None:
    v = np.reshape(v, (4, 12))
    # plot the state-value function
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    _ = ax.imshow(v, cmap="cool")
    for (j, i), label in np.ndenumerate(v):
        ax.text(i, j, np.round(label, 3), ha="center", va="center", fontsize=14)
    plt.tick_params(bottom="off", left="off", labelbottom="off", labelleft="off")
    plt.title("State-Value Function")
    plt.show()


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    d = env.observation_space
    qa = QLearningAgent2(env, Tabular, d=d)
    qa.train(0.01, max_steps=True, num_episodes=10_000, policy="off")
    v = np.empty(48)
    for i in range(d.start, d.start + d.n):
        print(i)
        _, v[i] = qa.val_act(np.int64(i))
    plot_values(v)
    qa.live_play(path="cliffwalker-off.mp4", max_steps=400)
