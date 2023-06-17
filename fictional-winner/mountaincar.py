import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from core.agent2 import *
from core.fun import *
from matplotlib import colormaps


def plot_policy(agent) -> None:
    """Fonction that plots a deterministic policy pi on the plane pos/speed."""
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    xs = np.linspace(-1.2, 0.6, 200, dtype=np.float32)
    vs = np.linspace(-0.07, 0.07, 200, dtype=np.float32)
    x, v = np.meshgrid(xs, vs)
    states = np.c_[x.ravel(), v.ravel()]
    decisions = np.array([agent.act(s) for s in states]).reshape(x.shape)
    ax.contourf(x, v, decisions, cmap=colormaps["brg"], alpha=0.2)
    plt.title("Best action")
    plt.show()


if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    qa = QLearningAgent2(env, LinearTiling)
    qa.train_elligibility(0.01, max_steps=True, num_episodes=3_000, lambd=0.2)
    plot_policy(qa)
    qa.live_play(path="mountain-on.mp4", max_steps=400)
