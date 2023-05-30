import numpy as np


def plot_policy(ax, agent) -> None:
    """Fonction that plots a deterministic policy pi on the plane pos/speed."""
    xs = np.linspace(0.0, 1.0, 200)
    vs = np.linspace(0.0, 1.0, 200)
    x, v = np.meshgrid(xs, vs)
    states = np.c_[x.ravel(), v.ravel()]
    decisions = agent.act(states).reshape(x.shape)
    ax.contourf(x, v, decisions, cmap=colormaps["brg"], alpha=0.2)
