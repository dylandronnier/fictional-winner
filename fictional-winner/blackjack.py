import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from ia import *
from matplotlib.patches import Patch


def create_grids(agent):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the policy grid for plotting
    policy_grid1 = np.apply_along_axis(
        lambda obs: agent.act((obs[0], obs[1], np.int64(0))),
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    policy_grid2 = np.apply_along_axis(
        lambda obs: agent.act((obs[0], obs[1], np.int64(1))),
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return policy_grid1, policy_grid2


def create_plots(policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    _, ax = plt.subplots()

    ax = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax.set_title(f"Policy: {title}")
    ax.set_xlabel("Player sum")
    ax.set_ylabel("Dealer showing")
    ax.set_xticklabels(range(12, 22))
    ax.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))


if __name__ == "__main__":
    env = gym.make("Blackjack-v1", render_mode="rgb_array", sab=True)
    qa = QLearningAgent(env, Tabular)
    qa.train(0.001, max_steps=True, num_episodes=10_000_000, policy="off")
    mean, std = qa.mean_score(iter=50_000)
    print(f"Mean: {mean} +- {1.96 * std / np.sqrt(50_000)}")
    g1, g2 = create_grids(qa)
    create_plots(g1, title="without usable")
    create_plots(g2, title="with usable")
    plt.show()
    qa.live_play(path="blackjack.mp4", max_steps=600)
