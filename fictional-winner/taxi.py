import gymnasium as gym
from ia import *

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    qa = QLearningAgent(env, Tabular)
    qa.train(0.01, max_steps=True, num_episodes=150000, policy="off")
    qa.live_play(path="taxi-off1.mp4", max_steps=600)
