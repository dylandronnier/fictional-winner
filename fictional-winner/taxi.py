import gymnasium as gym
from core.agent2 import *
from core.fun import *

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    qa = QLearningAgent(env, Tabular)
    qa.train(0.01, max_steps=True, num_episodes=10000, policy="off")
    qa.live_play(path="taxi-off1.mp4", max_steps=600)
