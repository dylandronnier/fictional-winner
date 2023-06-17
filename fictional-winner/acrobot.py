import gymnasium as gym
import numpy as np
from core.agent2 import *
from core.fun import *

if __name__ == "__main__":
    env = gym.make("Acrobot-v1", render_mode="rgb_array")
    qa = QLearningAgent2(env, LinearTiling)
    qa.train_elligibility(0.01, max_steps=True, num_episodes=1_200, lambd=0.25, freq=20)
    score = qa.live_play(path="acrobot-on.mp4", max_steps=True)
    print(f"Score de la video: {score}")
