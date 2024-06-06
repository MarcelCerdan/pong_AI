import PIL.Image
import gymnasium as gym
import my_gym_env
from gymnasium.utils.env_checker import check_env
from time import sleep

env = gym.make("PongEnv-v0", render_mode="human")
obs, inf = env.reset()
print(obs)
sleep(3)
obs = env.step(1)
print(obs)
sleep(3)
