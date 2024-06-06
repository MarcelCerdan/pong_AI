import PIL.Image
import gymnasium as gym
import my_gym_env
from gym.utils.env_checker import check_env
from time import sleep

env = gym.make("PongEnv-v0", render_mode="human")
env.reset()
sleep(3)
env.step(1)
sleep(3)
# check_env(env.unwrapped)
