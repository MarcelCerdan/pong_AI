import PIL.Image
import gymnasium as gym
import my_gym_env
from gymnasium.utils.env_checker import check_env
from time import sleep

actions = [0,0]
env = gym.make("PongEnv-v0", render_mode="human")
obs, inf = env.reset()
while True:
	obs = env.step(actions)
	# print(obs)
	# if reward != 0:
	# 	break
	#sleep(0.1)
# print(obs)
# sleep(1)
# obs = env.step(actions)
# print(obs)
# sleep(1)
