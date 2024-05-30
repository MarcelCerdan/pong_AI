from gymnasium.envs.registration import register

register(
	id="my_gym_env/PongEnv-v0",
	entry_point="my_gym_env.envs:PongEnv",
	max_episode_steps=300
)