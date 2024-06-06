from gymnasium.envs.registration import register

register(
	id="PongEnv-v0",
	entry_point="my_gym_env.envs:PongEnv",
)