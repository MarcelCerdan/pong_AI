import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

class PongEnv(gym.Env):
	ACTION_SPACE_SIZE = 3
	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1000}

	def __init__(self, render_mode=None):
		self.window_size = (1600, 1200)
		# Observations are dictionnaries with 3 keys: player1, agent, ball wich contains the coordinates of the player1,
		# agent and ball + velocity of the ball
		self.observation_space = spaces.Dict(
			{
				"agent1": spaces.Dict(
				{
					"position": spaces.Box(-10, 10, shape=(2,), dtype=np.float64),
					"score": spaces.Box(0, 10, shape=(1,), dtype=np.int32)
				}),
				"agent2": spaces.Dict(
				{
					"position": spaces.Box(-10, 10, shape=(2,), dtype=np.float64),
					"score": spaces.Box(0, 10, shape=(1,), dtype=np.int32)
				}),
				"ball": spaces.Dict(
				{
					"position": spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float64),
					"velocity": spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float64)
				})
			}
		)

		self._border_up = 6.25
		self._border_down = -6.25

		# We have 3 actions: move up, move down, do nothing
		self.action_space = spaces.Discrete(3)

		self._action_to_direction = {
			0: np.array([0.0, 0.0], dtype=np.float64),
			1: np.array([0.0, 0.095], dtype=np.float64),
			2: np.array([0.0, -0.095], dtype=np.float64)
		}

		assert render_mode is None or render_mode in self.metadata["render_modes"]
		self.render_mode = render_mode

		self.window = None
		self.clock = None

	def _get_observation(self, name="agent2"):
		if name == "agent1":
			return {
				"agent":
				{
					"position": np.array(self._agent1_location, dtype=np.float64),
					"score": np.array([self._agent1_score], dtype=np.int32)
				},
				"ball": 
				{
					"position": np.array(self._ball_location, dtype=np.float64),
					"velocity": np.array(self._ball_velocity, dtype=np.float64)
				}
			}
		else:
			return {
				"agent":
				{
					"position": np.array(self._agent2_location, dtype=np.float64),
					"score": np.array([self._agent2_score], dtype=np.int32)
				},
				"ball": 
				{
					"position": np.array(self._ball_location, dtype=np.float64),
					"velocity": np.array(self._ball_velocity, dtype=np.float64)
				}
			}
	
	def _check_collision_with_paddles(self):
		if self._ball_location[0] >= self._agent2_location[0] and self._ball_location[0] <= self._agent2_location[0] + 0.5:
			if self._ball_location[1] <= self._agent2_location[1] + 1 and self._ball_location[1] >= self._agent2_location[1] - 1:
				return True
		if self._ball_location[0] <= self._agent1_location[0] and self._ball_location[0] >= self._agent1_location[0] - 0.5:
			if self._ball_location[1] <= self._agent1_location[1] + 1 and self._ball_location[1] >= self._agent1_location[1] - 1:
				return True
		return False
	
	def _collides_with_paddle(self):
		self._ball_velocity[0] *= -1
		if self._ball_velocity[0] > 0:
			self._ball_velocity[0] += 0.009
		elif self._ball_velocity[0] < 0:
			self._ball_velocity[0] -= 0.009
		if self._ball_location[0] > 0:
			self._ball_velocity[1] = 0.08 * (self._ball_location[1] - self._agent2_location[1])
		elif self._ball_location[0] < 0:
			self._ball_velocity[1] = 0.08 * (self._ball_location[1] - self._agent1_location[1])

	def _collides_with_border(self):
		if self._ball_velocity[1] > 0 and self._ball_location[1] >= self._border_up:
			self._ball_velocity[1] = -self._ball_velocity[1]
		elif self._ball_velocity[1] < 0 and self._ball_location[1] <= self._border_down:
			self._ball_velocity[1] = -self._ball_velocity[1]
		self._ball_velocity[1] *= -1

	def _check_if_scored(self):
		reward = 0
		if self._ball_location[0] >= 12:
			reward = -1
			self._agent1_score += 1
			self._ball_location = np.array([0, 0], dtype=np.float64)
			self._ball_velocity = np.array([0.055, 0], dtype=np.float64)
		elif self._ball_location[0] <= -12:
			reward = 1
			self._agent2_score += 1
			self._ball_location = np.array([0, 0], dtype=np.float64)
			self._ball_velocity = np.array([-0.055, 0], dtype=np.float64)
		return reward
		

	def _move_ball(self):
		self._ball_location += self._ball_velocity

	def reset(self, name="agent2", seed=None, options=None):
		super().reset(seed=seed)

		# Initialize positions, scores and ball velocity
		self._agent1_location = np.array([-9.5, 0.0])
		self._agent1_score = 0
		self._agent2_location = np.array([9.5, 0.0])
		self._agent2_score = 0
		self._ball_location = np.array([0.0, 0.0])
		self._ball_velocity = np.array([-0.055, 0.0])

		observation = self._get_observation(name)
		info = {}

		if self.render_mode == "human":
			self._render_frame()

		return observation, info

	def step(self, action, name="agent2"):
		terminated = False
		
		action1, action2 = action
		# Update agents positions and check if they collide with the border
		if (self._agent1_location[1] + 1 >= self._border_up and action1 == 1) or (self._agent1_location[1] - 1 <= self._border_down and action1 == 2):
			action1 = 0
		if (self._agent2_location[1] + 1 >= self._border_up and action2 == 1) or (self._agent2_location[1] - 1 <= self._border_down and action2 == 2):
			action2 = 0
		self._agent1_location += self._action_to_direction[action1]
		self._agent2_location += self._action_to_direction[action2]

		if self._check_collision_with_paddles():
			self._collides_with_paddle()

		self._collides_with_border()
		reward = self._check_if_scored()
		if reward != 0:
			print("Reward: ", reward)
		if (reward != 0):
			terminated = True
		self._move_ball()

		observation = self._get_observation(name)
		info = {}
		if self.render_mode == "human":
			self._render_frame()

		return (observation, reward, terminated, False, info)
	
	def render(self):
		if self.render_mode == "rgb_array":
			return self._render_frame()
		elif self.render_mode == "human":
			self._render_frame()

	def _render_frame(self):
		if self.window is None and self.render_mode == "human":
			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode((1600, 1200))
			if self.clock is None and self.render_mode == "human":
				self.clock = pygame.time.Clock()

		canvas = pygame.Surface((1600, 1200))
		canvas.fill((0, 0, 0))

		scale_factor = 50

		agent1_paddle_top = min(max(0, 600 - int(self._agent1_location[1] * scale_factor)), 1100 - 100)
		agent2_paddle_top = min(max(0, 600 - int(self._agent2_location[1] * scale_factor)), 1100 - 100)

		ball_x = 800 + int(self._ball_location[0] * scale_factor)
		ball_y = 600 + int(self._ball_location[1] * scale_factor)

		pygame.draw.rect(canvas, (0, 255, 0), (100, agent1_paddle_top, 20, 100))
		pygame.draw.rect(canvas, (255, 0, 0), (1480, agent2_paddle_top, 20, 100))
		pygame.draw.circle(canvas, (255, 255, 255), (ball_x, ball_y), 20)


		if self.render_mode == "human":
			self.window.blit(canvas, canvas.get_rect())
			pygame.event.pump()
			pygame.display.update()
			self.clock.tick(self.metadata["render_fps"])
		elif self.render_mode == "rgb_array":
			return np.transpose(np.array(pygame.surfarray.array3d(canvas)), axes=(1, 0, 2))
		
	def close(self):
		if self.window is not None:
			pygame.display.quit()
			pygame.quit()
			self.window = None
	