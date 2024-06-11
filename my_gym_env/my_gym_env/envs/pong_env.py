import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

class PongEnv(gym.Env):
	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

	def __init__(self, render_mode=None):
		self.window_size = (800, 600)
		# Observations are dictionnaries with 3 keys: player1, agent, ball wich contains the coordinates of the player1,
		# agent and ball + velocity of the ball
		self.observation_space = spaces.Dict(
			{
				"agent1": spaces.Dict(
				{
					"position": spaces.Box(-10.0, 10.0, shape=(1, 1), dtype=np.float64),
					"score": spaces.Box(0, 10, shape=(1,), dtype=np.int32)
				}),
				"agent2": spaces.Dict(
				{
					"position": spaces.Box(-10.0, 10.0, shape=(1, 1), dtype=np.float64),
					"score": spaces.Box(0, 10, shape=(1,), dtype=np.int32)
				}),
				"ball": spaces.Dict(
				{
					"position": spaces.Box(-10.0, 10.0, shape=(1, 1), dtype=np.float64),
					"velocity": spaces.Box(-10.0, 10.0, shape=(1, 1), dtype=np.float64)
				})
			}
		)

		self._borders_up = 6.25
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

	def _get_observation(self):
		return {
			"agent1": 
			{
				"position": self._agent1_location,
				"score": self._agent1_score
			},
			"agent2":
			{
				"position": self._agent2_location,
				"score": self._agent2_score
			},
			"ball": 
			{
				"position": self._ball_location,
				"velocity": self._ball_velocity
			}
		}
	
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		# Initialize positions, scores and ball velocity
		self._agent1_location = np.array([-9.5, 0.0])
		self._agent1_score = 0
		self._agent2_location = np.array([9.5, 0.0])
		self._agent2_score = 0
		self._ball_location = np.array([0.0, 0.0])
		self._ball_velocity = np.array([0.055, 0.0])

		observation = self._get_observation()
		info = {}

		if self.render_mode == "human":
			self._render_frame()

		return observation, info

	def step(self, action1, action2):
		terminated = False
		reward = 0
		# Update agents positions
		self._agent1_location += self._action_to_direction[action1]
		self._agent2_location += self._action_to_direction[action2]

		# Check if ball collides with borders
		if self._ball_velocity[1] > 0 and self._ball_location[1] >= self._borders_up:
			self._ball_velocity[1] = -self._ball_velocity[1]
		elif self._ball_velocity[1] < 0 and self._ball_location[1] <= self._border_down:
			self._ball_velocity[1] = -self._ball_velocity[1]
		self._ball_velocity[1] *= -1

		# Check if ball is in player1 or agent's goal
		if self._ball_location[0] >= 12:
			reward = -1
			self._agent1_score += 1
			self._ball_location = {0, 0}
			self._ball_velocity = {0.055, 0}
		elif self._ball_location[0] < self._agent1_location[0] - 2:
			reward = 1
			self._agent2_score += 1
			self._ball_location = {0, 0}
			self._ball_velocity = {-0.055, 0}

		observation = self._get_observation()
		info = {}
		if (self._agent1_score == 10 or self._agent2_score == 10):
			terminated = True

		if self.render_mode == "human":
			self._render_frame()

		return observation, reward, terminated, False, info
	
	def render(self):
		if self.render_mode == "rgb_array":
			return self._render_frame()
		elif self.render_mode == "human":
			self._render_frame()

	def _check_collision_with_paddles(self):
		if self._ball_location[0] >= self._agent1_location[0]

	def _render_frame(self):
		if self.window is None and self.render_mode == "human":
			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode((800, 800))
			if self.clock is None and self.render_mode == "human":
				self.clock = pygame.time.Clock()
			
			canvas = pygame.Surface(self.window.get_size())
			canvas.fill((255, 255, 255))
			

			# Draw player1
			pygame.draw.rect(canvas, (0, 255, 0), (0, 400, 10, 50))

			# Draw agent
			pygame.draw.rect(canvas, (255, 0, 0), (790, 400, 10, 50))

			# Draw ball
			pygame.draw.circle(canvas, (0, 0, 0), (400, 400), 5)

			if self.render_mode == "human":
				self.window.blit(canvas, canvas.get_rect())
				pygame.event.pump()
				pygame.display.update()
				self.clock.tick(60)
			else:
				return np.transpose(np.array(pygame.surfarray.array3d(canvas), axes=(1, 0, 2)))
		
	def close(self):
		if self.window is not None:
			pygame.display.quit()
			pygame.quit()
			self.window = None
	
