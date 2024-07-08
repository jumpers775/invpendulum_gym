import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class InvPend(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, start: int | float=np.pi/4, setpoint: int | float=0, size: int | float=1, mass: int | float=1):
        self._theta = start
        self._velocity = 0
        self._gravity = 9.81
        self._mass = mass
        self._length = size
        self._setpoint = setpoint

        self._lastpass = None

        self.action_space = spaces.Box(low=-50, high=50, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None

    def _get_obs(self):
        return np.array([self._theta, self._setpoint - self._theta], dtype=np.float32)
    def _get_info(self):
        return {"distance": self._theta - self._setpoint}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._theta = self.np_random.uniform(low=-np.pi, high=np.pi)
        self._velocity = 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame(True)
        return observation, info

    def step(self, action):
        force = action[0]
        self._velocity += (self._gravity / self._length) * np.sin(self._theta) + force / (self._mass * (self._length ** 2))
        self._theta += self._velocity
        if self._lastpass:
            terminated = bool((self._lastpass > -0.1 and self._lastpass < 0.1 and 0 > self._theta - self._setpoint and self._theta - self._setpoint < 0.1) or self._theta > np.pi or self._theta < -np.pi)
        else:
            terminated = False
        self._lastpass = self._theta - self._setpoint
        reward = -100 * np.square(self._theta - self._setpoint) - 0.1 * np.square(force)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self, first = False):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((400, 400))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((self.window_size, self.window_size))
        if first:
            canvas.fill((255, 0, 0))
        else:
            canvas.fill((255, 255, 255))

        point1 = 200, 200
        point2 = 200 + 100* self._length * np.sin(self._theta), 200 - 100*self._length * np.cos(self._theta)

        pygame.draw.line(canvas, (0, 0, 0), point1, point2, 1)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

