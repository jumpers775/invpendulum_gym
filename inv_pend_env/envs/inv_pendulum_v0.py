import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt





class InvPend(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, 
                 render_mode=None, 
                 setpoint: int | float=0, size: int | float=1, 
                 mass: int | float=1, 
                 plot: bool = False, 
                 seed: int | float = None, 
                 disallowcontrol: bool = False, 
                 start: int | float=None, 
                 timestep: int | float = 0.1,
                 terminate: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._theta = start or self.np_random.uniform(low=-np.pi/2, high=np.pi/2)
        self._start = start
        self._velocity = 0
        self._gravity = 9.81
        self._mass = mass
        self._length = size
        self._setpoint = setpoint
        
        self.timestep = timestep
        self.disallowcontrol = disallowcontrol
        self.seed = seed or None
        self.plot = plot
        self._lastpass = None
        self.terminate = terminate
        self.action_space = spaces.Box(low=-50, high=50, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None

        self.steps = 0

        self.controls = []
        self.thetas = []

    def _get_obs(self):
        return np.array([self._theta, self._setpoint - self._theta], dtype=np.float32)
    def _get_info(self):
        return {"distance": self._theta - self._setpoint}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed or self.seed or None)
        self._theta = self._start or self.np_random.uniform(low=-np.pi/2, high=np.pi/2)
        self._velocity = 0
        self.steps = 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame(True)
        return observation, info

    def step(self, action):
        self._lastpass = self._theta
        force = action[0] if not self.disallowcontrol else 0
        if self.steps == 0:
            self.controls.append([])
            self.thetas.append([])
        self.controls[-1].append(action[0])
        self.thetas[-1].append(self._theta)

        self._velocity += ((self._gravity / self._length) * np.sin(self._theta) + force / (self._mass * (self._length ** 2))) * self.timestep
        self._theta += self._velocity
        terminated = (
            (
                self._theta > np.pi/2 or self._theta < -np.pi/2
             )
              ) if self.terminate else False
        
        #reward = 100 * (np.square(np.pi-abs(self._theta - self._setpoint)) - 0.1 * np.square(force))
        reward = (np.pi-abs(self._theta - self._setpoint))/np.pi
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.steps +=1
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
        super().close()
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        if self.plot:
            self._plot()
    def _plot(self):
        for num in range(len(self.controls)):
            fig, ax = plt.subplots(1, 2)

            ax[0].plot(np.linspace(0, len(self.thetas[num]), len(self.thetas[num])), self.thetas[num], label="theta")
            ax[0].set(xlabel='Time', ylabel='theta')

            ax[1].plot(np.linspace(0, len(self.controls[num]), len(self.controls[num])), self.controls[num], label="control amnt")
            ax[1].set(xlabel='Time', ylabel='control force')

            fig.subplots_adjust(wspace=0.3) 
            fig.suptitle('Inverted Pendulum')
            plt.show()