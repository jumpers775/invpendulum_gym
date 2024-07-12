import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy import integrate
import datetime
from pathlib import Path




class InvPend(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, 
                 render_mode=None, 
                 setpoint: int | float=0,
                 length: int | float=1, 
                 mass: int | float=1, 
                 gravity: int | float=9.81,
                 plot: bool = False, 
                 seed: int | float = None, 
                 disallowcontrol: bool = False, 
                 timestep: int | float = 0.1,
                 terminate: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._theta = self.np_random.uniform(low=-np.pi/2, high=np.pi/2)
        self._velocity = 0
        self._gravity = gravity
        self._mass = mass
        self._length = length
        self._setpoint = setpoint
        
        self.timestep = timestep
        self.disallowcontrol = disallowcontrol
        self.seed = seed or None
        self.plot = plot
        self.terminate = terminate

        self.action_space = spaces.Box(low=-50, high=50, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None

        self.steps = 0
        self.lasttime = 0
        self.controls = []
        self.thetas = []


        self.thetatimes = []
    def _get_obs(self):
        return np.array([self._theta, self._setpoint - self._theta], dtype=np.float32)
    def _get_info(self):
        return {"distance": self._theta - self._setpoint}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed or self.seed or None)
        self._theta = self.np_random.uniform(low=-np.pi/2, high=np.pi/2)
        self._velocity = 0
        self.steps = 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame(True)
        return observation, info
    def getpendupdate(self,t: int | float, theta: list , u: int | float) -> tuple:
        timestep = t-self.lasttime
        self.lasttime = t
        current_velocity = theta[0]
        dvelocity = ((self._gravity / self._length) * np.sin(theta[1]) + u / (self._mass * (self._length ** 2))) * timestep
        dtheta = current_velocity + dvelocity
        return dvelocity, dtheta
    def step(self, action):
        force = action[0] if not self.disallowcontrol else 0
        if self.steps == 0:
            self.controls.append([])
            self.thetas.append([])
            self.thetatimes.append([])
        self.controls[-1].append(action[0])
        
        
        differential = integrate.solve_ivp(self.getpendupdate, [self.steps, self.steps+1], [self._velocity,self._theta], args=(force,))
        self.thetas[-1] += list(differential.y[1])
        self.thetatimes[-1] += list(differential.t)
        self._velocity = differential.y[0][-1]
        self._theta = differential.y[1][-1]


        terminated = (
                        self._theta > np.pi/2 or self._theta < -np.pi/2
                    ) if self.terminate else False
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


            ax[0].plot(self.thetatimes[num], self.thetas[num], label="theta", color="blue")
            ax[0].set(xlabel='Time', ylabel='theta')


            ax[1].plot(np.linspace(0, len(self.controls[num]), len(self.controls[num])), self.controls[num], label="control amnt")
            ax[1].set(xlabel='Time', ylabel='control force')

            fig.subplots_adjust(wspace=0.3) 
            fig.suptitle('Inverted Pendulum')
            # Save the plot as an image with a name based on the current date and time
            
            # Create the 'graphs' directory if it doesn't exist
            Path("graphs").mkdir(exist_ok=True)
            
            # Generate a unique filename using the current date and time
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"graphs/inverted_pendulum_{timestamp}.png"
            
            # Save the plot to the specified file
            fig.savefig(filename)
            
            # Close the figure to free up resources
            plt.close(fig)
