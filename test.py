import inv_pend_env
import gymnasium
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

class PIDController:
    def __init__(self, setpoint, kp, ki, kd):
        self.controlhistory = []
        self.integral = 0
        self.times = []
        self.previous_error = 0
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def control(self, t, y):
        self.times.append(t)

        error = self.setpoint - y
        if len(self.times) > 1:
            dt = self.times[-1] - self.times[-2]
            if dt > 0:
                self.integral += error * dt
                derivative = (error - self.previous_error) / dt
            else:
                derivative = 0
        else:
            dt = 0
            derivative = 0
        if self.ki !=0:
          self.integral = max(-1/self.ki, min(1/self.ki, self.integral))

        self.previous_error = error

        u = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.controlhistory.append(u)

        return u
# known good
kp = 10
ki = 0
kd = 8

def inverted_pendulum(t, y, setpoint, kp, ki, kd):
    global gravity, length, mass
    u = controller.control(t, y)

    controls[-1].append(u)
    controls[-1].append(u)

    theta, velocity = y
    dtheta_dt = velocity
    dvelocity_dt = (gravity / length) * np.sin(theta) + u / (mass * (length ** 2))
    print(dtheta_dt, dvelocity_dt)
    return dtheta_dt, dvelocity_dt



gravity = 9.81
length = 1.0
mass = 1.0

env = gymnasium.make('inv_pend_env/inv_pendulum_v0', seed=0, plot=True, disallowcontrol=False, terminate=False, gravity=gravity, length=length, mass=mass, render_mode="human")
observation, info = env.reset()
controller = PIDController(0, kp, ki, kd)
timestep = 0
control = 0
thetas = [[-1]]
controls = [[]]
for _ in range(125):
    #time.sleep(0.5)
    controls[-1].append(control)
    controls[-1].append(control)
    observation, reward, terminated, truncated, info = env.step([control])
    thetas[-1].append(observation[0])
    print("Reward: " + str(reward))
    control = controller.control(timestep, observation[0])
    if terminated or truncated:
        thetas.append([])
        controls.append([])
        observation, info = env.reset()
        thetas[-1].append(observation[0])
        timestep = 0
        controller = PIDController(0, kp, ki, kd)
        control = controller.control(timestep, observation[0])
    
    timestep += 1
env.close()