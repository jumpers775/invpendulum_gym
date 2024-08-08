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
kp = 400
ki = 0
kd = 500

def inverted_pendulum(t, y, setpoint, kp, ki, kd):
    global gravity, length, mass
    u = controller.control(t, y)

    theta, velocity = y
    dtheta_dt = velocity
    dvelocity_dt = (gravity / length) * np.sin(theta) + u / (mass * (length ** 2))
    print(dtheta_dt, dvelocity_dt)
    return dtheta_dt, dvelocity_dt



gravity = 9.81
length = 1.0
mass = 1.0



passes = 10
env = gymnasium.make('inv_pend_env/inv_pendulum_v0')
conditions = np.linspace(-np.pi/2, np.pi/2, passes)
observation, info = env.reset(val=conditions[0])
controller = PIDController(0, kp, ki, kd)
timestep = 0
control = controller.control(timestep, observation[0])
starts = []

for i in range(0,passes):
    terminated = False
    truncated = False
    start = observation[0]
    passes = 0

    observation, info = env.reset(val=conditions[i])
    timestep = 0
    controller = PIDController(0, kp, ki, kd)
    control = controller.control(timestep, observation[0])
    while not terminated and not truncated:
        passes +=1
        observation, reward, terminated, truncated, info = env.step([control])
        #print("Reward: " + str(reward))
        control = controller.control(timestep, observation[0])
        if terminated or truncated:
            starts.append(passes==200)
    
    timestep += 1
env.close()

# Plotting conditions vs starts
plt.figure()
plt.scatter(conditions, starts, c=starts, cmap='RdYlGn')
plt.xlabel('Conditions')
plt.ylabel('Starts')
plt.title('Starts vs Conditions')
plt.grid(True)
plt.show()