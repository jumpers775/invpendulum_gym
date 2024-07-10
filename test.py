import inv_pend_env
import gymnasium
import time
import numpy as np

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
kd = 2


theta = np.pi/4
env = gymnasium.make('inv_pend_env/inv_pendulum_v0', seed=0, plot=True, disallowcontrol=False, terminate=False)
observation, info = env.reset()
controller = PIDController(0, kp, ki, kd)
timestep = 0
control = 0
for _ in range(200):
    #time.sleep(0.5)
    observation, reward, terminated, truncated, info = env.step([control])
    print("Reward: " + str(reward))
    control = controller.control(timestep, observation[0])
    if terminated or truncated:
        observation, info = env.reset()
        timestep = 0
        controller = PIDController(0, kp, ki, kd)
        control = controller.control(timestep, observation[0])

    timestep += 1
env.close()