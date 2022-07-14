import math
import numpy as np
import gym


class SmartMouseEnv(gym.Env):

    def __init__(self, goal_velocity=0):
        self.min_action = -1.0
        self.max_action = 1.0

        # The move of the mouse is x-axis,y-axis and z-axis(click or not).
        self.action_space = gym.spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(3,),
            dtype=np.float32
        )

        self.height = 100
        self.width = 100

        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8
        )

        self.reset()


    def step(self, action):
        pass
        return self.state, reward, done, {}

    def reset(self):
        pass
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)
