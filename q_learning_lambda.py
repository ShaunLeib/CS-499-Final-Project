import numpy as np
from rl_wrapper import RLWrapper

class QLearningLambda(RLWrapper):
    def __init__(self, obs, agent, goal):
        super().__init__(obs, agent, goal)