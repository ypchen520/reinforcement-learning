import torch
import torch.nn.functional as F
import numpy as np
import random
from .td_learning import TDLearningAgent

class QLearningAgent(TDLearningAgent):
    def __init__(self, env=None, env_name=None, agent_init_info=None):
        super(QLearningAgent, self).__init__(env, env_name, agent_init_info)

    def update(self, state, next_state, action, reward, is_terminal):
        target = 0
        if not is_terminal:
            greedy_action = self.greedy(next_state)
            target = reward + self.gamma * self.q[next_state][greedy_action]
        else:
            target = reward + self.gamma * 0
        self.q[state][action] = self.q[state][action] + self.alpha * (target - self.q[state][action])

    def __str__(self) -> str:
        return f"The {self.__class__.__name__} agent is playing in the {self.env_name} environment. The enviroment has {self.num_states} states. The agent has {self.num_actions} possible actions"