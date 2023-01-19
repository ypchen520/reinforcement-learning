import torch
import torch.nn.functional as F
import numpy as np
import random
from .td_learning import TDLearningAgent

class Sarsa(TDLearningAgent):
    def __init__(self, env=None, env_name=None, agent_init_info=None):
        super(Sarsa, self).__init__(env, env_name, agent_init_info)

    def update(self, state, next_state, action, reward, is_terminal):
        """
        Sarsa: on-policy using epsilon-greedy
        """
        target = reward
        if not is_terminal:
            # the reward is the only update target in the terminal state since the q value is 0 in the terminal state
            on_policy_action = super(Sarsa, self).epsilon_greedy(next_state)
            target += self.gamma * self.q[next_state][on_policy_action]
        self.q[state][action] = self.q[state][action] + self.alpha * (target - self.q[state][action])

    def __str__(self) -> str:
        return f"The {self.__class__.__name__} agent is playing in the {self.env_name} environment. The enviroment has {self.num_states} states. The agent has {self.num_actions} possible actions"