import gym
import torch
import torch.nn.functional as F
import numpy as np
import random

class QLearningAgent:
    def __init__(self, env=None, env_name=None, agent_init_info=None):
        """
        Args:
        agent_init_info (dict):
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            alpha (float): The step-size,
            gamma (float): The discount factor,
        }

        """
        # self.num_states = agent_init_info['num_states']
        # self.num_actions = agent_init_info['num_actions']
        # self.q = np.zeros((self.num_states, self.num_actions))
        # self.epsilon = agent_init_info['epsilon']
        # self.alpha = agent_init_info['alpha'] # step size
        # self.gamma = agent_init_info['gamma'] # discount

        self.env = env
        self.env_name = env_name

        ## Using self.__dict__, probably not that readable
        for key, val in agent_init_info.items():
            self.__dict__[key] = val

        self.q = np.zeros((self.num_states, self.num_actions))
    
    def greedy(self, state):
        """
        Args:
            state (int): the state of interest
        Returns:
            action (int): the greedy action w.r.t self.q[state]
        """
        return np.argmax(self.q[state])
    
    def epsilon_greedy(self, state) -> int:
        """
        Args:
            state (int): the state of interest
        Returns:
            action (int): 
                1-self.epsilon: the greedy action w.r.t self.q[state]
                self.epsilon: a random action among self.num_actions
        """
        action = self.greedy(state) # greedy
        sample = random.random() # Random float:  0.0 <= x < 1.0
        if sample < self.epsilon:
            ## explore
            action = torch.randint(self.num_actions,(1,)).item() # Python Random's randint() is [a,b], I prefer to use [a,b), which is the case in PyTorch
        return action

    def update(self, state, next_state, action, reward, is_terminal):
        target = 0
        if not is_terminal:
            greedy_action = self.greedy(next_state)
            target = reward + self.gamma * self.q[next_state][greedy_action]
        else:
            target = reward + self.gamma * 0
        self.q[state][action] = self.q[state][action] + self.alpha * (target - self.q[state][action])
            
    def agent_step(self, observation):
        """
        Args:
            observation (int): the initial state of this episode
        """
        is_terminal = False
        state = observation
        while not is_terminal:
            action = self.epsilon_greedy(state)
            # print(f"taking action {action} in state {state}")
            next_state, reward, terminated, truncated, info = self.env.step(action)
            # print(f"getting to state {next_state} with reward: {reward}")
            if terminated or reward == -100: #TODO: define specific terminal states for different environments
                is_terminal = True
                self.agent_end(state, action, reward)
            else:
                self.update(state, next_state, action, reward, is_terminal)
            state = next_state
    
    def agent_end(self, prev_state, action, reward):
        """
        Run a last update for this episode.
        Args:
            prev_state (int)
            action (int)
            reward (float): can only be +1 or -100 for CliffWalking
        """
        self.update(prev_state, -1, action, reward, True) #TODO: using -1 to represent the terminal states for now


    def __str__(self) -> str:
        return f"The {self.__class__.__name__} agent is playing in the {self.env_name} environment. The enviroment has {self.num_states} states. The agent has {self.num_actions} possible actions"