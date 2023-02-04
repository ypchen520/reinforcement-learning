import torch
import numpy as np
import random
import abc

class TDLearningAgent(metaclass=abc.ABCMeta):
    """
    Superclass that is used to define common attributes and methods for basic Temporal Difference Methods
    The difference between the Sarsa, Q-Learning, and Expected Sarsa is just the update rule
    """
    def __init__(self, env=None, env_name=None, agent_init_info=None):
        """
        Args:
        agent_init_info (dict):
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The initial epsilon parameter for exploration,
            epsilon_decay (float) = The decay for epsilon,
            final_epsilon (float): The final epsilon value,
            alpha (float): The step-size,
            gamma (float): The discount factor,
        }

        """
        self.env = env
        self.env_name = env_name

        ## Using self.__dict__, probably not that readable
        for key, val in agent_init_info.items():
            self.__dict__[key] = val

        self.q = np.zeros((self.num_states, self.num_actions))
    
    @abc.abstractmethod
    def update(self, state, next_state, action, reward, is_terminal):
        raise NotImplementedError

    def expected_q(self, state, policy: str):
        expected_q_value = 0
        if policy == "epsilon_greedy":
            greedy_action = np.argmax(self.q[state])
            for action in range(self.num_actions):
                if action == greedy_action:
                    expected_q_value += (self.epsilon / self.num_actions + 1 - self.epsilon) * self.q[state][action]
                else:
                    expected_q_value += self.epsilon / self.num_actions * self.q[state][action]
        return expected_q_value
                

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
    
    def agent_step(self, observation):
        """
        The behavior policy is always epsilon-greedy
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
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        return self.epsilon