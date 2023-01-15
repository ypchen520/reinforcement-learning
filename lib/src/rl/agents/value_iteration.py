import gym
import torch
import torch.nn.functional as F
import numpy as np

class ValueIterationAgent:
    def __init__(self, env=None, env_name=None, gamma=0.9):
        self.env = env
        self.env_name = env_name
        self.num_states = env.observation_space.n
        self.states_plus = np.arange(self.num_states)
        self.states = np.arange(self.num_states-1)
        self.num_actions = env.action_space.n
        self.actions = np.arange(self.num_actions)
        self.state_values = [0 for _ in range(self.num_states)]
        self.policy = F.one_hot(torch.arange(0, self.num_states) % self.num_actions, num_classes=self.num_actions).numpy() # shape: (self.num_state, self.num_actions) (cliff walking: (48,4))
        self.gamma = gamma
    
    def display_initial_values(self) -> None:
        print(f"initial random state values: {self.state_values}")
        print(f"initial random policy: {self.policy}")
        
    def value_iteration(self, theta=0.1):
        """
        An extreme case of generalized policy iteration
        - policy evaluation + policy improvement
        - one step pf E instead of a sweep over the state space
        - and then greedify (improve) the policy according to the updated value function
        - using Bellman optimality equation as the update rule
        """
        while True:
            delta = 0
            for s in self.states:
                # a sweep
                v = self.state_values[s] # store the old value
                self.bellman_optimality_update(s) # updates self.state_values[s]
                delta = max(delta, abs(v-self.state_values[s])) # keep track of the maximum difference between the old state-value and the new state-value
            if delta < theta:
                break
        # we need another greedify step (this step was taken every time we call bellman_optimality_update(s))
        for s in self.states:
            self.q_greedify_policy(s)
        return self.policy

    def q_greedify_policy(self, state=None) -> None:
        """
        This function performs a policy improvement step.
        Modify self.policy to be greedy w.r.t the q-values induced by self.state_values
        Formula: pi = argmax_a sigma (p(s',r|s,a)[r + gamma * V(s')])
        """
        # the available actions in the input state: A(s)
        # A(s) is not specifically defined for the cliff walking environment because all actions are available in each state
        q_values = [0] * self.num_actions # to select the action that has the largest q value
        for a, a_prob in enumerate(self.policy[state]):
            (probability, next_state, reward, terminated) = self.env.P[state][a][0]
            # for the cliff walking environment, there's only one possible transition for each state-action pair
            q_values[a] = probability * (reward + self.gamma * self.state_values[next_state])
        greedy_action = np.argmax(q_values)
        for i in range(len(self.policy[state])):
            if i == greedy_action:
                self.policy[state][i] = 1
            else:
                self.policy[state][i] = 0

    def bellman_optimality_update(self, state=None):
        """
        Modify state_values according to the Bellman optimality update equation.
        Find maximum over actions instead of considering all actions
        """
        max_q = float("-inf")
        for a in self.actions:
            # available actions: 0, 1, 2, 3
            # for the cliff walking environment, there's only one possible transition for each state-action pair
            # transition: [(probability, next_state, reward, terminated)]
            (probability, next_state, reward, terminated) = self.env.P[state][a][0]
            q = probability * (reward + self.gamma * self.state_values[next_state])
            max_q = max(max_q, q)
        self.state_values[state] = max_q
        
    def __str__(self) -> str:
        return f"The {self.__class__.__name__} agent is playing in the {self.env_name} environment. The enviroment has {self.num_states} states. The agent has {self.num_actions} possible actions"