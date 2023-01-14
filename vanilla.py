import numpy as np
import gym
import torch
import torch.nn.functional as F
import copy
import tcod

import random

class Utils:
    @staticmethod
    def get_CP437_ch(idx):
        return chr(tcod.tileset.CHARMAP_CP437[idx])
    
    @staticmethod
    def visualize(env_name, env_shape, policy):
        if env_name == "CliffWalking-v0":
            rows, cols = env.shape
            buffer = np.zeros(
                shape=(rows+2, cols+2),
                dtype=tcod.console.Console.DTYPE,
                order="C",
            )
            for i in range(rows+2):
                for j in range(cols+2):
                    if i == 0 and j == 0:
                        buffer[i][j] = ord(Utils.get_CP437_ch(201))
                    elif i == rows+1 and j == 0:
                        buffer[i][j] = ord(Utils.get_CP437_ch(200))
                    elif i == 0 and j == cols+1:
                        buffer[i][j] = ord(Utils.get_CP437_ch(187))
                    elif i == rows+1 and j == cols+1:
                        buffer[i][j] = ord(Utils.get_CP437_ch(188))
                    elif i == 0 or i == rows+1:
                        buffer[i][j] = ord(Utils.get_CP437_ch(205))
                    elif j == 0 or j == cols+1:
                        buffer[i][j] = ord(Utils.get_CP437_ch(186))
                    elif i == rows and j not in {1, cols}:
                        buffer[i][j] = ord(Utils.get_CP437_ch(223))
                    elif i == rows and j == cols:
                        buffer[i][j] = ord(Utils.get_CP437_ch(1))
                    else:
                        state = (i-1)*cols + (j-1)
                        action = np.argmax(policy[state])
                        arrow_idx = 24
                        if action == 0:
                            arrow_idx = 24
                        elif action == 1:
                            arrow_idx = 26
                        elif action == 2:
                            arrow_idx = 25
                        elif action == 3:
                            arrow_idx = 27
                        buffer[i][j] = ord(chr(tcod.tileset.CHARMAP_CP437[arrow_idx]))
            print(tcod.console.Console(rows+2, cols+2, order="C", buffer=buffer))

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
    
    def visualize(self) -> None:
        Utils.visualize(self.env_name, self.env.shape, self.policy)
        
    def __str__(self) -> str:
        return f"The {self.__class__.__name__} agent is playing in the {self.env_name} environment. The enviroment has {self.num_states} states. The agent has {self.num_actions} possible actions"

# env = gym.make("Blackjack-v1", sab=True)
"""
CliffWalking-v0
0: Move up
1: Move right
2: Move down
3: Move left
"""
if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    env.reset()
    # v = ValueIterationAgent(env, "CliffWalking-v0")
    # policy = v.value_iteration(0.001)
    # v.visualize()

    agent_info = {
        "num_actions": env.action_space.n, 
        "num_states": env.observation_space.n, 
        "epsilon": 0.1, 
        "alpha": 0.5, 
        "gamma": 1.0
    }
    q = QLearningAgent(env, "CliffWalking-v0", agent_info)
    num_episode = 100
    for i in range(num_episode):
        observation, info = env.reset()
        # print(f"start state: {observation}, info: {info}")
        q.agent_step(observation)
    # print(q.q)
    Utils.visualize("CliffWalking-v0", env.shape, q.q)

"""Notes
(Not sure if this makes sense)
I'm considering implementing a base class that handles the environment information.
The Agent classes can inherit this base class, so the subclasses only have to deal with the learning part
"""