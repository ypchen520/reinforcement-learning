import gym
from lib.src.rl.utils.utils import Utils
from lib.src.rl.utils.visualize import Visualization as Vis
from lib.src.rl.agents.q_learning import QLearningAgent
from lib.src.rl.agents.sarsa import Sarsa
from lib.src.rl.agents.expected_sarsa import ExpectedSarsa
from lib.src.rl.agents.value_iteration import ValueIterationAgent

# env = gym.make("Blackjack-v1", sab=True)
"""
CliffWalking-v0
0: Move up
1: Move right
2: Move down
3: Move left
"""
if __name__ == "__main__":
    spacer = Utils.get_spacer(":crocodile:", 7)
    env = gym.make("CliffWalking-v0")
    env.reset()
    
    ## Value Iteration Agent Demo
    print(spacer)
    print(f"Value Iteration Agent Demo")
    v = ValueIterationAgent(env, "CliffWalking-v0")
    print(v)
    policy = v.value_iteration(0.001)
    Vis.visualize("CliffWalking-v0", env.shape, v.policy)

    ## Q-Learning Agent Demo
    print(spacer)
    env.reset()
    agent_info = {
        "num_actions": env.action_space.n, 
        "num_states": env.observation_space.n, 
        "epsilon": 0.1, 
        "alpha": 0.5, 
        "gamma": 1.0
    }
    num_episode = 500

    print(f"Q-Learning Agent Demo")
    q = QLearningAgent(env, "CliffWalking-v0", agent_info)
    print(q)
    for i in range(num_episode):
        observation, info = env.reset()
        # print(f"start state: {observation}, info: {info}")
        q.agent_step(observation)
    Vis.visualize("CliffWalking-v0", env.shape, q.q)

    print(f"Sarsa Agent Demo")
    s = Sarsa(env, "CliffWalking-v0", agent_info)
    print(s)
    for i in range(num_episode):
        observation, info = env.reset()
        # print(f"start state: {observation}, info: {info}")
        s.agent_step(observation)
    Vis.visualize("CliffWalking-v0", env.shape, s.q)

    print(f"ExpectedSarsa Agent Demo")
    es = ExpectedSarsa(env, "CliffWalking-v0", agent_info)
    print(es)
    for i in range(num_episode):
        observation, info = env.reset()
        # print(f"start state: {observation}, info: {info}")
        es.agent_step(observation)
    Vis.visualize("CliffWalking-v0", env.shape, es.q)
