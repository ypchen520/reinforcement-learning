import gym
from lib.src.rl.utils.utils import Utils
from lib.src.rl.utils.visualize import Visualization as Vis
from lib.src.rl.agents.q_learning import QLearningAgent
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
    policy = v.value_iteration(0.001)
    Vis.visualize("CliffWalking-v0", env.shape, v.policy)

    ## Q-Learning Agent Demo
    print(spacer)
    print(f"Q-Learning Agent Demo")
    env.reset()
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
    Vis.visualize("CliffWalking-v0", env.shape, q.q)

"""Notes
(Not sure if this makes sense)
I'm considering implementing a base class that handles the environment information.
The Agent classes can inherit this base class, so the subclasses only have to deal with the learning part
"""