import sys
import gymnasium as gym
import numpy as np
from minigrid.wrappers import SymbolicObsWrapper
from sarsa import SARSA
from q_learning_lambda import QLearningLambda
import matplotlib.pyplot as plt

#Actions = {"left" : 0, "right" : 1, "forward" : 2} -> Actions = {"left" : 2, "right" : 1, "forward" : 0}

# Map of object type to integers
# OBJECT_TO_IDX = {
#     "empty": 1,
#     "wall": 2,
#     "goal": 8,
#     "agent": 10,
# }

# DIR_TO_IDX  = {
#     "right" : 0,
#     "down" : 1,
#     "left" : 2,
#     "up" : 3
# }

"""
obs['image'].type = np.ndarray
obs['image'].shape = (19, 19, 3)
(X, Y, OBJECT_IDX)
"""

def random_episode(env, max_steps):
    for i in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

def plot_learning_curve(agent1, agent2) -> None:
    """
    Calc average returns across episodes and plot
    """
    avg_r_1 = np.mean(agent1.R, axis = 0)
    avg_r_2 = np.mean(agent2.R, axis = 0)
    plt.plot(avg_r_1)
    plt.plot(avg_r_2)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

def part_a():
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000)
    train_env = SymbolicObsWrapper(train_env)
    final_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000, render_mode = "human") # add  render_mode = "human" for visual
    final_env = SymbolicObsWrapper(final_env)

<<<<<<< HEAD
    num_trials = 50
    num_episodes = 50
=======
    num_trials = 1
    num_episodes = 100

    sarsa_agent = SARSA(train_env, num_trials, num_episodes, randomize = False)
    sarsa_agent.trial()
    sarsa_agent.visual(final_env)

>>>>>>> 332e90e8ad98f1413006779e7bd4e2091fb13546
    q_learning_agent = QLearningLambda(train_env, num_trials, num_episodes, randomize = False)
    q_learning_agent.trial()
    q_learning_agent.visual(final_env)

    plot_learning_curve(q_learning_agent, sarsa_agent) # replace None with q_learning_agent or sarsa_agent

def part_b():
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 20000)
    train_env = SymbolicObsWrapper(train_env)
    num_trials = 1
    num_episodes = 10
    q_learning_agent = QLearningLambda(train_env, num_trials, num_episodes, randomize = True)
    q_learning_agent.trial()

    test_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 20000, render_mode = "human") # add  render_mode = "human" for visual
    test_env = SymbolicObsWrapper(test_env)
    q_learning_agent.test(test_env)
    train_env.close()
    test_env.close()


def main() -> None:
    part_a()
    # part_b()

   




if __name__ == "__main__":
    main()