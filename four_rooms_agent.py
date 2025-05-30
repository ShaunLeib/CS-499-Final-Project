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

def plot_learning_curve(agent1, agent2):
    """
    Calc average returns across episodes and plot
    """
    avg_r_1 = np.mean(agent1.R, axis = 0)
    avg_r_2 = np.mean(agent2.R, axis = 0)
    plt.plot(avg_r_1, label="Q-Learning-Lambda")
    plt.plot(avg_r_2, label="SARSA")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

def part_a(num_trials = 50, num_episodes = 50):
    #SARSA
    print('SARSA')
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000)
    train_env = SymbolicObsWrapper(train_env)
    # final_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000, render_mode = "human") # add  render_mode = "human" for visual
    # final_env = SymbolicObsWrapper(final_env)

    sarsa_agent = SARSA(train_env, num_trials, num_episodes, randomize = False)
    seed = sarsa_agent.set_seed()
    sarsa_agent.trial()
    # sarsa_agent.visual(final_env)
    train_env.close()
    # final_env.close()

    # Q-Learning-Lambda
    print("\nLambda Q-Learning")
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000)
    train_env = SymbolicObsWrapper(train_env)
    final_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000, render_mode = "human") # add  render_mode = "human" for visual
    final_env = SymbolicObsWrapper(final_env)

    q_learning_agent = QLearningLambda(train_env, num_trials, num_episodes, randomize = False)
    q_learning_agent.set_seed(seed) #ensure Q-learning and SARSA use the same seed
    q_learning_agent.trial()
    q_learning_agent.visual(final_env)
    train_env.close()
    final_env.close()

    plot_learning_curve(q_learning_agent, sarsa_agent) # replace None with q_learning_agent or sarsa_agent

def part_b(num_trials = 1, num_episodes = 5000):
    # SARSA
    print('SARSA')
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 20000)
    train_env = SymbolicObsWrapper(train_env)
    test_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 20000, render_mode = "human") # add  render_mode = "human" for visual
    test_env = SymbolicObsWrapper(test_env)

    sarsa_agent = SARSA(train_env, num_trials, num_episodes, randomize = True)
    sarsa_agent.trial()
    seed = sarsa_agent.set_seed()

    sarsa_agent.test()
    if sarsa_agent.test_reward > 0.95:
        sarsa_agent.visual(test_env, 0.6)
    train_env.close()
    test_env.close()

    # Q-learning-lambda
    print("\nLambda Q-Learning")
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 20000)
    train_env = SymbolicObsWrapper(train_env)
    test_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 20000, render_mode = "human") # add  render_mode = "human" for visual
    test_env = SymbolicObsWrapper(test_env)

    q_learning_agent = QLearningLambda(train_env, num_trials, num_episodes, randomize = True)
    q_learning_agent.trial()

    q_learning_agent.set_seed(seed) # test both agents using same seed
    q_learning_agent.test()
    if q_learning_agent.test_reward > 0.95:
        q_learning_agent.visual(test_env, 0.6)
    train_env.close()
    test_env.close()


def main() -> None:
    part_a(1, 20)
    # part_b(1, 10)


if __name__ == "__main__":
    main()