import sys
import gymnasium as gym
import numpy as np
from minigrid.wrappers import SymbolicObsWrapper
from sarsa import SARSA
from q_learning_lambda import QLearningLambda
import matplotlib.pyplot as plt

# REMOVE ME FOR GITHUB (I'll no doubt forget)
from ping_me import ping_dc # notify me on discord when training is complete
import time

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

def plot_learning_curve(agent1, agent2=None):
    """
    Calc average returns across episodes and plot
    """
    avg_r_1 = np.mean(agent1.R, axis = 0)
    plt.plot(avg_r_1, label=agent1.__class__.__name__)

    if agent2 is not None:
        avg_r_2 = np.mean(agent2.R, axis=0)
        plt.plot(avg_r_2, label=agent2.__class__.__name__)

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("SARSA vs Q-learning λ")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def part_a(num_trials = 50, num_episodes = 50):
    #SARSA
    print('SARSA')
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000)
    train_env = SymbolicObsWrapper(train_env)
    final_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000, render_mode = "human") # add  render_mode = "human" for visual
    final_env = SymbolicObsWrapper(final_env)

    sarsa_agent = SARSA(train_env, num_trials, num_episodes, randomize = False)
    seed = sarsa_agent.set_seed(15)
    # sarsa_agent.trial()
    # sarsa_agent.visual(final_env)
    train_env.close()
    final_env.close()

    #TEMP
    num_trials = 1

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
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000)
    train_env = SymbolicObsWrapper(train_env)
    # train_env.reset(seed=1499) # omit line for random seed
    test_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000)#, render_mode = "human") # add  render_mode = "human" for visual
    test_env = SymbolicObsWrapper(test_env)
    # test_env.reset(seed=1499) # omit line for random seed

    # 1 for training mode, 0 for test (from file)
    train = 0

    sarsa_agent = SARSA(train_env, num_trials, num_episodes, randomize = True)
    train_epsilon = sarsa_agent.epsilon # REMOVE ME FOR GITHUB required for ping_dc because it gets overwritten with `test_epsilon = 0.05`
    if train:
        sarsa_agent.trial()
    else:
        sarsa_agent.Q = np.load("trains/sarsa_qtable_epi4000_a00008_e09_d1_steps2000_1748900091.npy")
    seed = sarsa_agent.set_seed() # set set foor Q lambda to use same seed?

    test_epsilon = 0.05
    sarsa_agent.test(test_epsilon, 14)
    # sarsa_agent.test(test_epsilon, 15)
    # sarsa_agent.test(test_epsilon, 16)
    # sarsa_agent.test(test_epsilon, 17)
    # sarsa_agent.test(test_epsilon, 18)
    # sarsa_agent.test(test_epsilon, 19)
    # sarsa_agent.test(test_epsilon, 20)
    # sarsa_agent.test(test_epsilon, 21)
    
    if train:
        train_params = (f"epi{num_episodes}_a{sarsa_agent.alpha}_e{train_epsilon}_d{sarsa_agent.decay}").replace('.', '')
        filename = f"sarsa_qtable_{train_params}_steps{sarsa_agent.env.unwrapped.step_count}_{int(time.time())}.npy"
        np.save(f"trains/{filename}", sarsa_agent.Q)
        ping_dc( # REMOVE ME FOR GITHUB
            f"  Paramters of the run:\n"
            f"    `{filename}`\n"
            f"    Episodes: `{num_episodes}`\n"
            f"    Alpha:    `{sarsa_agent.alpha}`\n"
            f"    Epsilon:  `{train_epsilon}`\n"
            f"    Decay:    `{sarsa_agent.decay}`\n"
            f"    Steps to goal: `{sarsa_agent.env.unwrapped.step_count}`\n  ---"
        )
    
    # if True: # will be: if sarsa_agent.test_reward > 0.95:
    if (sarsa_agent.env.unwrapped.step_count < 2000):
        visual_env = gym.make("MiniGrid-FourRooms-v0", max_steps=2000, render_mode="human") # visual_env created purely so it can have human visuals while test_env does not
        visual_env = SymbolicObsWrapper(visual_env)
        sarsa_agent.visual(visual_env, test_epsilon)
        visual_env.close()

    train_env.close()
    test_env.close()
    
    # plot_learning_curve(sarsa_agent)


    # Q-learning-lambda
    # print("\nLambda Q-Learning")
    # train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 20000)
    # train_env = SymbolicObsWrapper(train_env)
    # test_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 20000, render_mode = "human") # add  render_mode = "human" for visual
    # test_env = SymbolicObsWrapper(test_env)

    # q_learning_agent = QLearningLambda(train_env, num_trials, num_episodes, randomize = True)
    # q_learning_agent.trial()
    # np.save("q_learning_agent.npy", q_learning_agent.Q)
    # q_learning_agent.Q = np.load("q_learning_agent.npy")

    # q_learning_agent.set_seed(seed) # test both agents using same seed
    # q_learning_agent.test()
    # if q_learning_agent.test_reward > 0.95:
    #     q_learning_agent.visual(test_env, 0.6)

    # train_env.close()
    # test_env.close()


def main() -> None:
    # part_a(10, 80)
    part_b(1, 12000)

if __name__ == "__main__":
    main()