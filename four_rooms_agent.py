import sys
import gymnasium as gym
import numpy as np
from minigrid.wrappers import SymbolicObsWrapper
from sarsa import SARSA
from q_learning_lambda import QLearningLambda
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_learning_curve(agent1, agent2 = None):
    """
    Calc average returns across episodes and plot
    """
    def filtered_avg(R):
        # Replace 0 or negative rewards with np.nan, then compute nanmean
        R_filtered = np.where(R > 0, R, np.nan)
        return np.nanmean(R_filtered, axis=0)

    # avg_r_1 = np.mean(agent1.R, axis = 0)
    avg_r_1 = filtered_avg(agent1.R)
    plt.plot(avg_r_1, label = agent1.__class__.__name__)
    plt.title(f"{agent1.__class__.__name__} Learning Curve")

    if agent2 is not None:
        plt.title("SARSA vs Q-learning λ Learning Curve")
        avg_r_2 = np.mean(agent2.R, axis = 0)
        plt.plot(avg_r_2, label = agent2.__class__.__name__)

    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0.45)  # Set minimum y-axis value
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
    seed = sarsa_agent.set_seed()
    sarsa_agent.trial()
    _ = input("Done: Enter")
    sarsa_agent.visual(final_env)
    train_env.close()
    final_env.close()

    # Q-Learning-Lambda
    print("\nLambda Q-Learning")
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000)
    train_env = SymbolicObsWrapper(train_env)
    final_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000, render_mode = "human") # add  render_mode = "human" for visual
    final_env = SymbolicObsWrapper(final_env)

    q_learning_agent = QLearningLambda(train_env, num_trials, num_episodes, randomize = False)
    q_learning_agent.set_seed(seed) #ensure Q-learning and SARSA use the same seed
    q_learning_agent.trial()
    _ = input("Done: Enter")
    q_learning_agent.visual(final_env)
    train_env.close()
    final_env.close()

    plot_learning_curve(q_learning_agent, sarsa_agent) 


def part_b(num_trials = 1, num_episodes = 2000):
    # SARSA
    print('SARSA')
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000)
    train_env = SymbolicObsWrapper(train_env)
    sarsa_agent = SARSA(train_env, num_trials, num_episodes, randomize = True)
    seed = sarsa_agent.set_seed() # set set foor Q lambda to use same seed
    
    train = 1 # 1 for SARSA training mode, 0 for SARSA re-test (from npy Q table file)
    if train:
        sarsa_agent.trial()
        np.save("sarsa_qtable.npy", sarsa_agent.Q)
        plot_learning_curve(sarsa_agent)

        train_epsilon = 0.9 # defined in sarsa.py, this one is just for the dynamic file name
        train_params = (f"rse{sarsa_agent.rand_every_n_episodes}_epi{num_episodes}_a{sarsa_agent.alpha}_e{train_epsilon}_d{sarsa_agent.decay}").replace('.', '')
        filename = f"sarsa_qtable_{train_params}_steps{sarsa_agent.env.unwrapped.step_count}_{int(time.time())}.npy"
        np.save(f"trains/{filename}", sarsa_agent.Q)
    else:
        sarsa_agent.Q = np.load("trains/sarsa_qtable_rse80_epi4000_a00012_e09_d099927_steps1364_1749036041.npy")
        test_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 2000, render_mode = "human") # add  render_mode = "human" for visual
        test_env = SymbolicObsWrapper(test_env)

        test_epsilon = 0.5
        test_seed = 14
        sarsa_agent.test(test_epsilon, test_seed)
        # sarsa_agent.visual(test_env, test_epsilon)
    

    s_rewards = []
    for _ in range(50):
        sarsa_agent.set_seed(14) # Random seed or use seed 14
        sarsa_agent.test()
        s_rewards.append(sarsa_agent.test_reward)
    print("AVG SARSA TEST REWARD:", np.mean(s_rewards))

    train_env.close()

    # Q-learning-lambda
    print("\nLambda Q-Learning")
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 4000)
    train_env = SymbolicObsWrapper(train_env)

    q_learning_agent = QLearningLambda(train_env, num_trials, num_episodes, randomize = True)
    # q_learning_agent.Q = np.load("qlamb_qtable.npy")
    q_learning_agent.trial()
    plot_learning_curve(q_learning_agent)
    # np.save("qlamb_qtable.npy", q_learning_agent.Q)  

    rewards = []
    for _ in range(50):
        q_learning_agent.set_seed(seed) # Random seed or use seed 14
        q_learning_agent.test()
        rewards.append(q_learning_agent.test_reward)
    print("AVG Q-L-L TEST REWARD:", np.mean(rewards))

    train_env.close()

def generate_heat_map():
    agent_q_values = np.load("sarsa_qtable.npy")
    V = agent_q_values.max(axis=-1)  # Shape becomes (19, 19, 4)

    # For each direction, create a separate heatmap
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    directions = ['N', 'E', 'S', 'W']

    for i, ax in enumerate(axs.flat):
        sns.heatmap(V[:, :, i], ax=ax, cmap='viridis', cbar=True)
        ax.set_title(f'Value Function - Facing {directions[i]}')
        ax.invert_yaxis()  # Optional: origin at bottom-left
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.tight_layout()
    plt.show()


def main() -> None:
    part_a()
    # part_b(1, 4000)
    # generate_heat_map()

if __name__ == "__main__":
    main()