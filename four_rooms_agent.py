import gymnasium as gym
import numpy as np
from minigrid.wrappers import SymbolicObsWrapper
from sarsa import SARSA
from q_learning_lambda import QLearningLambda
import matplotlib.pyplot as plt
import seaborn as sns

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
    avg_r_1 = np.mean(agent1.R, axis = 0)
    
    plt.plot(avg_r_1, label="Q-Learning-Lambda")
    if agent2 is not None:
        avg_r_2 = np.mean(agent2.R, axis = 0)
        plt.plot(avg_r_2, label="SARSA")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
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
    train_env = gym.make("MiniGrid-FourRooms-v0", max_steps = 4000)
    train_env = SymbolicObsWrapper(train_env)

    sarsa_agent = SARSA(train_env, num_trials, num_episodes, randomize = True)
    # sarsa_agent.Q = np.load("sarsa_qtable.npy")
    sarsa_agent.trial()
    plot_learning_curve(sarsa_agent)
    np.save("sarsa_qtable.npy", sarsa_agent.Q)

    for _ in range(50):
        q_learning_agent.set_seed(14) # Random seed or use seed 14
        q_learning_agent.test()
        rewards.append(q_learning_agent.test_reward)
    print("AVG SARSA TEST REWARD:", np.mean(rewards))

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
        q_learning_agent.set_seed(14) # Random seed or use seed 14
        q_learning_agent.test()
        rewards.append(q_learning_agent.test_reward)
    print("AVG Q-L-L TEST REWARD:", np.mean(rewards))

    train_env.close()

def generate_heat_map():
    agent_q_values = np.load("qlamb_qtable.npy")
    V = agent_q_values.max(axis=-1)  # Shape becomes (19, 19, 4)

    # For each direction, create a separate heatmap
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    directions = ['N', 'E', 'S', 'W']

    for i, ax in enumerate(axs.flat):
        sns.heatmap(V[:, :, i], ax=ax, cmap='viridis', cbar=True)
        ax.set_title(f'Value Function - Facing {directions[i]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.tight_layout()
    plt.show()

def main() -> None:
    # part_a()
    part_b()
    generate_heat_map()

if __name__ == "__main__":
    main()