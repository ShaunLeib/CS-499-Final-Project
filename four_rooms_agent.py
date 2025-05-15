import gymnasium as gym
import numpy as np
from minigrid.wrappers import SymbolicObsWrapper
from q_learning_lambda import QLearningLambda

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


def main() -> None:

    # Make the environment
    env = gym.make("MiniGrid-FourRooms-v0", max_steps = 100) # add  render_mode = "human" for visual

    env = SymbolicObsWrapper(env)
    # obs, _ = env.reset()

    # random_episode(env, 100)
    num_trials = 4
    num_episodes = 100
    q_learning_agent = QLearningLambda(env, num_trials, num_episodes, randomize = False)
    q_learning_agent.trial()
    q_learning_agent.plot_learning_curve()
    env.close()




if __name__ == "__main__":
    main()