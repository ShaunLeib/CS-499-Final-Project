import gymnasium as gym
import numpy as np
from minigrid.wrappers import SymbolicObsWrapper
from q_learning_lambda import QLearningLambda

#Actions = {"left" : 0, "right" : 1, "forward" : 2}

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

def episode(env, max_steps):
    for i in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

def trail():
    pass



def main() -> None:

    # Make the environment
    env = gym.make("MiniGrid-FourRooms-v0", render_mode = "human") # add  render_mode = "human" for visual

    env = SymbolicObsWrapper(env)
    obs, _ = env.reset()
    print(type(env))
    print(env)

    print(f"{obs['direction'] =}")
    print(obs['mission'])
    print(f"{obs['image'].shape = } ")
    agent = np.where(obs['image'][:, :, 2] == 10)
    goal = np.where(obs['image'][:, :, 2] == 8)
    print(f"{agent =}")
    print(f"{goal = }")
    episode(env, 100)
        




if __name__ == "__main__":
    main()