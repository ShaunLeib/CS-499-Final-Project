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


"""
obs['image'].type = np.ndarray
obs['image'].shape = (19, 19, 3)
(X, Y, OBJECT_IDX)
"""





def main() -> None:

    # Make the environment
    env = gym.make("MiniGrid-FourRooms-v0")

    env_obs = SymbolicObsWrapper(env)
    obs,_ = env_obs.reset()

    print(obs["direction"])
    print(obs['mission'])
    print(f"{obs['image'].shape = } ")
    agent = np.where(obs['image'][:, :, 2] == 10)
    goal = np.where(obs['image'][:, :, 2] == 8)
    print(f"{agent =}")
    print(f"{goal = }")
    q_learning_lambda = QLearningLambda(obs['image'], agent, goal)




if __name__ == "__main__":
    main()