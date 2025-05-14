import numpy as np
import random
from abc import ABC, abstractmethod
import copy

class RLWrapper(ABC):
    
    def __init__(self, env):
        self.env = env
        self.state_action_pairs = self.generate_sap_indices()
        self.Q = np.zeros((19, 19, 4, 3)) #(19, 19, 4,  3) (x, y, direction, action) (x, y, d) <- state
        self.seeds = [48, 24,  1024, 8888]
        # You need to set your hyperparams in child class

    @abstractmethod
    def episode(self):
        """
        Implement in your class
        """
        pass

    def trial(self, episode_count : int) -> None:
        """
        Runs episodes on same env
        Resets env every trial for 50 trials
        """
        for s in self.seeds:
            self.reset_env(s)
            for _ in range(episode_count):
                self.episode()
                self.restore_init_env_state(s)


    def format_state(self, obs) -> tuple[int]:
        """
        State should be in (x, y, d)
        """
        agent = np.where(obs['image'][:, :, 2] == 10)
        return (int(agent[0][0]), int(agent[1][0]), int(obs['direction']))
    
    
    def best_action(self, state : tuple[int], epsilon : float) -> int:
        """
        use epsilon greedy to generate the next best action based on Q values
        """
        exploit_prob = 1 - epsilon + epsilon / 3
        if random.random() < exploit_prob:
            #exploit
            return (np.argmax(self.Q[state]) + 2) % 3 # + 2 % 3 is done to make forward the main move instead of left
        else:
            #explore CHANGE: do we need to remove the best action?
            return random.choice([0,1,2])
        
    def generate_sap_indices(self) -> np.ndarray:
        """
        Generates all state action pair indices
        """
        i, j, k = np.indices((19, 19, 4)) # (x, y, d)
        i = np.repeat(i.flatten(), 3)
        j = np.repeat(j.flatten(), 3)
        k = np.repeat(k.flatten(), 3)

        actions = np.tile(np.arange(3), 19 * 19 * 4)
        state_action_pairs = np.stack((i, j, k, actions), axis=1)
        return state_action_pairs
    
    def restore_init_env_state(self, s : int) -> None:
        """
        Used to reset episode to og starting locations
        Doesn't reset Q values
        """
        obs, _ = self.env.reset(seed=s)
        self.obs = obs['image']
    
    def reset_env(self, s : int) -> None:
        """
        Reset Q values and four rooms environment
        Randomizes acording to seed (s) the agent , goal, and wall locations
        """
        self.Q = np.zeros((19, 19, 4, 3)) #(19, 19, 3) (x, y, direction, action) (x, y, d) <- state
        obs, _ = self.env.reset(seed=s)
        self.obs = obs['image']
        self.s_0 = self.format_state(obs)   
