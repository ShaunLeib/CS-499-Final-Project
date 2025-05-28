import numpy as np
import random
from abc import ABC, abstractmethod

class RLWrapper(ABC):
    def __init__(self, env, trail_count : int, episode_count : int, randomize : bool):
        self.env = env
        self.state_action_pairs = self.generate_sap_indices()
        self.episode_count = episode_count
        self.randomize = randomize
        self.trail_count = trail_count
        self.Q = np.zeros((19, 19, 4, 3)) #(19, 19, 4, 3) (x, y, direction, action) (x, y, d) <- state
        self.R = np.zeros((self.trail_count, self.episode_count)) # save the rewards for each episode for each trial for learning curve graph
        self.gamma = 0.95
        self.set_seed()
        # You need to set your specific hyperparams in child class


    @abstractmethod
    def episode(self, t: int, i : int) -> None:
        """
        Implement in your class
        """
        pass


    def set_seed(self, s : int = None):
        """
        Creates a random seed if one is not provided 
        Allows for setting of a defined seed or generation of new one
        Returns:
            provided seed or randomly generated one
        """
        if s is None:
            self.seed = int(np.random.randint(0, 5000, size = 1))
        else:
            self.seed = s
        return self.seed


    def trial(self) -> None:
        """
        Randomize = False:
            - runs for trail_count (50) and uses the same seed for every episode. 
            - resets Q values every trial, but keeps them for episodes
        Randomize = True:
            - Runs 1 trial but should be for lots of episodes
            - New seed every episode for domain randomization
            - Q values are saved for each episode & aren't reset (only 1 trial)
        """
        for t in range(self.trail_count):
            self.reset_env(self.seed)     
            for i in range(self.episode_count):
                if self.randomize:
                    self.set_seed()
                self.episode(t, i)
                print(f"{t} {i} : {self.R[t, i]}")
                self.restore_init_env_state(self.seed)


    def test(self) -> None:
        """
        For part b domain randomization. This tests a new env using the Q values we learned in trial()
        """
        self.test_reward = 0.0
        print(f"{self.seed = }")
        self.epsilon = 0.6 # exploit more for testing? 
        self.restore_init_env_state(self.seed)
        self.episode(-1, -1)


    def visual(self, new_env, e : float = 0.0):
        """
        Uses a new environment to show visually the learned policy
        """
        self.env = new_env
        self.epsilon = e # only exploit for demonstration? 
        self.restore_init_env_state(self.seed)
        self.episode(-1, -1)


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
            if np.all(self.Q[state] == 0.0):
                return 2
            return np.argmax(self.Q[state])
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
        Doesn't reset Q values, doesn't
        """
        obs, _ = self.env.reset(seed=s)
        self.obs = obs
        self.goal = np.where(self.obs['image'][:, :, 2] == 8)
        self.s_0 = self.format_state(self.obs)        
    

    def reset_env(self, s : int) -> None:
        """
        Reset Q values and four rooms environment
        Randomizes acording to seed (s) the agent , goal, and wall locations
        """
        self.Q = np.zeros((19, 19, 4, 3)) #(19, 19, 4, 3) (x, y, direction, action) (x, y, d) <- state
        self.restore_init_env_state(s)
        self.goal = np.where(self.obs['image'][:, :, 2] == 8)
        self.s_0 = self.format_state(self.obs)   
