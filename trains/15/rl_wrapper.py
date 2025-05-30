import numpy as np
import random
from abc import ABC, abstractmethod
from tqdm import tqdm # training progress bar
from minigrid.core.world_object import Goal # for randomize_agent_and_goal()

# REMOVE ME FOR GITHUB (I'll no doubt forget)
from ping_me import ping_dc, ping_sms # notify me on discord when training is complete

class RLWrapper(ABC):
    def __init__(self, env, trail_count : int, episode_count : int, randomize : bool):
        self.env = env
        self.state_action_pairs = self.generate_sap_indices()
        self.episode_count = episode_count
        self.randomize = randomize
        self.trail_count = trail_count
        if self.randomize:
            self.seeds = np.random.randint(0, 1000, size=trail_count) # 1024, 8888
            # self.seeds = [14, 24, 48]
        else:
            self.seeds = [14] * trail_count
        self.Q = np.zeros((19, 19, 4, 3)) #(19, 19, 4, 3) (x, y, direction, action) (x, y, d) <- state
        # self.Q = np.ones((19, 19, 4, 3)) * 0.5
        self.R = np.zeros((self.trail_count, self.episode_count)) # save the rewards for each episode for each trial for learning curve graph

        self.gamma = 0.95
        self.set_seed()
        # You need to set your specific hyperparams in child class

        self.success_count = 0 # debug  # total successes in the current trial


    @abstractmethod
    def episode(self, t: int, i : int) -> None:
        """
        Implement in each class
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


    def randomize_agent_and_goal(self):
        """
        Randomizes agent and goal positions **without modifying format_state behavior**.
        Only uses env.unwrapped to modify grid, then returns a valid symbolic observation.
        """
        if not self.randomize:
            return self.obs, {}

        env = self.env.unwrapped  # temporarily peek for setup

        # Place agent (internally sets agent_pos and agent_dir)
        env.place_agent()
        # env.agent_pos = (17, 17)    # OPTIONAL, MANUAL SET
        # env.agent_dir = 3           # OPTIONAL, MANUAL SET   # 0: right, 1: down, 2: left, 3: up

        # Remove previous goal, if one exists
        if hasattr(env, 'goal_pos') and env.goal_pos is not None:
            env.grid.set(*env.goal_pos, None)

        # Place a new goal object
        goal = Goal()
        env.goal_pos = env.place_obj(goal, max_tries=100) # randomly place
        # env.grid.set(2, 2, goal)    # OPTIONAL, MANUAL SET
        # env.goal_pos = (2, 2)       # OPTIONAL, MANUAL SET

        # Get symbolic obs again AFTER modifying the internal env
        obs, _ = self.env.reset()  # regenerate using symbolic wrapper pipeline
        self.obs = obs
        return obs, {}


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
        total_successes = 0
        total_episodes = self.trail_count * self.episode_count

        trial_iter = tqdm(range(self.trail_count), desc=f"Trials")
        for t in trial_iter:
            self.success_count = 0 # debug
            self.set_seed()
            self.reset_env(self.seed)     
            
            episode_iter = tqdm(range(self.episode_count), desc=f"Trial {t+1}/{self.trail_count}", leave=False) # training progress bar
            for i in episode_iter:
                # if self.randomize:
                if self.randomize and i % 80 == 0: # only change seed every 30 episodes
                    self.set_seed()
                    # self.set_seed(1499) # 14 move 2 quadrants, 137 move 1 quadrant, 1018 same quadrant, 4209 middle corners of spawn removed
                self.episode(t, i)
                # print(f"{t} {i} : {self.R[t, i]}")
                # print(f"Trial {t} | Episode {i} | Reward: {self.R[t, i]:.4f} | Success: {'Y' if self.R[t, i] > 0 else 'N'}")
                episode_iter.set_postfix(reward=self.R[t, i]) # training progress bar
                self.restore_init_env_state(self.seed)

            print(f"Trial {t} success rate: {self.success_count}/{self.episode_count}") # debug
            total_successes += self.success_count
        success_rate = total_successes / total_episodes
        print(f"Overall success rate: {total_successes}/{total_episodes} = {success_rate:.2%}")
        # ping_dc(f"✅ Training complete (SR:{total_successes}/{total_episodes})") # notify me on discord when training is complete
        # ping_sms("SARSA", " ✅ Training complete") # text me when training is complete

    def test(self, e : float = 0.0) -> None:
        """
        For part b domain randomization. This tests a new env using the Q values we learned in trial()
        """
        self.test_reward = 0.0
        self.epsilon = e # exploit more for testing?
        test_seed = 14 # 14 move 2 quadrants, 137 move 1 quadrant, 1018 same quadrant, 4209 middle corners of spawn removed
        self.seed = test_seed
        print(f"{self.seed = }")
        self.restore_init_env_state(self.seed)
        self.episode(-1, -1)
        # ping_dc(f"Reward: {self.test_reward} (Tested on seed {test_seed})") # notify me on discord when training is complete

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
        

    def best_action(self, state: tuple[int], epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return random.choice([0, 1, 2])  # explore
        q_vals = self.Q[state]
        if np.all(q_vals == 0.0):
            return random.choice([0, 1, 2])  # break tie with random, not hardcoded
        return int(np.argmax(q_vals))  # exploit


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
        self.obs, _ = self.env.reset(seed=s)
        if self.randomize:
            self.obs, _ = self.randomize_agent_and_goal()
        self.goal = np.where(self.obs['image'][:, :, 2] == 8)
        self.s_0 = self.format_state(self.obs)        
    

    def reset_env(self, s : int) -> None:
        """
        Reset Q values and four rooms environment
        Randomizes acording to seed (s) the agent , goal, and wall locations
        """
        self.Q = np.zeros((19, 19, 4, 3)) #(19, 19, 4, 3) (x, y, direction, action) (x, y, d) <- state
        # self.Q = np.ones((19, 19, 4, 3)) * 0.5
        self.restore_init_env_state(s)


    def is_facing_wall(self, obs: np.ndarray, direction: int, x: int, y: int) -> bool:
        """
        Returns True if the tile in front of the agent is a wall.
        Uses global (x, y) position and cardinal direction.
        """
        dx, dy = {
            0: (1, 0),   # right
            1: (0, 1),   # down
            2: (-1, 0),  # left
            3: (0, -1),  # up
        }[direction]

        fx, fy = x + dx, y + dy

        if 0 <= fx < 19 and 0 <= fy < 19:
            return obs[fx, fy, 0] == 2  # wall object
        return False
