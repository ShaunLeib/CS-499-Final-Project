import numpy as np
import random
from rl_wrapper import RLWrapper

class QLearningLambda(RLWrapper):
    def __init__(self, env, obs):
        super().__init__(env, obs)
        self.epsilon = 1.0
        self.gamma = 0.95
        self.alpha = 0.01
        self.e = np.zeros((19, 19, 4, 3)) # Trace variable (19, 19, 4,  3) (x, y, direction, action)
        

    def episode(self):
        s = self.s_0
        print("s0:", s)
        a = self.best_action(s, 0.0) # CHANGE: should be argmax(of q values of adj states)
        print("a_0:", a)
        terminated = False
        while terminated == False:
            obs, r, terminated, truncated, test = self.env.step(a)
            s_prime = self.format_state(obs)
            a_prime = self.best_action(s_prime, self.epsilon) # CHANGE: should be argmax(of q values of adj states) w/ (e-greedy)            
            print(f"{a_prime =}")
            self.env.render()
            err = r + self.gamma*self.Q[s_prime[0], s_prime[1], s_prime[2], a_prime] - self.Q[s[0], s[1], s[2], a]
            print(f"{err =}")

            # print(self.Q)
            x, y, d, actions = self.state_action_pairs[:, 0], self.state_action_pairs[:, 1], self.state_action_pairs[:, 2], self.state_action_pairs[:, 3]
            self.Q[x, y, d, actions] += self.alpha * err *  self.e[x, y, d, actions]
            print(self.Q)
            s = s_prime
            a = a_prime
        print(f"{r =}") 
        print(terminated)
        print(truncated)
        print(self.Q)
            # break

    def reset(self) -> None:
        self.e = np.zeros((19, 19, 4, 3)) #(19, 19, 3) (x, y, direction, action)
        super().reset()

    

