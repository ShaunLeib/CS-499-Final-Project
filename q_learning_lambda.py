import numpy as np
from rl_wrapper import RLWrapper

class QLearningLambda(RLWrapper):
    def __init__(self, env, trail_count : int, episode_count : int, randomize : bool):
        super().__init__(env, trail_count, episode_count, randomize)
        self.alpha = 0.001 # 0.01 for part a
        self.lamb = 0.5
        self.epsilon = 0.7 # 0.6 for part a with no decay
        self.e = np.zeros((19, 19, 4, 3)) # Trace variable (19, 19, 4,  3) (x, y, direction, action)
        self.decay = 0.9995
        

    def episode(self, t: int, i : int) -> None:
        s = self.s_0
        self.e = np.zeros((19, 19, 4, 3))
        a = self.best_action(s, 0.0) 
        terminated = False # Reach goal
        truncated = False # max_steps reached
        while terminated == False and truncated == False:
            obs, r, terminated, truncated, _ = self.env.step(a)
            self.env.render()

            s_prime = self.format_state(obs) #s'
            a_prime = self.best_action(s_prime, self.epsilon) #a'       
            a_star = self.best_action(s_prime, 0.0) #a*

            #time penalty only for part b
            r -= 0.002

            # Calculate TD error
            err = r + self.gamma*self.Q[s_prime[0], s_prime[1], s_prime[2], a_star] - self.Q[s[0], s[1], s[2], a]

            # Update eligibility trace for visiting this state
            self.e[s[0], s[1], s[2], a] += 1 

            # Update Q-values for all state action pairs 
            x, y, d, actions = self.state_action_pairs[:, 0], self.state_action_pairs[:, 1], self.state_action_pairs[:, 2], self.state_action_pairs[:, 3]
            self.Q[x, y, d, actions] += self.alpha * err *  self.e[x, y, d, actions]

            if a_prime == a_star:
                # Update/Decay all eligibility trace values 
                self.e[x, y, d, actions] *= self.gamma * self.lamb
            else:
                self.e[x, y, d, actions] = 0

            # set s <- s' and a <- a'
            s = s_prime
            a = a_prime

        self.epsilon = max(0.1, self.epsilon * self.decay)       

        if t != -1 and i != -1:
            self.R[t, i] = r
        else:
            self.test_reward = r
            print(f"REWARD {r}")

    def _print_q_values(self) -> None:
        non_zero_q = np.where(self.Q != 0.0)
        print(f"{len(non_zero_q[0])}")
        for x, y, d, a in zip(*non_zero_q):
            print(f"({x} {y} {d} {a}) : {self.Q[x, y, d, a]}")


    def reset_env(self, s: int) -> None:
        self.e = np.zeros((19, 19, 4, 3)) #(19, 19, 3) (x, y, direction, action)
        super().reset_env(s)

    

