import numpy as np
from rl_wrapper import RLWrapper

class QLearningLambda(RLWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.epsilon = 0.5
        self.gamma = 0.95
        self.alpha = 0.01
        self.lamb = 0.9
        self.e = np.zeros((19, 19, 4, 3)) # Trace variable (19, 19, 4,  3) (x, y, direction, action)
        

    def episode(self):
        s = self.s_0
        a = self.best_action(s, 0.0) # CHANGE: should be argmax(of q values of adj states)
        terminated = False # Reach goal
        truncated = False # max_steps reached
        while terminated == False and truncated == False:
            obs, r, terminated, truncated, _ = self.env.step(a)
            s_prime = self.format_state(obs)
            a_prime = self.best_action(s_prime, self.epsilon) # CHANGE: should be argmax(of q values of adj states) w/ (e-greedy)            
            self.env.render()

            # Calculate TD error
            err = r + self.gamma*self.Q[s_prime[0], s_prime[1], s_prime[2], a_prime] - self.Q[s[0], s[1], s[2], a]
            print(f"{err =}")

            # Update eligibility trace for visiting this state
            self.e[s[0], s[1], s[2], a] += 1 

            # Update Q-values for all state action pairs 
            x, y, d, actions = self.state_action_pairs[:, 0], self.state_action_pairs[:, 1], self.state_action_pairs[:, 2], self.state_action_pairs[:, 3]
            self.Q[x, y, d, actions] += self.alpha * err *  self.e[x, y, d, actions]

            # Update/Decay all eligibility trace values 
            self.e[x, y, d, actions] *= self.gamma * self.lamb

            # set s <- s' and a <- a'
            s = s_prime
            a = a_prime

        print(f"REWARD {r}")
        self._print_q_values()
        print(f"{terminated = } {truncated = }")

    def _print_q_values(self):
        non_zero_q = np.where(self.Q != 0.0)
        print(f"{len(non_zero_q[0])}")
        for x, y, d, a in zip(*non_zero_q):
            print(f"({x} {y} {d} {a}) : {self.Q[x, y, d, a]}")


    def reset_env(self, s: int) -> None:
        self.e = np.zeros((19, 19, 4, 3)) #(19, 19, 3) (x, y, direction, action)
        super().reset_env(s)

    

