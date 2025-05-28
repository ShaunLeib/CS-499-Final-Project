import numpy as np
from rl_wrapper import RLWrapper

class SARSA(RLWrapper):
    def __init__(self, env, trail_count : int, episode_count : int, randomize : bool):
        super().__init__(env, trail_count, episode_count, randomize)
        self.alpha = 0.05
        self.epsilon = 0.6
    
    def episode(self, t: int, i : int) -> None:
        s = self.s_0 # init S
        a = self.best_action(s, self.epsilon) # choose A from S using epsilon-greedy policy

        terminated = False
        truncated = False

        while not terminated and not truncated: # loop until S terminates
            obs, r, terminated, truncated, _ = self.env.step(a) # take action A, oberse R, S'
            self.env.render() # show visuals
            s_prime = self.format_state(obs)                    # extract S'
            a_prime = self.best_action(s_prime, self.epsilon)   # choose A' from S' using epsilon-greedy
            
            ### SARSA TD UPDATE ###
            # Q(S,A) ← Q(S,A) + alpha [R + gamma Q(S', A') − Q(S,A)]
            q_curr = self.Q[s[0], s[1], s[2], a]                            # Q(S,A)
            q_next = self.Q[s_prime[0], s_prime[1], s_prime[2], a_prime]    # Q(S', A')
            td_err = (r + self.gamma * q_next) - q_curr                     # [R + gamma Q(S', A') − Q(S,A)]
            self.Q[s[0], s[1], s[2], a] += self.alpha * td_err

            # # TD Update: Q(S,A) ← Q(S,A) + α [R + γ Q(S′, A′) − Q(S,A)]
            # self.Q[s[0], s[1], s[2], a] += self.alpha * (
            #     r + self.gamma * self.Q[s_prime[0], s_prime[1], s_prime[2], a_prime]
            #     - self.Q[s[0], s[1], s[2], a]
            # )



            s = s_prime     # S <- S'
            a = a_prime     # A <- A'

        if t != -1 and i != -1:
            self.R[t, i] = r
        else:
            self.test_reward = r
            print(f"REWARD {r}")