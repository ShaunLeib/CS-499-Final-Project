import numpy as np
from rl_wrapper import RLWrapper
from collections import deque

class SARSA(RLWrapper):
    def __init__(self, env, trail_count : int, episode_count : int, randomize : bool):
        super().__init__(env, trail_count, episode_count, randomize)
        # # part a
        # self.alpha = 0.07 # < 0.004, 0.0004-0.0006
        # self.epsilon = 0.6
        # self.decay = 1 # 1=no decay, 0.99992 gets 0.9 -> 0.65 in 4000 episodes

        # part b
        self.alpha = 0.0012 # < 0.004, 0.0004-0.0006
        self.epsilon = 0.9
        self.decay = 0.99927 # 1=no decay, 0.99927 gets 0.9 -> 0.1 in 3000 episodes

    def episode(self, t: int, i : int) -> None:
        s = self.s_0 # init S
        a = self.best_action(s, self.epsilon) # choose A from S using epsilon-greedy policy
        total_reward = 0
        steps_taken = 0

        terminated = False
        truncated = False

        while not terminated and not truncated: # loop until S terminates
            obs, r, terminated, truncated, _ = self.env.step(a) # take action A, oberse R, S'
            steps_taken += 1
            # self.env.render() # show visuals
            s_prime = self.format_state(obs)                    # extract S'
            a_prime = self.best_action(s_prime, self.epsilon)   # choose A' from S' using epsilon-greedy

            # Small time penalty for each step
            # r -= 0.03

            ### SARSA TD UPDATE ###
            # Q(S,A) ← Q(S,A) + alpha [R + gamma Q(S', A') − Q(S,A)]
            q_curr = self.Q[s[0], s[1], s[2], a]                            # Q(S,A)
            q_next = self.Q[s_prime[0], s_prime[1], s_prime[2], a_prime]    # Q(S', A')
            td_err = (r + self.gamma * q_next - q_curr)                     # [R + gamma Q(S', A') − Q(S,A)]
            self.Q[s[0], s[1], s[2], a] += self.alpha * td_err

            # print(f"Seed: {self.seed}, State: {s}, Action: {a}, Reward: {r:.3f}")
            s = s_prime     # S <- S'
            a = a_prime     # A <- A'
            total_reward += r
        # print(f"Terminated: {terminated}, Truncated: {truncated}, Step count: {self.env.unwrapped.step_count}")

        self.epsilon = max(0.1, self.epsilon * self.decay)
        # print(f"  ε:{round(self.epsilon,3)}  | ", end="")



        if t != -1 and i != -1:
            self.R[t, i] = total_reward
        else:
            self.test_reward = r
            print(f" Total Reward: {total_reward} | test REWARD {r} | STEPS TO GOAL: {self.env.unwrapped.step_count}")