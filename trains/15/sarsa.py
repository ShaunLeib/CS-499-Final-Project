import numpy as np
from rl_wrapper import RLWrapper

class SARSA(RLWrapper):
    def __init__(self, env, trail_count : int, episode_count : int, randomize : bool):
        super().__init__(env, trail_count, episode_count, randomize)
        self.alpha = 0.1
        self.epsilon = 0.6
        self.decay = 1#0.9984
    
    def episode(self, t: int, i : int) -> None:
        s = self.s_0 # init S
        a = self.best_action(s, self.epsilon) # choose A from S using epsilon-greedy policy
        self.env.render() # show visuals
        total_reward = 0

        terminated = False
        truncated = False

        while not terminated and not truncated: # loop until S terminates
            obs, r, terminated, truncated, _ = self.env.step(a) # take action A, oberse R, S'
            s_prime = self.format_state(obs)                    # extract S'
            a_prime = self.best_action(s_prime, self.epsilon)   # choose A' from S' using epsilon-greedy

            # # Penalize actual forward wall bumps
            # if a == 0 and self.is_facing_wall(obs["image"], s[2], s[0], s[1]):
            #     r -=0.05

            # Penalize not moving
            if s[:2] == s_prime[:2]:
                r -= 0.02  # Stronger penalty for staying in the same place and discourage spin+wait loops

            # Encourage forward motion (position change)
            if a == 0 and s_prime[:2] != s[:2]:
                r += 0.05

            # Mild penalty for any turn (direction change)
            if s[2] != s_prime[2] and a in [1, 2]:
                r -= 0.03 # More than the standing still penalty

            # Small time penalty for each step
            r -= 0.005

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


            # print(f"Seed: {self.seed}, State: {s}, Action: {a}, Reward: {r:.3f}")
            s = s_prime     # S <- S'
            a = a_prime     # A <- A'
            total_reward += r
        # print(f"Terminated: {terminated}, Truncated: {truncated}, Step count: {self.env.unwrapped.step_count}")

        self.epsilon = max(0.1, self.epsilon * self.decay)
        # print(f"  ε:{round(self.epsilon,3)}  | ", end="")

        if t != -1 and i != -1:
            self.R[t, i] = r
            if r > 0: # episode ended by reaching the goal
                self.success_count += 1 # debug
                self.Q[s[0], s[1], s[2], a] += self.alpha * 2.0 # incentive to read the goal state
                steps_used = self.env.unwrapped.step_count
                time_bonus = max(0, 1.0 - (steps_used / 3000))
                time_bonus += 1.0  # Main reward?
                self.Q[s[0], s[1], s[2], a] += self.alpha * time_bonus
                # print(f" Total Reward: {total_reward} | test REWARD {r} | STEPS TO GOAL: {self.env.unwrapped.step_count}")
        else:
            self.test_reward = r
            print(f" Total Reward: {total_reward} | test REWARD {r} | STEPS TO GOAL: {self.env.unwrapped.step_count}")
