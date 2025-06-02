import numpy as np
from rl_wrapper import RLWrapper
from collections import deque

class SARSA(RLWrapper):
    def __init__(self, env, trail_count : int, episode_count : int, randomize : bool):
        super().__init__(env, trail_count, episode_count, randomize)
        self.alpha = 0.0008 # < 0.004, 0.0004-0.0006
        self.epsilon = 0.9
        self.decay = 1 # 1=no decay, 0.99992 gets 0.9 -> 0.65 in 4000 episodes
    
    def episode(self, t: int, i : int) -> None:
        s = self.s_0 # init S
        a = self.best_action(s, self.epsilon) # choose A from S using epsilon-greedy policy
        total_reward = 0
        steps_taken = 0
        consecutive_turns = 0
        recent_positions = deque(maxlen=25)  # remember last 25 tiles
        self.has_seen_goal = 0.8

        terminated = False
        truncated = False

        while not terminated and not truncated: # loop until S terminates
            obs, r, terminated, truncated, _ = self.env.step(a) # take action A, oberse R, S'
            steps_taken += 1
            # self.env.render() # show visuals

            # # === Begin shaped reward logic ===
            # goal_visible = np.any(obs['image'][:, :, 2] == 8)   # Look for goal in visible observation

            # if terminated and r == 1.0:
            #     r = 2.0  # success
            # elif goal_visible:
            # # elif goal_visible:
            #     r = min(1, 1 * (1 - self.has_seen_goal))  # bonus for seeing the goal that drops off
            #     self.has_seen_goal *= 1.025 # about 8 times
            # else:
            #     r = -0.01  # time penalty
            # # === End shaped reward logic ===

            s_prime = self.format_state(obs)                    # extract S'
            a_prime = self.best_action(s_prime, self.epsilon)   # choose A' from S' using epsilon-greedy


            # # Penalize actual forward wall bumps
            # if a == 0 and self.is_facing_wall(obs["image"], s[2], s[0], s[1]):
            #     r -=0.05

            # if a == 0 and s[:2] != s_prime[:2]: # reward any forward motion
            #     r += 0.02   # heavier spin/stall penalty

            # if s[:2] == s_prime[:2]: # penalize not moving spaces
            #     r -= 0.01

            # # Check if this action is a turn
            # if a in [1, 2]:  # turn left or right
            #     consecutive_turns += 1
            # else:
            #     consecutive_turns = 0
            # if consecutive_turns >= 2:    # penalty for excessive spinning
            #     r -= 0.01 * consecutive_turns  # Penalize 2nd+ consecutive turn

            # IDEA GRAVEYARD # discourage revisits of recent (last 25) spaces # DOES NOT WORK AS IS (drops 55% to 7-14%)
            # pos = s_prime[:2]
            # if pos in recent_positions and r <= 0:
            #     r -= 0.01  # discourage wasteful revisits #0.01 better than 0.02 by 3% ish
            # recent_positions.append(pos)

            # # encourage new spaces (not in last 25)
            # pos = s_prime[:2]
            # if pos not in recent_positions:
            #     r += 0.005  # optional: only if r <= 0 to avoid double-counting
            # recent_positions.append(pos)

            # # Small time penalty for each step
            # r -= 0.0005

            ### SARSA TD UPDATE ###
            # Q(S,A) ← Q(S,A) + alpha [R + gamma Q(S', A') − Q(S,A)]
            q_curr = self.Q[s[0], s[1], s[2], a]                            # Q(S,A)
            q_next = self.Q[s_prime[0], s_prime[1], s_prime[2], a_prime]    # Q(S', A')
            td_err = (r + self.gamma * q_next - q_curr)                     # [R + gamma Q(S', A') − Q(S,A)]
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
            max_steps = 2000 #TEMP
            if terminated and steps_taken < max_steps: # episode ended by reaching the goal
                self.success_count += 1 # debug
                # r += 1.0  # modest bump to reward itself, not Q directly
                # r = 1.0 + max(0, 1.0 - (self.env.unwrapped.step_count / 2000))

                # self.Q[s[0], s[1], s[2], a] += self.alpha * 2.0 # incentive to reach the goal state
                # steps_used = self.env.unwrapped.step_count
                # time_bonus = max(0, 1.0 - (steps_used / 3000))
                # time_bonus += 1.0  # Extra main reward?
                # self.Q[s[0], s[1], s[2], a] += self.alpha * time_bonus
                # print(f" Total Reward: {total_reward} | test REWARD {r} | STEPS TO GOAL: {self.env.unwrapped.step_count}")
        else:
            self.test_reward = r
            print(f" Total Reward: {total_reward} | test REWARD {r} | STEPS TO GOAL: {self.env.unwrapped.step_count}")