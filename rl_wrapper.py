
class RLWrapper():
    
    def __init__(self, obs, agent, goal):
        self.obs = obs
        self.agent = agent
        self.goal = goal
        self.epsilon = 0.3 # change to value for e-greedy
        self.alpha = 0.1 # change for learning rate
        self.gamma = 0.99 # change for discount rate
        self.states = [State(c[0], c[1], c[2]) for r in self.obs for c in r] # shape = (361,)


    def next_state(self):
        # s <- s'
        # Do we need a state class?
        pass

    def calc_reward(self) -> float:
        # A reward of ‘1 - 0.9 * (step_count / max_steps)’ is given for success, and ‘0’ for failure.
        pass

    def check_terminal(self) -> bool:
        # return T/F if s ==
        pass


class State():
    def __init__(self, x, y, obj_id):
        self.q = 0 # Q value
        self.e = 0 # eligibility trace
        self.x = x # x coord
        self.y = y # y coord
        self.obj_id = obj_id

    def __str__(self):
        return f"{self.x}, {self.y}, {self.obj_id}"
    
    


