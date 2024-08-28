import numpy as np

def reward_fun_state_of_art(self, n, t, index, plus):
    # the reward function we proposed
    if plus == 0:
        return -10000
    if index == self.terminate or t >= self.T:
        return 0
    if n > 0:
        eta = 0.01
        g_k_1 = (self.M ** self.epsilon) * ((n * self.M / self.N) ** (1 - self.epsilon))
        g_k_2 = (n * self.M / self.N)
        g_k = g_k_1 - g_k_2
        reward = g_k_1 - (t / self.T) * g_k + eta
        ## upper bound of the reward
        if reward > self.cap * self.M:
            reward = self.cap * self.M
    else: 
        reward = 0
    return reward

def reward_fun_reward_acc_value(self, n, t, index, plus):
    # linear reward function
    if plus == 0:
        return -10000
    if index == self.terminate or t >= self.T: 
        return 0
    if n > 0:
        eta = 0.01
        time_bonus = 0.1
        time_reward = time_bonus * (1-t/self.T)
        reward = (n * self.M / self.N) + eta + time_reward
        ## upper bound of the reward
        if reward > self.cap * self.M:
            reward = self.cap * self.M
    else:
        reward = 0
    return reward

def reward_fun_no_bounty(self, n, t, index, plus):
    # reward function without bounty
    if plus == 0:
        return -10000
    else:
        return 0
