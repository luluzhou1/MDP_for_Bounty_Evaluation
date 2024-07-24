import numpy as np

# Here are all the reward functions we feed to MDP.

time_bonus = 0.1
full_key_bonus = 0.05


def reward_fun_sqrt(self, n, t, index, plus):
    """
    n: number of key shares
    t: turn_in time
    """
    if plus == 0:
        return -10000
    if index == self.terminate or t >= self.T:
        return 0
    if n == 0:
        reward = 0
    elif n == self.N:
        reward = self.M + time_bonus * (1-t/self.T) + full_key_bonus
    else:
        reward = self.M * (np.sqrt(n / self.N) - (t / self.T) * (np.sqrt(n / self.N) - np.sqrt((n - 1) / self.N)))
    return reward


def reward_fun_linear(self, n, t, index, plus):
    """
    n: number of key shares
    t: turn_in time
    """
    if plus == 0:
        return -10000
    if index == self.terminate or t >= self.T:
        return 0
    if n == 0:
        reward = 0
    elif n == self.N:
        reward = self.M + time_bonus * (1-t/self.T) + full_key_bonus
    else:
        reward = self.M * (n / self.N - (t / self.T) * (n / self.N - (n - 1) / self.N))
    return reward


def reward_fun_square(self, n, t, index, plus):
    """
    n: number of key shares
    t: turn_in time
    """
    if plus == 0:
        return -10000
    if index == self.terminate or t >= self.T:
        return 0
    if n == 0:
        reward = 0
    elif n == self.N:
        reward = self.M + time_bonus * (1-t/self.T) + full_key_bonus
    else:
        reward = self.M * (np.square(n / self.N) - (t / self.T) * (np.square(n / self.N) - np.square((n-1) / self.N)))
    return reward


def reward_fun_epsilon(self, n, t, index, plus):
    if plus == 0:
        return -10000
    if index == self.terminate or t >= self.T:
        return 0
    if n == self.N:
        reward = self.M + time_bonus * (1-t/self.T) + full_key_bonus
    else:
        reward = 0
    return reward


def reward_fun_one_share(self, n, t, index, plus):
    if plus == 0:
        return -10000
    if index == self.terminate or t >= self.T:
        return 0
    if n > 0:
        reward = self.M + time_bonus * (1-t/self.T) + full_key_bonus
        # reward = self.M
    else:
        reward = 0
    return reward

def reward_fun_state_of_art(self, n, t, index, plus):
    if plus == 0:
        return -10000
    if index == self.terminate or t >= self.T:
        return 0
    if n > 0:
        # epsilon = 0.435
        delta = 1
        eta = 0.01
        g_k_1 = (self.M ** self.epsilon) * ((n * self.M / self.N) ** (1 - self.epsilon))
        g_k_2 = (n * self.M / self.N)
        g_k = g_k_1 - g_k_2
        time_reward = time_bonus * (1-t/self.T)
        reward = g_k_1 - ((t / self.T) ** delta) * g_k + eta + time_reward
        ## upper bound of the reward
        if reward > self.cap * self.M:
            reward = self.cap * self.M
    else: 
        reward = 0
    return reward

def reward_fun_reward_acc_value(self, n, t, index, plus):
    if plus == 0:
        return -10000
    if index == self.terminate or t >= self.T: 
        return 0
    if n > 0:
        eta = 0.01
        time_reward = time_bonus * (1-t/self.T)
        reward = (n * self.M / self.N) + eta + time_reward
        # reward = (n * self.M / self.N) + eta
        ## upper bound of the reward
        if reward > self.cap * self.M:
            reward = self.cap * self.M
    else:
        reward = 0
    return reward

def reward_fun_no_bounty(self, n, t, index, plus):
    if plus == 0:
        return -10000
    else:
        return 0
    
def reward_fun_all_shares_same_value(self, n, t, index, plus):
    if plus == 0:
        return -10000
    if index == self.terminate or t >= self.T: 
        return 0
    if n > 0:
        time_reward = 1 * (1-t/self.T)
        reward = 2*(self.M / self.N) + time_reward
        ## upper bound of the reward
        if reward > self.cap * self.M:
            reward = self.cap * self.M
    else:
        reward = 0
    return reward
