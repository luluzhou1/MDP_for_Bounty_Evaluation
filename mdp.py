#########################
# Tips for MDP:
# When an action is not allowed, set the reward to -inf
#########################
# remaining problem:
# After restricting the attacking numbers of enclaves, we do not need to modify the distribution
#########################
# The attacker decide the number of enclaves to attack at step t, 
# and decide whether to turn in / exploit / keep attacking in the next step at t+.

import mdptoolbox
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
import matplotlib


class MDP:
    def __init__(self, N, T, M, p, c, epsilon, cap, reward_fun, reward_fun_name, allow_sell_share):
        self.N = N  # Total number of enclaves (also key shares), i = 0, 1, ..., N
        self.T = T  # Total number of time steps, t = 0, 1, ..., T-1
        self.M = M  # money in wallet
        self.p = p  # prob of breaking one enclave at one step when attacking
        self.c = c  # c = M/N * p
        self.epsilon = epsilon # param in reward function
        self.cap = cap # param in reward function (maximum amount of money paid in bounty, in terms of percentage of secret value)
        self.terminate = T * 2 * (N + 1)  # index of state terminate
        self.total_num_of_states = self.terminate + 1
        self.total_num_of_actions = N+4  # attack 0, ..., N enclaves; keep attacking, turn in, move
        self.reward_fun = reward_fun
        self.reward_fun_name = reward_fun_name  # name for printing in plot
        self.allow_sell_share = allow_sell_share  # whether allow attacker to sell key shares or only allow full key to be exploited
        self.model = None  # will be assigned after calling run_mdp
        self.P = None   # the state transit matrix
        self.R = None   # the reward matrix

    def move_fund(self, n, index, plus):
        """
        assigning reward for moving funds
        """
        if plus == 0:
            return -10000  # cannot move funds
        if index == self.terminate:
            return 0
        
        if self.allow_sell_share:
            # assuming that attacker could sell key shares
            reward = n * self.M / self.N
        else:
            # assuming that only full key could be exploited
            if n == self.N:
                reward = self.M
            else:
                reward = 0

        return reward

    def index_to_state(self, index):
        """
        index: int
        turn index into meaningful states (time, plus, num_key_shares)
        """
        # (time, plus, #_key_shares)
        # time = 0, 1, 2, ..., T-1
        # plus = 0, 1
        # #_of_key_shares = 0, 1, 2, ..., N
        time = index // (2*(self.N+1))
        plus = index % (2*(self.N+1)) // (self.N+1)
        num_key_shares = index % (2*(self.N+1)) % (self.N+1)
        return [time, plus, num_key_shares]

    def state_to_index(self, state):
        """
        turn states into index
        """
        # (time, plus, #_key_shares)
        time, plus, num_key_shares = state
        index = time*(2*(self.N+1)) + plus*(self.N+1) + num_key_shares
        return index

    def get_binomial_distribution(self, a):
        """
        get a binomial distribution of success cases in a attempts
        """
        # number of broken enclaves: 0, 1, ..., a
        distribution = []
        for b in range(a+1):
            prob = np.math.factorial(a)/(np.math.factorial(b)*np.math.factorial(a-b))*np.power(self.p, b)*np.power(1 - self.p, a-b)
            distribution.append(prob)
        return distribution

    # construct the transition matrix P
    def construct_P(self):
        P = []
        # action 0, ..., N: attack * enclaves
        for action in range(0, self.N+1):
            transit_matrix = np.zeros([self.total_num_of_states, self.total_num_of_states])
            for i in range(self.terminate):
                state = self.index_to_state(i)  # get the state
                time, plus, num_of_key_shares = state  # assign attribute value
                if plus == 1:
                    # at t+, the attacker are not deciding which enclave to attack. Let them stay at cur state.
                    transit_matrix[i, i] = 1
                else:
                    # next state is t+
                    next_time = time
                    next_plus = 1
                    # distribution of increased number of key shares
                    next_num_of_increased_shares_distribution = np.zeros([self.N-num_of_key_shares+1])  # increase 0, 1, ..., N-num_shares
                    if self.N-num_of_key_shares+1 >= action+1:
                        # increased number of key shares is a binomial distribution
                        distribution = self.get_binomial_distribution(action)
                        next_num_of_increased_shares_distribution[0 : len(distribution)] = distribution  
                    else:
                        next_num_of_increased_shares_distribution = self.get_binomial_distribution(self.N-num_of_key_shares+1)

                    increased_shares = 0
                    for prob in next_num_of_increased_shares_distribution:
                        next_num_of_shares = num_of_key_shares + increased_shares
                        next_state_index = self.state_to_index([next_time, next_plus, next_num_of_shares])
                        transit_matrix[i, next_state_index] = prob
                        increased_shares += 1

            transit_matrix[self.terminate, self.terminate] = 1  # terminate->terminate
            P.append(cp.deepcopy(transit_matrix))

        # action N+1: keep attacking in next round
        transit_matrix = np.zeros([self.total_num_of_states, self.total_num_of_states])
        for i in range(self.terminate):
            state = self.index_to_state(i)  # get the state
            time, plus, num_of_key_shares = state  # assign attribute value
            if plus == 0:
                # at t, we let them stay at cur state
                transit_matrix[i, i] = 1
            elif time == self.T-1:  # wait at (T-1)+ leads to terminate
                transit_matrix[i, self.terminate] = 1
            else:
                next_index = self.state_to_index([time+1, 0, num_of_key_shares])
                transit_matrix[i, next_index] = 1
        transit_matrix[self.terminate, self.terminate] = 1 # terminate->terminate
        P.append(cp.deepcopy(transit_matrix))

        # action N+2: move funds
        transit_matrix = np.zeros([self.total_num_of_states, self.total_num_of_states])
        for i in range(self.total_num_of_states):
            transit_matrix[i, self.terminate] = 1
        P.append(cp.deepcopy(transit_matrix))

        # action N+3: turn in keys
        P.append(cp.deepcopy(transit_matrix))  # terminate with prob=1 like move funds.

        # set the transit matrix 
        self.P = P
        return P


    # construct the reward matrix R
    def construct_R(self):
        R = []

        ## action 0, ..., N: attack a enclaves
        # give -a*c reward to every state transit, only based on actions
        for action in range(self.N+1):
            reward_matrix = np.zeros([self.total_num_of_states, self.total_num_of_states])
            for index in range(self.terminate):  # last row for terminate is left zeros
                state = self.index_to_state(index)  # get the state
                time, plus, num_of_key_shares = state  # assign attribute value
                if plus == 0:
                    if self.N-num_of_key_shares < action:
                        reward = -10000  # we do not allow attacker to attack # of more than unbroken enclaves
                    else:
                        reward = -action*self.c   # number of attacked enclaves * unit_cost
                else:
                    reward = -10000   # at t+, attacker is not allowed to attack
                reward_matrix[index, :] = np.full([1, self.total_num_of_states], reward)
            R.append(reward_matrix)

        ## action N+1: wait
        # give 0 to every state transit
        reward_matrix = np.zeros([self.total_num_of_states, self.total_num_of_states])
        for index in range(self.terminate):   # last row for terminate is left zeros
            state = self.index_to_state(index)  # get the state
            time, plus, num_of_key_shares = state  # assign attribute value
            if plus == 0:
                reward = -10000  # at t, attacker is not allowed to wait
            else:
                reward = 0
            reward_matrix[index, :] = np.full([1, self.total_num_of_states], reward)
        R.append(reward_matrix)

        ## action N+2: move funds
        # give reward_fun(num_key_shares, time) to every state transit
        reward_matrix = np.zeros([self.total_num_of_states, self.total_num_of_states])
        for index in range(self.terminate):
            state = self.index_to_state(index)  # get the state
            time, plus, num_of_key_shares = state  # assign attribute value
            reward = self.move_fund(num_of_key_shares, index, plus)
            reward_matrix[index, :] = np.full([1, self.total_num_of_states], reward)
        R.append(reward_matrix)

        ## action N+3: turn in keys
        # give move_fund(num_key_shares, time) to every state transit
        reward_matrix = np.zeros([self.total_num_of_states, self.total_num_of_states])
        for index in range(self.terminate):
            state = self.index_to_state(index)  # get the state
            time, plus, num_of_key_shares = state  # assign attribute value
            reward = self.reward_fun(self, num_of_key_shares, time, index, plus)
            reward_matrix[index, :] = np.full([1, self.total_num_of_states], reward)
        R.append(reward_matrix)

        self.R = R
        return R

    def run_mdp(self, discount=1):
        P = self.construct_P()
        P = np.array(P)
        R = self.construct_R()
        R = np.array(R)
        mdp_one_wallet = mdptoolbox.mdp.ValueIteration(P, R, discount)
        mdp_one_wallet.run()
        self.model = mdp_one_wallet

    def __get_action(self, a):
        if a <= self.N:
            action = f"Attack {a} enclaves"
        elif a == self.N + 1:
            action = 'wait'
        elif a == self.N + 2:
            action = 'exploit key share(s)'
        else:
            action = 'turn in key share(s)'
        return action

    def __get_color(self, a):
        if a <= self.N:
            color = matplotlib.colors.to_hex([1, 0, 0, a/self.N], keep_alpha=True)
        elif a == self.N + 1:
            color = '#FFFF00'
        elif a == self.N + 2:
            color = matplotlib.colors.to_hex([1, 0, 0, 1], keep_alpha=True)
        else:
            color = '#86DC3D'
        return color

    def __get_plus(self, plus):
        """
        :return:
        """
        if plus == 0:
            return ''
        elif plus == 1:
            return '+'

    def get_action_matrix(self):
        action_list = []
        for state_index in range(self.total_num_of_states):
            state = self.index_to_state(state_index)  # get the state
            time, plus, num_of_key_shares = state  # assign attribute value
            action = self.__get_action(self.model.policy[state_index])
            # reward_increase = self.reward_fun(self, num_of_key_shares+1, time+1, state_index, 1) - \
            #                   self.reward_fun(self, num_of_key_shares, time, state_index, 1)
            # action = action + f'reward increase = {reward_increase}'
            action_list.append(action)
            # print(f'time:{time} plus: {plus} num_key_shares: {num_of_key_shares} action:{action}')
        action_list = action_list[0:-1]
        action_matrix = np.reshape(action_list, [2 * self.T, self.N + 1])
        return action_matrix
    
    def get_state_probs(self):
        policy_list = self.model.policy
        state_prob_list = np.zeros(self.total_num_of_states)
        state_prob_list[0] = 1  # start from state (t=0, n=0)
        for i in range(self.total_num_of_states):
            action = policy_list[i]  # get optimal policy
            next_prob_distribution = self.P[action][i]
            for j in range(len(next_prob_distribution)):
                state_prob_list[j] += state_prob_list[i] * next_prob_distribution[j]
        return state_prob_list


    def get_state_prob_matrix(self):
        """
        compute the prob of reaching each state under optimal policy
        :return: a matrix of states (T(+), N)
        """
        state_prob_list = self.get_state_probs()
        state_prob_matrix = np.reshape(state_prob_list[0:-1], [2 * self.T, self.N + 1])  # get rid of terminal state and reshape
        return state_prob_matrix

    def get_table_contents(self):
        """
        :return: a matrix with action and state probability
        """
        table_content = []
        action_matrix = self.get_action_matrix()
        state_prob_matrix = self.get_state_prob_matrix()
        for i in range(2 * self.T):
            table_content_row = []
            for j in range(self.N + 1):
                table_content_row.append(f'{action_matrix[i, j]} {round(state_prob_matrix[i, j], 4)}')
            table_content.append(table_content_row)
        return table_content

    def get_reachable_state(self):
        policy_list = self.model.policy[0:-1]  # list of policy without policy at state terminate
        policy_matrix = np.reshape(policy_list, [2 * self.T, self.N + 1])
        reachable_state_matrix = np.zeros([self.T*2, self.N+1], dtype=bool)  # all states are set to not reachable
        reachable_state_matrix[0, 0] = True  # the first reachable state is (t=0, n=0)
        for t in range(1, self.T*2):
            for n in range(0, self.N+1):
                if reachable_state_matrix[t-1, n]:
                    action = policy_matrix[t-1, n]
                    if action <= self.N:
                        reachable_state_matrix[t, n:n+action+1] = True
                    elif action == self.N+1:
                        reachable_state_matrix[t, n] = True
        return reachable_state_matrix

    def get_color_matrix(self):
        reachable_state_matrix = self.get_reachable_state()
        policy_list = self.model.policy[0:-1]  # list of policy without policy at state terminate
        policy_matrix = np.reshape(policy_list, [2 * self.T, self.N + 1])  # reshape the policy vector
        color_list = []
        for t in range(0, self.T * 2):
            for n in range(0, self.N + 1):
                if reachable_state_matrix[t, n]:
                    color = self.__get_color(policy_matrix[t, n])
                    color_list.append(color)
                else:
                    color_list.append('#949494')
        color_matrix = np.reshape(color_list, [2 * self.T, self.N + 1])
        return color_matrix

    def expected_reward(self):
        expected_reward = self.model.V[0]
        print(f'Expected reward from built-in function: {expected_reward}')
        return expected_reward
    
    def compute_terminal_state_possibility(self):
        """
        TODO: compute terminal state possibility using actions.
        Note that our model include "time" into state and time never goes back
        After a certain number of steps, any path arrives the state with same time attribute
        Our method is based on this observation
        """
        # record the last state before "terminate"
        terminate_state_probs = np.zeros(self.total_num_of_states)
        # start from state 0
        cur_state = {0: 1}  # (state index, probability)
        next_state = {}
        for i in range(self.T * 2):  # this game ends in self.T * 2 steps
            while bool(cur_state):  # current state is not empty
                (state_index, probability) = cur_state.popitem()
                # get best strategy
                action = self.model.policy[state_index]
                # get the transition probs starting from this state
                transit_prob = self.P[action][state_index]
                for i in range(len(transit_prob)):
                    if transit_prob[i] > 0:
                        if i == self.terminate:
                            terminate_state_probs[state_index] += transit_prob[i]*probability
                        if i not in next_state.keys():
                            next_state[i] = transit_prob[i]*probability
                        else:
                            next_state[i] += transit_prob[i]*probability
            cur_state = cp.deepcopy(next_state)
            next_state = {}
        return terminate_state_probs[0:-1]
    
    def compute_reward_of_best_policy(self):
        """
        compute the money earned by attacker under optimal policy
        compute the cost of attacker under optimal policy
        compute the profit of attacker under optimal policy, which should be equal to self.model.V[0] in expected_reward()
        """
        terminate_state_probs = self.compute_terminal_state_possibility()
        # compute the reward
        average_reward = 0
        for i in range(len(terminate_state_probs)):
            if terminate_state_probs[i] > 0: 
                [time, plus, num_key_shares] = self.index_to_state(i)
                action = self.model.policy[i]
                reward = self.R[action][i][self.terminate]
                # print(f'state:({time}, {plus}, {num_key_shares}), reward: {reward}, prob: {terminate_state_probs[i]}')
                average_reward += reward * terminate_state_probs[i]

        # print(terminate_state_probs)
        # compute the cost
        average_cost = 0
        state_prob_matrix = self.get_state_prob_matrix()
        state_prob_vector = [item for row in state_prob_matrix for item in row]
        # print(state_prob_vector)
        for i in range(len(state_prob_vector)):
            [time, plus, num_key_shares] = self.index_to_state(i)
            if plus == 0:  # decision state
                action = self.model.policy[i]
                if action <= self.N:
                    average_cost += action * self.c * state_prob_vector[i]
        average_profit = average_reward - average_cost
        print(f'reward: {average_reward}, cost: {average_cost}, profit: {average_profit}')
        return [average_reward, average_cost, average_profit]

    def plot(self):
        average_reward, average_cost, average_profit = self.compute_reward_of_best_policy()

        table_content = self.get_table_contents()
        color_matrix = self.get_color_matrix()
        num_of_key_shares_list = [i for i in range(self.N + 1)]
        time_list = []
        for t in range(self.T):
            for plus in ['', '+']:
                time_list.append(f"{t}{plus}")

        # print(action_list)
        # print(num_of_key_shares_list)
        # print(time_list)
        plt.figure(figsize=(20, 10))
        plt.axis('off')
        table = plt.table(
            cellText=table_content,
            rowLabels=time_list,
            colLabels=num_of_key_shares_list,
            colWidths=[0.1 for x in num_of_key_shares_list],
            cellColours=color_matrix,
            cellLoc='center',
            loc='upper left')
        title = f'Best Policy, c_k = {self.N}, T = {self.T}, c_m = {self.M}, c = {self.c}, p = {self.p}, \n reward_fun = {self.reward_fun_name}, reward: {average_reward}, cost: {average_cost}, profit: {average_profit} '
        plt.title(title)
        plt.savefig("./plot/" + title + '.pdf')
        plt.show()
    
    def compute_holding_time(self):
        """
        compute the expected holding time: 
        starting from the first time step that the attacker has a key share 
        to the time that the game ends (turn in or exploit key shares)
        """
        state_prob_list = self.get_state_probs()
        best_policy = self.model.policy
        # (t, 0, n1) -> (t, 1, n2), n1 <= n2
        # (t, 1, n1) -> (t+1, 0, n1) or terminate
        expected_holding_time = 0
        start_state_dictionaries = []
        for k in range(self.total_num_of_states):
            start_state_dictionaries.append({})
        for t in range(self.T):
            for plus in range(2):
                for n in range(self.N+1):
                    state = self.state_to_index((t, plus, n))
                    if plus == 0 and n == 0:
                        # (t, 0, 0) --attack-> (t, 1, n) & n > 0
                        for next_n in range(1, self.N + 1):
                            next_state = self.state_to_index((t, 1, next_n))
                            start_state_dictionaries[next_state][t] = self.P[best_policy[state]][state][next_state] * state_prob_list[state]
                    else:
                        # attack best_policy[i] enclaves
                        # plus == 0, n > 0
                        if best_policy[state] <= self.N:
                            for start_time in start_state_dictionaries[state].keys():
                                for next_n in range(n, self.N + 1):
                                    next_state = self.state_to_index((t, 1, next_n))
                                    if start_time not in start_state_dictionaries[next_state].keys():
                                        start_state_dictionaries[next_state][start_time] = start_state_dictionaries[state][start_time] * self.P[best_policy[state]][state][next_state]
                                    else:
                                        start_state_dictionaries[next_state][start_time] += start_state_dictionaries[state][start_time] * self.P[best_policy[state]][state][next_state]
                        # wait 
                        elif best_policy[state] == self.N+1:
                            # (t, 1, n) --wait-> (t+1, 0, n) 
                            next_state = self.state_to_index((t+1, 0, n))
                            start_state_dictionaries[next_state] = cp.deepcopy(start_state_dictionaries[state])
                        # move funds or turn in keys (lead to termination of game)
                        elif best_policy[state] >= self.N+2:
                            for start_time in start_state_dictionaries[state].keys():
                                holding_time = t - start_time
                                expected_holding_time += start_state_dictionaries[state][start_time] * holding_time
        print(f'Expected holding time: {expected_holding_time}')
        return expected_holding_time


    def compute_average_key_shares(self):
        terminate_state_probs = self.compute_terminal_state_possibility()
        # print(terminate_state_probs)
        # print(f'Sum of Prob: {sum(terminate_state_probs)}')
        average_num_of_shares = 0
        for i in range(len(terminate_state_probs)):
            [time, plus, num_key_shares] = self.index_to_state(i)
            average_num_of_shares += num_key_shares * terminate_state_probs[i]
        print(f'Expected number of shares at the end of game: {average_num_of_shares}')
        return average_num_of_shares
    
    def compute_average_terminate_time(self):
        terminate_state_probs = self.compute_terminal_state_possibility()
        average_ter_time = 0
        for i in range(len(terminate_state_probs)):
            [time, plus, num_key_shares] = self.index_to_state(i)
            average_ter_time += time * terminate_state_probs[i]
        print(f'Expected number of shares at the end of game: {average_ter_time}')
        return average_ter_time
    
    def compute_exploit_key_prob(self):
        action_matrix = self.get_action_matrix()
        state_prob_matrix = self.get_state_prob_matrix()
        exploit_prob = 0
        for i in range(2 * self.T):
            table_content_row = []
            for j in range(self.N + 1):
                if action_matrix[i, j] == 'exploit key share(s)':
                    exploit_prob += state_prob_matrix[i, j]
        print("The prob of exploiting key is: ", exploit_prob)
        return exploit_prob
