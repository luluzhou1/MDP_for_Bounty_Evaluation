import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns
from mdp import MDP
from reward import *

"""
In this file, we change the cost (c) and success rate (p) to model attackers with different ability.
We first compute the optimal epsilon for a pair (c, p) which renders the lowest score (based on the objective function).
Second, for each c and p in a range, we compute the scores.
Then, we show the scores in a heatmap.
"""

def objective_fun(reward, exploit_prob, expected_holding_time, alpha_1, alpha_2, M, T):
    """
    The objective function is the weighted average of the normalized reward, exploit probability and expected holding time
    alpha_1 and alpha_2 are parameters set by defender
    """
    print(f"reward: {reward}, exploit_prob: {exploit_prob}, expected_holding_time: {expected_holding_time}")
    return alpha_1 * (reward / M) + alpha_2 * exploit_prob + (1 - alpha_1 - alpha_2) * (expected_holding_time / T)

def get_score(N, T, M, c, p, epsilon, cap, alpha_1, alpha_2, reward_fun, allow_sell_share=True):
    """
    Given all the params and reward fun, compute the score (based on the objective function)
    """
    model = MDP(N=N, T=T, M=M, p=p, c=c, epsilon = epsilon, cap = cap, reward_fun=reward_fun, reward_fun_name=" ", allow_sell_share=True)  # the reward_fun_name here does not matter
    model.run_mdp()
    reward = model.compute_reward_of_best_policy()[0]
    exploit_prob = model.compute_exploit_key_prob()
    expected_holding_time = model.compute_holding_time()
    score = objective_fun(reward, exploit_prob, expected_holding_time, alpha_1, alpha_2, M, T)
    return score

def find_epsilon(c, p, N, T, M, alpha_1, alpha_2, cap, reward_fun, allow_sell_share=True):
    """ 
    find the optimal epsilon given c, p, N, T, M
    """
    epsilon_list = np.arange(0, 1, 0.01)  # we will pick an optimal one from the list
    opt_score = 10000  # a large number
    for epsilon in epsilon_list:
        # compute the reward and exploit_prob
        cur_score = get_score(N=N, T=T, M=M, p=p, c=c, epsilon = epsilon, cap = cap, alpha_1=alpha_1, alpha_2=alpha_2, reward_fun=reward_fun, allow_sell_share=allow_sell_share)
        if cur_score < opt_score:  # since the score is attacker's reward and exploit probability, we want smaller score.
            opt_score = cur_score
            opt_epsilon = epsilon
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(f"optimal epsilon is: {opt_epsilon} with cap: {cap}")
    return opt_epsilon

def compute_f_score_list(estimated_c, estimated_p, reward_fun, c_list, p_list, N, T, M, alpha_1, alpha_2, cap, file_name, reward_fun_name, allow_sell_share=True):
    """
    find optimal epsilon (if needed), compute the scores (using get_score) and store them in .csv file
    """
    if reward_fun_name == "Our reward function":
        opt_epsilon = find_epsilon(estimated_c, estimated_p, N, T, M, alpha_1, alpha_2, cap, reward_fun, allow_sell_share)
    else:
        opt_epsilon = 0  # optimization of epsilon is not needed for other reward functions, epsilon is not used.

    # opt_epsilon = 0.2

    score_list = []
    for c in c_list:
        for p in p_list:
            score = get_score(N, T, M, c, p, opt_epsilon, cap, alpha_1, alpha_2, reward_fun, allow_sell_share)
            # exploit_prob_list.append((c * N / (p * M), p, exploit_prob))
            score_list.append((round(c, 1), round(p, 1), score))  # in our use-case p increases by 0.1, so we round it by 1.

    # write the result to a csv file
    with open(file_name, 'w') as f:
        for tuple in score_list:
            csv_writer = csv.writer(f)
            csv_writer.writerow(tuple)

def get_data_from_csv(data_file_name):
    # get the data from the csv file 
    data = []
    with open(data_file_name) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            data.append((float(row[0]), float(row[1]), float(row[2])))

    df = pd.DataFrame(data, columns=['relative_c', 'p', 'exploit_prob'])

    # Pivot the DataFrame to create a matrix suitable for a heatmap
    heatmap_data = df.pivot('relative_c', 'p', 'exploit_prob')
    return heatmap_data

def draw_subplots_under_certain_reward_fun_with_diff_caps(data_file_name_list, cap_list, suptitle):
    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    for i in range(len(data_file_name_list)):
        heatmap_data = get_data_from_csv(data_file_name_list[i])
        ax = axes[i]
        # Create the heatmap using Seaborn
        heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='Reds', ax=ax)
        title = f"Max bounty reward / secret value = {cap_list[i]}"
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel("probability of success at one step", fontsize=15, fontweight='bold')
        ax.set_ylabel("cost per step", fontsize=15, fontweight='bold')

    plt.suptitle(suptitle)

    # Save and show the plot
    plt.savefig("./plot/" + suptitle + '.pdf', format='pdf', bbox_inches='tight')
    # plt.show()

def draw_subplots_under_same_cap_diff_reward_funs(data_file_name_list, suptitle, reward_fun_name_list):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for i in range(len(data_file_name_list)):
        heatmap_data = get_data_from_csv(data_file_name_list[i])
        ax = axes[i]
        # Create the heatmap using Seaborn
        heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='Reds', ax=ax)
        title = reward_fun_name_list[i]
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel("probability of success at one step $p_s$", fontsize=15, fontweight='bold')
        ax.set_ylabel("cost per step $c_a$", fontsize=15, fontweight='bold')

    # plt.suptitle(suptitle)

    # Save and show the plot
    plt.savefig("./plot/" + suptitle + '.pdf', format='pdf', bbox_inches='tight')
    print(f"figure saved to ./plot/{suptitle}.pdf")
    # plt.show()

def plot_under_certain_reward_fun_with_diff_caps(T, M , N, cap_list, c_list, p_list, alpha_1, alpha_2, reward_fun, reward_fun_name, compute_csv=False, allow_sell_share=True):
    # experiments assuming key shares could be sold
    # edit mdp.py, function: self.move_found to change
    data_file_name_list = []
    for cap in cap_list:
        # generate csv file name list
        if allow_sell_share:
            file_name = f'./output/scores_T={T}_M={M}_N={N}_alpha_1={alpha_1}_alpha_2={alpha_2}_cap={cap}_allow_selling_key_share_reward_fun={reward_fun_name}.csv'
        else:
            file_name = f'./output/scores_T={T}_M={M}_N={N}_alpha_1={alpha_1}_alpha_2={alpha_2}_cap={cap}_selling_only_full_key_reward_fun={reward_fun_name}.csv'
        data_file_name_list.append(file_name)
        # compute the exploit probability if compute_csv is set as True (only need to be run once, for a fixed set of parameters)
        if compute_csv:
            compute_f_score_list(estimated_c=0.4, estimated_p=0.4, reward_fun=reward_fun,
                                    c_list=c_list, p_list=p_list, N=N, T=T, M=M, alpha_1=alpha_1, alpha_2=alpha_2, cap=cap, 
                                    file_name=file_name, reward_fun_name=reward_fun_name, allow_sell_share=allow_sell_share)
    if allow_sell_share:
        suptitle = f"Score: alpha_1={alpha_1}, alpha_2={alpha_2}, reward_fun: {reward_fun_name}, T={T}, M={M}, N={N}, allow selling key shares"
    else:
        suptitle = f"Score: alpha_1={alpha_1}, alpha_2={alpha_2}, reward_fun: {reward_fun_name}, T={T}, M={M}, N={N}, only allow selling full keys"
    draw_subplots_under_certain_reward_fun_with_diff_caps(data_file_name_list, cap_list, suptitle)
    plt.show()

def plot_under_same_cap_diff_reward_funs(T, M, N, cap, c_list, p_list, alpha_1, alpha_2, reward_fun_list, reward_fun_name_list, compute_csv=False, allow_sell_share=True):
    # experiments assuming key shares could be sold
    # edit mdp.py, function: self.move_found to change
    data_file_name_list = []
    for i in range(len(reward_fun_list)):
        # generate csv file name list
        if allow_sell_share:
            file_name = f'./output/scores_T={T}_M={M}_N={N}_alpha_1={alpha_1}_alpha_2={alpha_2}_cap={cap}_allow_selling_key_share_with_one_epsilon_reward_fun={reward_fun_name_list[i]}.csv'
        else:
            file_name = f'./output/scores_T={T}_M={M}_N={N}_alpha_1={alpha_1}_alpha_2={alpha_2}_cap={cap}_selling_only_full_key_with_one_epsilon_reward_fun={reward_fun_name_list[i]}.csv'
        data_file_name_list.append(file_name)
        # compute the exploit probability (only need to be run once)
        if compute_csv:
            compute_f_score_list(estimated_c=0.4, estimated_p=0.4, reward_fun=reward_fun_list[i],
                                    c_list=c_list, p_list=p_list, N=N, T=T, M=M, alpha_1=alpha_1, alpha_2=alpha_2, cap=cap, 
                                    file_name=file_name, reward_fun_name=reward_fun_name_list[i], allow_sell_share=allow_sell_share)
    if allow_sell_share:
        suptitle = f"Score: alpha_1={alpha_1}, alpha_2={alpha_2}, reward_fun: several reward funs, cap={cap}, T={T}, M={M}, N={N}, allow selling key shares"
    else:
        suptitle = f"Score: alpha_1={alpha_1}, alpha_2={alpha_2}, reward_fun: several reward funs, cap={cap}, T={T}, M={M}, N={N}, only allow selling full keys"
    draw_subplots_under_same_cap_diff_reward_funs(data_file_name_list, suptitle, reward_fun_name_list)

def experiment_under_certain_reward_fun_diff_caps(allow_sell_share = True):
    """
    This function produces heatmaps for different capital caps, but the same reward function.
    """
    ## experiments assuming key shares could be sold, same reward function in a plot.
    # generate the list of c (cost)
    c_list = np.arange(0, 1, 0.1)
    # generate the list of p (success probability)
    p_list = np.arange(0, 1, 0.1)
    cap_list = [0.2, 0.4, 0.6, 0.8]
    # reward function = linear
    plot_under_certain_reward_fun_with_diff_caps(10, 6, 3, cap_list, c_list, p_list, 0, 0, reward_fun_reward_acc_value, "linear", True, allow_sell_share)
    # reward_fun state of art
    plot_under_certain_reward_fun_with_diff_caps(10, 6, 3, cap_list, c_list, p_list, 0, 0, reward_fun_state_of_art, "state of art", True, allow_sell_share)
    # reward_fun no bounty
    plot_under_certain_reward_fun_with_diff_caps(10, 6, 3, cap_list, c_list, p_list, 0, 0, reward_fun_no_bounty, "no bounty", True, allow_sell_share)

def experiment_under_same_cap_diff_reward_funs(cap, compute_csv, allow_sell_share = True):
    """
    This function produces a heatmap for different reward functions, but the same capital cap.
    """
    ## experiments assuming key shares could be sold, same capital cap in a plot.
    # generate the list of c (cost)
    c_list = np.arange(0, 1, 0.1)
    # generate the list of p (success probability)
    p_list = np.arange(0, 1, 0.1)
    reward_fun_list = [reward_fun_state_of_art, reward_fun_reward_acc_value, reward_fun_no_bounty]
    reward_fun_name_list = ["Our reward function", "linear", "no bounty"]
    plot_under_same_cap_diff_reward_funs(10, 6, 3, cap, c_list, p_list, 1/3, 1/3, reward_fun_list, reward_fun_name_list, compute_csv=compute_csv, allow_sell_share=allow_sell_share)

if __name__ == "__main__":
    experiment_under_same_cap_diff_reward_funs(cap=0.8, compute_csv=True, allow_sell_share=True)
