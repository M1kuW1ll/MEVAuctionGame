import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm

# target_mean = 0.00027
# target_std = 0.097


# def loss(params) :
#     # Unpack parameters
#     mu, sigma = params
#
#     # Compute the mean and standard deviation for the log-normal distribution
#     mean = np.exp(mu + sigma ** 2 / 2.0)
#     std = ((np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)) ** 0.5
#
#     # Return the squared difference between the target and computed values
#     return (mean - target_mean) ** 2 + (std - target_std) ** 2
#
#
# # Initial guess
# init_params = [-5, 0.1]
#
# # Use the minimize function from scipy.optimize
# result = minimize(loss, init_params, method='L-BFGS-B', bounds=[(-20, 20), (0, 10)])
#
# # Extract the best parameters
# mu_best, sigma_best = result.x
#
# print(f"Best µ: {mu_best:.5f}, Best σ: {sigma_best:.5f}")

# import math
#
#
# def compute_log_params(mu_X, sigma_X2) :
#     # Calculate mean of the natural logarithm of X
#     mu = math.log(mu_X ** 2 / math.sqrt(mu_X ** 2 + sigma_X2))
#
#     # Calculate variance of the natural logarithm of X
#     sigma2 = math.log(1 + sigma_X2 / mu_X ** 2)
#     sigma = math.sqrt(sigma2)
#     return mu, sigma
#
#
# # Example usage:
# mu_X = 0.00027  # mean of X
# sigma_X2 = 0.0977  # variance of X, which is square of standard deviation
#
# mu, sigma = compute_log_params(mu_X, sigma_X2)
# print(f"Mean of ln(X): {mu}")
# print(f"Variance of ln(X): {sigma}")


# import pandas as pd
# import matplotlib.pyplot as plt
#
# all_simulation_results = pd.read_csv('')
#
# num_first_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 7)])
# num_second_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['winning_agent'] <= 15)])
# # num_bluff_winning_true = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['Profit'] > 0)])
# # num_bluff_winning_bluff = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['Profit'] < 0) &
# #                                                      (all_simulation_results['winning_bid_value'] > 0.25)])
#
# first_winning_counts = []
# second_winning_counts = []
#
# print("First Agents Winning:", num_first_winning)
# # print("Bluff Agents Winning with true bid value:", num_bluff_winning_true, num_bluff_winning_true/(30000-num_bluff_winning_bluff))
# # print("Bluff Agents Winning with bluff bid value", num_bluff_winning_bluff)
# print("Second Agents Winning:", num_second_winning)
#
# for delay in range (1, 11):
#     num_first_winning_delay = len(all_simulation_results[
#                                       (all_simulation_results['winning_agent'] >= 0) &
#                                       (all_simulation_results['winning_agent'] <= 7) &
#                                       (all_simulation_results['Delay'] == delay)
#                                       ])
#     first_winning_counts.append(num_first_winning_delay)
#     print("First Agents Winning with delay", delay, ":", num_first_winning_delay)
#
# print("\n")
#
# for delay in range (1, 11):
#     num_second_winning_delay = len(all_simulation_results[
#                                       (all_simulation_results['winning_agent'] >= 8) &
#                                       (all_simulation_results['Delay'] == delay)])
#     second_winning_counts.append(num_second_winning_delay)
#     print("Second Agents Winning with delay", delay, ":", num_second_winning_delay)
#
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), first_winning_counts, label='Adaptive Agents)', marker='o')
# plt.plot(range(1, 11), second_winning_counts, label='Last-minute Agents', marker='o')
# plt.xlabel('Delay')
# plt.ylabel('Number of Winning Agents')
# plt.title('8 Adaptive VS 8 Last-Minute (NEW)')
# plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
# plt.xticks(range(1, 11))
# plt.grid(True)
# plt.show()
#
# avg_profit_first = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 7)]['Profit'].mean()
# avg_profit_second = all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['winning_agent'] <= 15)]['Profit'].mean()
#
# print("Average Profit for First Agents:", avg_profit_first)
# print("Average Profit for Second Agents:", avg_profit_second)
# first_profit_means = []
# second_profit_means = []
#
# # Calculate average profit for each delay
# for delay in range(1, 11):  # Assuming delays from 1 to 4 as in your provided code
#     naive_profit = all_simulation_results[
#         (all_simulation_results['winning_agent'] >= 0) &
#         (all_simulation_results['winning_agent'] <= 7) &
#         (all_simulation_results['Delay'] == delay)
#     ]['Profit'].mean()
#     first_profit_means.append(naive_profit)
#
#     adapt_profit = all_simulation_results[
#         (all_simulation_results['winning_agent'] >= 8) &
#         (all_simulation_results['winning_agent'] <= 15) &
#         (all_simulation_results['Delay'] == delay)
#     ]['Profit'].mean()
#     second_profit_means.append(adapt_profit)
#
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), first_profit_means, label='First Agents Profit', marker='o')
# plt.plot(range(1, 11), second_profit_means, label='Second Agents Profit', marker='o')
# plt.xlabel('Delay')
# plt.ylabel('Average Profit')
# plt.title('8 Adaptive VS 8 Last-minute (delta 0.0001)')
# plt.legend(loc='center right', bbox_to_anchor=(0.35, 0.85))
# plt.xticks(range(1, 11))
#
# plt.grid(True)
# plt.show()

