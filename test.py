# import matplotlib.pyplot as plt
# from scipy.stats import poisson, lognorm
import random

# def E(sigma, mean):
#     return poisson.rvs(sigma, mean)
#
# sigma = 0.5
# mean = 0.1
#

# random_variates = [E(sigma, mean) for t in times]
# cumulative_variates = np.cumsum(random_variates)
# plt.figure(figsize=(10, 6))
# plt.plot(times, cumulative_variates)
# plt.plot(times, random_variates)
# plt.xlabel('Time')
# plt.ylabel('E(t)')
# plt.title('Private Lognorm Process')
# plt.grid(True)
# plt.show()

# import numpy as np
# from scipy.optimize import minimize
# from scipy.stats import lognorm
#
# target_mean = 0.00027
# target_std = 0.097
#
#
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



import pandas as pd
import matplotlib.pyplot as plt

all_simulation_results = pd.read_csv('8naive8stealth_run3.csv')

num_naive_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 7)])
num_other_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['winning_agent'] <= 15)])
# num_bluff_winning_true = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['Profit'] > 0)])
# num_bluff_winning_bluff = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['Profit'] < 0) &
#                                                      (all_simulation_results['winning_bid_value'] > 0.25)])

other_winning_counts = []
bluff_winning_counts = []

print("Naive Agents Winning:", num_naive_winning)
# print("Bluff Agents Winning with true bid value:", num_bluff_winning_true, num_bluff_winning_true/(30000-num_bluff_winning_bluff))
# print("Bluff Agents Winning with bluff bid value", num_bluff_winning_bluff)
print("Other Agents Winning:", num_other_winning)

for delay in range (1, 11):
    num_stealth_winning_delay = len(all_simulation_results[
                                      (all_simulation_results['winning_agent'] >= 0) &
                                      (all_simulation_results['winning_agent'] <= 7) &
                                      (all_simulation_results['Delay'] == delay)
                                      ])
    other_winning_counts.append(num_stealth_winning_delay)
    print("Stealth Agents Winning with delay", delay, ":", num_stealth_winning_delay)

print("\n")

for delay in range (1, 11):
    num_bluff_winning_delay = len(all_simulation_results[
                                      (all_simulation_results['winning_agent'] >= 8) &
                                      (all_simulation_results['Delay'] == delay) &
                                      (all_simulation_results['Profit'] > 0)])
    bluff_winning_counts.append(num_bluff_winning_delay)
    print("Bluff Agents Winning with delay", delay, ":", num_bluff_winning_delay)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), other_winning_counts, label='Naive Agents', marker='o')
plt.plot(range(1, 11), bluff_winning_counts, label='Stealth Agents', marker='o')
plt.xlabel('Delay')
plt.ylabel('Number of Winning Agents')
plt.title('8 Naive VS 8 Stealth')
plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()