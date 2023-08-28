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

import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm

# Given mean and standard deviation
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

all_simulation_results = pd.read_csv('all_simulation_results_16agents2_run3.csv')

num_naive_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 3)])
num_adapt_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 4) & (all_simulation_results['winning_agent'] <= 7)])
num_lastminute_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['winning_agent'] <= 11)])
num_stealth_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 12) & (all_simulation_results['winning_agent'] <= 15)])
num_bluff_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 16) & (all_simulation_results['Profit'] > 0)
                                                ])

naive_winning_counts = []
adapt_winning_counts = []
lastminute_winning_counts = []
stealth_winning_counts = []
bluff_winning_counts = []

print("Naive Agents Winning:", num_naive_winning, num_naive_winning/30000)
print("Adaptive Agents Winning:", num_adapt_winning, num_adapt_winning/30000)
print("Last-minute Agents Winning:", num_lastminute_winning, num_lastminute_winning/30000)
print("Stealth Agents Winning:", num_stealth_winning, num_stealth_winning/30000)
print("Bluff Agents Winning:", num_bluff_winning, num_bluff_winning/30000)
print("\n")
for delay in range (1, 11):
    num_naive_winning_delay = len(all_simulation_results[
                                      (all_simulation_results['winning_agent'] >= 0) &
                                      (all_simulation_results['winning_agent'] <= 3) &
                                      (all_simulation_results['Delay'] == delay)
                                      ])
    naive_winning_counts.append(num_naive_winning_delay)
    print("Naive Agents Winning with delay", delay, ":", num_naive_winning_delay)
print("\n")
for delay in range (1, 11):
    num_adapt_winning_delay = len(all_simulation_results[
                                      (all_simulation_results['winning_agent'] >= 4) &
                                      (all_simulation_results['winning_agent'] <= 7) &
                                      (all_simulation_results['Delay'] == delay)
                                      ])
    adapt_winning_counts.append(num_adapt_winning_delay)
    print("Adaptive Agents Winning with delay", delay, ":", num_adapt_winning_delay)
print("\n")
for delay in range (1, 11):
    num_lastminute_winning_delay = len(all_simulation_results[
                                      (all_simulation_results['winning_agent'] >= 8) &
                                      (all_simulation_results['winning_agent'] <= 11) &
                                      (all_simulation_results['Delay'] == delay)
                                      ])
    lastminute_winning_counts.append(num_lastminute_winning_delay)
    print("Last-minute Agents Winning with delay", delay, ":", num_lastminute_winning_delay)
print("\n")
for delay in range (1, 11):
    num_stealth_winning_delay = len(all_simulation_results[
                                      (all_simulation_results['winning_agent'] >= 12) &
                                      (all_simulation_results['winning_agent'] <= 15) &
                                      (all_simulation_results['Delay'] == delay)
                                      ])
    stealth_winning_counts.append(num_stealth_winning_delay)
    print("Stealth Agents Winning with delay", delay, ":", num_stealth_winning_delay)

for delay in range (1, 11):
    num_bluff_winning_delay = len(all_simulation_results[(all_simulation_results['winning_agent'] == 16) &
                                      (all_simulation_results['Delay'] == delay) & (all_simulation_results['Profit'] > 0)])
    bluff_winning_counts.append(num_bluff_winning_delay)
    print("Bluff Agents Winning with delay", delay, ":", num_bluff_winning_delay)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), naive_winning_counts, label='Naive Agents', marker='o')
plt.plot(range(1, 11), adapt_winning_counts, label='Adaptive Agents', marker='o')
plt.plot(range(1, 11), lastminute_winning_counts, label='Last-minute Agents', marker='o')
plt.plot(range(1, 11), stealth_winning_counts, label='Stealth Agents', marker='o')
plt.plot(range(1, 11), bluff_winning_counts, label='Bluff Agents', marker='o')
plt.xlabel('Delay')
plt.ylabel('Number of Winning Agents')
plt.title('Number of Winning Agents by Agent Type and Delay')
plt.legend(loc='center right', bbox_to_anchor=(1, 0.7))
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()