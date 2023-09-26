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



import pandas as pd
import matplotlib.pyplot as plt

all_simulation_results = pd.read_csv('NaivePairwise/8naive8adapt(delta0.001)3.csv')

num_first_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 7)])
num_second_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['winning_agent'] <= 15)])
# num_bluff_winning_true = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['Profit'] > 0)])
# num_bluff_winning_bluff = len(all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['Profit'] < 0) &
#                                                      (all_simulation_results['winning_bid_value'] > 0.25)])

first_winning_counts = []
second_winning_counts = []

print("First Agents Winning:", num_first_winning)
# print("Bluff Agents Winning with true bid value:", num_bluff_winning_true, num_bluff_winning_true/(30000-num_bluff_winning_bluff))
# print("Bluff Agents Winning with bluff bid value", num_bluff_winning_bluff)
print("Second Agents Winning:", num_second_winning)

for delay in range (1, 11):
    num_first_winning_delay = len(all_simulation_results[
                                      (all_simulation_results['winning_agent'] >= 0) &
                                      (all_simulation_results['winning_agent'] <= 7) &
                                      (all_simulation_results['Delay'] == delay)
                                      ])
    first_winning_counts.append(num_first_winning_delay)
    print("First Agents Winning with delay", delay, ":", num_first_winning_delay)

print("\n")

for delay in range (1, 11):
    num_second_winning_delay = len(all_simulation_results[
                                      (all_simulation_results['winning_agent'] >= 8) &
                                      (all_simulation_results['Delay'] == delay)])
    second_winning_counts.append(num_second_winning_delay)
    print("Second Agents Winning with delay", delay, ":", num_second_winning_delay)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), first_winning_counts, label='Naive Agents', marker='o')
plt.plot(range(1, 11), second_winning_counts, label='Adaptive Agents', marker='o')
plt.xlabel('Delay')
plt.ylabel('Number of Winning Agents')
plt.title('8 Naive VS 8 Adaptive (delta 0.001)')
plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()