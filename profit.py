import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_simulation_results = pd.read_csv('GlobalDelay_10Naive')

avg_profit_naive = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 9)]['Profit'].mean()
# avg_profit_adapt = all_simulation_results[(all_simulation_results['winning_agent'] >= 4) & (all_simulation_results['winning_agent'] <= 7)]['Profit'].mean()
# avg_profit_lastminute = all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['winning_agent'] <= 11)]['Profit'].mean()
# avg_profit_stealth = all_simulation_results[(all_simulation_results['winning_agent'] >= 12) & (all_simulation_results['winning_agent'] <= 15)]['Profit'].mean()
# avg_profit_bluff_true = all_simulation_results[(all_simulation_results['winning_agent'] >= 16) & (all_simulation_results['Profit'] > 0)]['Profit'].mean()
# avg_profit_bluff_fake = all_simulation_results[(all_simulation_results['winning_agent'] >= 16) & (all_simulation_results['Profit'] < 0)]['Profit'].mean()

print("Average Profit for Naive Agents:", avg_profit_naive)
# print("Average Profit for Adaptive Agents:", avg_profit_adapt)
# print("Average Profit for Last-minute Agents:", avg_profit_lastminute)
# print("Average Profit for Stealth Agents:", avg_profit_stealth)
# print("Average Profit for Bluff Agents (True bid):", avg_profit_bluff_true)
# print("Average Profit for Bluff Agents (Fake bid):", avg_profit_bluff_fake)

naive_profit_means = []
adapt_profit_means = []
lastminute_profit_means = []
stealth_profit_means = []
bluff_profit_means = []

# Calculate average profit for each delay
for delay in range(1, 11):  # Assuming delays from 1 to 4 as in your provided code
    naive_profit = all_simulation_results[
        (all_simulation_results['winning_agent'] >= 0) &
        (all_simulation_results['winning_agent'] <= 9) &
        (all_simulation_results['Delay'] == delay)
    ]['Profit'].mean()
    naive_profit_means.append(naive_profit)

    adapt_profit = all_simulation_results[
        (all_simulation_results['winning_agent'] >= 4) &
        (all_simulation_results['winning_agent'] <= 7) &
        (all_simulation_results['Delay'] == delay)
    ]['Profit'].mean()
    adapt_profit_means.append(adapt_profit)

    lastminute_profit = all_simulation_results[
        (all_simulation_results['winning_agent'] >= 8) &
        (all_simulation_results['winning_agent'] <= 11) &
        (all_simulation_results['Delay'] == delay)
    ]['Profit'].mean()
    lastminute_profit_means.append(lastminute_profit)

    stealth_profit = all_simulation_results[
        (all_simulation_results['winning_agent'] >= 12) &
        (all_simulation_results['winning_agent'] <= 15) &
        (all_simulation_results['Delay'] == delay)
    ]['Profit'].mean()
    stealth_profit_means.append(stealth_profit)

    bluff_profit_true = all_simulation_results[
        (all_simulation_results['winning_agent'] >= 16) &
        (all_simulation_results['Profit'] > 0) &
        (all_simulation_results['Delay'] == delay)
    ]['Profit'].mean()
    bluff_profit_means.append(bluff_profit_true)

# Plotting average profit for each strategy type at different delays
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), naive_profit_means, label='Naive Agents Profit', marker='o')
plt.plot(range(1, 11), adapt_profit_means, label='Adaptive Agents Profit', marker='o')
plt.plot(range(1, 11), lastminute_profit_means, label='Last-minute Agents Profit', marker='o')
plt.plot(range(1, 11), stealth_profit_means, label='Stealth Agents Profit', marker='o')
plt.plot(range(1, 11), bluff_profit_means, label='Bluff Agents Profit (True bid)', marker='o')
plt.xlabel('Delay')
plt.ylabel('Average Profit')
plt.title('Average Profit by Strategy on Different Global Delays')
plt.legend(loc='center right', bbox_to_anchor=(0.35, 0.85))
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

naive_profits = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 3)]['Profit']
adapt_profits = all_simulation_results[(all_simulation_results['winning_agent'] >= 4) & (all_simulation_results['winning_agent'] <= 7)]['Profit']
lastminute_profits = all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['winning_agent'] <= 11)]['Profit']
stealth_profits = all_simulation_results[(all_simulation_results['winning_agent'] >= 12) & (all_simulation_results['winning_agent'] <= 15)]['Profit']

data = [naive_profits, adapt_profits, lastminute_profits, stealth_profits]
colors = ['green', 'blue', 'yellow', 'purple']

plt.figure(figsize=(10, 6))
boxplot = plt.boxplot(data, vert=True, patch_artist=True, whis=100)

for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

for median in boxplot['medians']:
    median.set(color='black', linewidth=2)

plt.title('Profit Distribution by Strategy')
plt.xlabel('Strategy')
plt.ylabel('Profit')
plt.xticks([1, 2, 3, 4], ['Naive', 'Adaptive', 'Last-minute', 'Stealth'])
plt.ylim(0.006, 0.007)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

strategies = ['Naive', 'Adaptive', 'Last-minute', 'Stealth']
for strategy_data, strategy_name in zip(data, strategies) :
    q1 = np.percentile(strategy_data, 25)
    q3 = np.percentile(strategy_data, 75)
    median_val = np.median(strategy_data)

    print(f"Strategy: {strategy_name}")
    print(f"Q1: {q1:.8f}")
    print(f"Median: {median_val:.8f}")
    print(f"Q3: {q3:.8f}")
    print("\n")
