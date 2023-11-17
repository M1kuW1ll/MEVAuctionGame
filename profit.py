import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_simulation_results = pd.read_csv('round4_profitupdate/std0_new/bluff_eps0std0.csv')

naive_profits = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 3) & (all_simulation_results['Delay'] == 5)]['Profit']
adapt_profits = all_simulation_results[(all_simulation_results['winning_agent'] >= 4) & (all_simulation_results['winning_agent'] <= 7) & (all_simulation_results['Delay'] == 5)]['Profit']
lastminute_profits = all_simulation_results[(all_simulation_results['winning_agent'] >= 2) & (all_simulation_results['winning_agent'] <= 2) & (all_simulation_results['Delay'] == 5)]['Profit']
stealth_profits = all_simulation_results[(all_simulation_results['winning_agent'] >= 3) & (all_simulation_results['winning_agent'] <= 3) & (all_simulation_results['Delay'] == 5)]['Profit']

data = [naive_profits, adapt_profits, lastminute_profits, stealth_profits]
colors = ['green', 'blue', 'yellow', 'purple']

plt.figure(figsize=(10, 6))
boxplot = plt.boxplot(data, vert=True, patch_artist=True, whis=100)

for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

for median in boxplot['medians']:
    median.set(color='black', linewidth=2)

plt.title('Profit Distribution Stealth Players at Global Delay=2 (Profit Per Win)')
plt.xlabel('Strategy')
plt.ylabel('Profit (ETH)')
plt.xticks([1, 2, 3, 4], ['Naive', 'Adaptive', 'Last-minute', 'Stealth'])
plt.ylim(0.00645, 0.0067)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

avg_profit_naive = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 3)]['Profit'].sum()
avg_profit_adapt = all_simulation_results[(all_simulation_results['winning_agent'] >= 4) & (all_simulation_results['winning_agent'] <= 7)]['Profit'].sum()
avg_profit_lastminute = all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['winning_agent'] <= 11)]['Profit'].sum()
avg_profit_stealth = all_simulation_results[(all_simulation_results['winning_agent'] >= 12) & (all_simulation_results['winning_agent'] <= 15)]['Profit']


print("Average Profit for Naive Agents:", avg_profit_naive)
print("Average Profit for Adaptive Agents:", avg_profit_adapt)
print("Average Profit for Last-minute Agents:", avg_profit_lastminute)
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
        (all_simulation_results['winning_agent'] <= 3) &
        (all_simulation_results['Delay'] == delay)
    ]['Profit'].sum()
    naive_profit_means.append(naive_profit)


    adapt_profit = all_simulation_results[
        (all_simulation_results['winning_agent'] >= 4) &
        (all_simulation_results['winning_agent'] <= 7) &
        (all_simulation_results['Delay'] == delay)
    ]['Profit'].sum()
    adapt_profit_means.append(adapt_profit)


    lastminute_profit = all_simulation_results[
        (all_simulation_results['winning_agent'] >= 8) &
        (all_simulation_results['winning_agent'] <= 11) &
        (all_simulation_results['Delay'] == delay)
    ]['Profit'].sum()
    lastminute_profit_means.append(lastminute_profit)


    stealth_profit = all_simulation_results[
        (all_simulation_results['winning_agent'] >= 12) &
        (all_simulation_results['winning_agent'] <= 15) &
        (all_simulation_results['Delay'] == delay)
    ]['Profit'].sum()
    stealth_profit_means.append(stealth_profit)


    bluff_profit_true = all_simulation_results[
        (all_simulation_results['winning_agent'] >= 16) &
        (all_simulation_results['Profit'] > 0) &
        (all_simulation_results['Delay'] == delay)
    ]['Profit'].sum()
    bluff_profit_means.append(bluff_profit_true)

# Plotting average profit for each strategy type at different delays
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), naive_profit_means, label='Naive Agents Profit', marker='o')
plt.plot(range(1, 11), adapt_profit_means, label='Adaptive Agents Profit', marker='o')
plt.plot(range(1, 11), lastminute_profit_means, label='Bluff Agents Profit', marker='o')
# plt.plot(range(1, 11), stealth_profit_means, label='Bluff Agents Profit', marker='o')
# plt.plot(range(1, 11), bluff_profit_means, label='Bluff Agents Profit (True bid)', marker='o')
plt.xlabel('Delay')
plt.ylabel('Aggregated Profit')
plt.title('Aggregated Profit on Different Global Delays')
plt.legend(loc='center right', bbox_to_anchor=(0.35, 0.85))

plt.xticks(range(1, 11))
plt.grid(True)
plt.show()


# Setting up the bar width and positions
bar_width = 0.15
r1 = np.arange(1, 11)  # Positions for naive bars
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]

plt.figure(figsize=(12, 7))

# Creating bar plots
plt.bar(r1, naive_profit_means, width=bar_width, label='Naive Agents Profit', color='blue')
plt.bar(r2, adapt_profit_means, width=bar_width, label='Adaptive Agents Profit', color='green')
plt.bar(r3, lastminute_profit_means, width=bar_width, label='Bluff Agents Profit', color='red')
# plt.bar(r4, stealth_profit_means, width=bar_width, label='Bluff Agents Profit', color='yellow')
# plt.bar(r5, bluff_profit_means, width=bar_width, label='Bluff Agents Profit (True bid)', color='purple')

plt.xlabel('Delay (Step)')
plt.ylabel('Aggregated Profit (ETH)')
plt.title('Aggregated Profit by Strategy on Different Global Delays')
plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
plt.xticks([r + 2*bar_width for r in range(1, 11)], range(1, 11))
plt.tight_layout()
plt.grid(axis='y')
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
