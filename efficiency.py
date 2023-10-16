import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_simulation_results = pd.read_csv('GlobalDelay_10Naive.csv')
# efficiency = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 9)]['efficiency']
# efficiency_means = []
#
# for delay in range(1, 11):  # Assuming delays from 1 to 4 as in your provided code
#     efficiency_delay = all_simulation_results[
#         (all_simulation_results['winning_agent'] >= 0) &
#         (all_simulation_results['winning_agent'] <= 9) &
#         (all_simulation_results['Delay'] == delay)
#     ]['efficiency'].mean()
#     efficiency_means.append(efficiency_delay)
#
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), efficiency_means, label='Efficiency', marker='o')
# plt.xlabel('Delay')
# plt.ylabel('Average Efficiency')
# plt.title('Average Efficiency on Different Global Delays')
# plt.legend(loc='center right', bbox_to_anchor=(0.35, 0.85))
# plt.xticks(range(1, 11))
# plt.grid(True)
# plt.show()

winning_bid = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) &
                                     (all_simulation_results['winning_agent'] <= 9)]['Profit']
winning_bid_delay = []
for delay in range(1,11):
    bid = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (
            all_simulation_results['winning_agent'] <= 9) & (all_simulation_results['Delay'] == delay)]['Profit']
    winning_bid_delay.append(bid)

plt.figure(figsize=(10, 6))
boxplot = plt.boxplot(winning_bid_delay, vert=True, patch_artist=True, whis=100)

for median in boxplot['medians']:
    median.set(color='black', linewidth=2)

plt.title('Efficiency Distribution on Different Global Delays')
plt.xlabel('Delay (Step)')
plt.ylabel('Profit (ETH)')
plt.xticks(range(1,11))
plt.ylim(0.0063, 0.007)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
