import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_simulation_results = pd.read_csv('../round4_profitupdate/eff_10naive_0.8eof.csv')
efficiency = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 9)]['efficiency']
efficiency_means = []

for delay in range(1, 21):  # Assuming delays from 1 to 4 as in your provided code
    efficiency_delay = all_simulation_results[
        (all_simulation_results['winning_agent'] >= 0) &
        (all_simulation_results['winning_agent'] <= 9) &
        (all_simulation_results['Delay'] == delay)
    ]['efficiency'].mean()
    efficiency_means.append(efficiency_delay)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), efficiency_means, label='Efficiency', marker='o')
plt.xlabel('Delay')
plt.ylabel('Average Efficiency')
plt.title('Average Efficiency on Different Global Delays')
plt.legend(loc='center right', bbox_to_anchor=(0.35, 0.85))
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()

winning_bid = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) &
                                     (all_simulation_results['winning_agent'] <= 9)]['Profit']
efficiency_delay = []
for delay in range(1,21):
    bid = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (
            all_simulation_results['winning_agent'] <= 9) & (all_simulation_results['Delay'] == delay)]['efficiency']
    efficiency_delay.append(bid)

plt.figure(figsize=(10, 6))
boxplot = plt.boxplot(efficiency_delay, vert=True, patch_artist=True, whis=100)

for median in boxplot['medians']:
    median.set(color='black', linewidth=2)
for box in boxplot['boxes'] :
    box.set(facecolor='yellow')
plt.title('Efficiency Distribution on Different Global Delays')
plt.xlabel('Delay (Step)')
plt.ylabel('Efficiency')
plt.xticks(range(1,21))
plt.ylim(0.90, 0.97)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# df1 = pd.read_csv('eff_90eof1delay/eff_90eof1delay_1naive1adapt.csv')
# df2 = pd.read_csv('eff_90eof1delay/eff_90eof1delay_2naive2adapt.csv')
# df3 = pd.read_csv('eff_90eof1delay/eff_90eof1delay_3naive3adapt.csv')
# df4 = pd.read_csv('eff_90eof1delay/eff_90eof1delay_4naive4adapt.csv')
# df5 = pd.read_csv('eff_90eof1delay/eff_90eof1delay_5naive5adapt.csv')
# df6 = pd.read_csv('eff_90eof1delay/eff_90eof1delay_6naive6adapt.csv')
# df7 = pd.read_csv('eff_90eof1delay/eff_90eof1delay_7naive7adapt.csv')
# df8 = pd.read_csv('eff_90eof1delay/eff_90eof1delay_8naive8adapt.csv')
# dfs = [df1, df2, df3, df4, df5, df6, df7, df8]
# efficiency_all = []
# positions = np.arange(len(dfs))
#
# for df in dfs:
#     efficiency = df[(df['winning_agent'] >= 0) & (df['winning_agent'] <= 15)]['efficiency'].mean()
#     efficiency_all.append(efficiency)
#
# plt.figure(figsize=(12, 7))
#
#
# plt.plot(efficiency_all, marker='o', linestyle='-')
# plt.xlabel('Number of Naive/Adaptive Players in the game')
# plt.ylabel('Average Efficiency')
# plt.title('Impact of Number of Players on Auction Efficiency')
# plt.xticks(positions, [f'{i}' for i in range(1, len(dfs) + 1)])  # Use custom labels for dataframes
#
#
# plt.grid(True)
# plt.show()
#
#
# dff1 = pd.read_csv('eff_90eof_rdelay/eff_90eof_rdelay_1naive1adapt.csv')
# dff2 = pd.read_csv('eff_90eof_rdelay/eff_90eof_rdelay_2naive2adapt.csv')
# dff3 = pd.read_csv('eff_90eof_rdelay/eff_90eof_rdelay_3naive3adapt.csv')
# dff4 = pd.read_csv('eff_90eof_rdelay/eff_90eof_rdelay_4naive4adapt.csv')
# dff5 = pd.read_csv('eff_90eof_rdelay/eff_90eof_rdelay_5naive5adapt.csv')
# dff6 = pd.read_csv('eff_90eof_rdelay/eff_90eof_rdelay_6naive6adapt.csv')
# dff7 = pd.read_csv('eff_90eof_rdelay/eff_90eof_rdelay_7naive7adapt.csv')
# dff8 = pd.read_csv('eff_90eof_rdelay/eff_90eof_rdelay_8naive8adapt.csv')
# dffs = [dff1, dff2, dff3, dff4, dff5, dff6, dff7, dff8]
# efficiency_all2 = []
# positions = np.arange(len(dfs))
#
# for df in dffs:
#     efficiency = df[(df['winning_agent'] >= 0) & (df['winning_agent'] <= 15)]['efficiency'].mean()
#     efficiency_all2.append(efficiency)
#
# plt.figure(figsize=(12, 7))
#
#
# plt.plot(efficiency_all2, marker='o', linestyle='-')
# plt.xlabel('Number of Naive/Adaptive Players in the game')
# plt.ylabel('Average Efficiency')
# plt.title('Impact of Number of Players on Auction Efficiency')
# plt.xticks(positions, [f'{i}' for i in range(1, len(dffs) + 1)])  # Use custom labels for dataframes
#
#
# plt.grid(True)
# plt.show()


# Calculate quartiles for each delay
delays = list(range(1, 21))
medians = []
q1s = []
q3s = []

for delay in delays:
    data = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) &
                                  (all_simulation_results['winning_agent'] <= 9) &
                                  (all_simulation_results['Delay'] == delay)]['efficiency']
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    medians.append(median)
    q1s.append(q1)
    q3s.append(q3)

plt.figure(figsize=(10, 6))

# Plot lines for median, Q1, and Q3
plt.plot(delays, medians, label='Median', marker='o', linestyle='-')
plt.plot(delays, q1s, label='Q1', marker='o', linestyle='--')
plt.plot(delays, q3s, label='Q3', marker='o', linestyle='--')

plt.title('Impact of Global Delay on Auction Efficiency')
plt.xlabel('Global Delay (ms)')
plt.ylabel('Efficiency')

xtick_labels = [str(10 * delay) for delay in delays]

plt.xticks(delays, xtick_labels)
plt.ylim(0.9, 0.965)

plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()



# Plotting the results
plt.figure(figsize=(10, 6))

# Plotting the median as a line
plt.plot(delays, medians, label='Median', color='black', linestyle='-')

# Plotting Q1 and Q3 as areas around the median
plt.fill_between(delays, q1s, q3s, color='skyblue', alpha=0.5, label='Interquartile Range (Q1-Q3)')

plt.title('Impact of Global Delay on Auction Efficiency')
plt.xlabel('Global Delay (ms)')
plt.ylabel('Efficiency')

xtick_labels = [str(10 * delay) for delay in delays]
plt.xticks(delays, xtick_labels)

# Assuming a similar scale as the provided image
plt.ylim(0.89, 0.97)

plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

print(medians)

