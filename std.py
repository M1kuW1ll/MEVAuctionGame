import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
# Read all your dataframes
all_simulation_results1 = pd.read_csv('round4_profitupdate/eps/last_eps0std0.csv')
all_simulation_results2 = pd.read_csv('round4_profitupdate/eps/last_eps0std0.05.csv')
all_simulation_results3 = pd.read_csv('round4_profitupdate/eps/last_eps0std0.1.csv')
all_simulation_results4 = pd.read_csv('round4_profitupdate/eps/last_eps0std0.15.csv')
all_simulation_results5 = pd.read_csv('round4_profitupdate/eps/last_eps0std0.2.csv')


# Initialize lists to store profit means for each dataframe
naive_profit_means = []
adapt_profit_means = []
lastminute_profit_means = []
stealth_profit_means = []

# Process each dataframe and calculate profit means
dataframes = [all_simulation_results1, all_simulation_results2, all_simulation_results3, all_simulation_results4,
              all_simulation_results5]

for df in dataframes:
    naive_profit = df[(df['winning_agent'] >= 0) & (df['winning_agent'] <= 3) & (df['Delay'] == 1)]['Profit'].sum()
    naive_profit_means.append(naive_profit)

    adapt_profit = df[(df['winning_agent'] >= 4) & (df['winning_agent'] <= 7) & (df['Delay'] == 1)]['Profit'].sum()
    adapt_profit_means.append(adapt_profit)

    lastminute_profit = df[(df['winning_agent'] >= 8) & (df['winning_agent'] <= 11) & (df['Delay'] == 1)]['Profit'].sum()
    lastminute_profit_means.append(lastminute_profit)

# Set up the bar width and positions
bar_width = 0.15
r = np.arange(len(dataframes))  # Number of dataframes as positions

plt.figure(figsize=(12, 7))

# Create bar plots for each strategy
colors = ['blue', 'green', 'red']  # Customize the colors for each dataframe
strategies = ['Naive', 'Adaptive', 'Bluff']

for i, (means, color, strategy) in enumerate(zip([naive_profit_means, adapt_profit_means, lastminute_profit_means], colors, strategies)):
    r_i = [x + i * bar_width for x in r]
    plt.bar(r_i, means, width=bar_width, label=f'{strategy}', color=color)

plt.xlabel('standard deviation ($\sigma)')
plt.ylabel('Average Profit ($10^{-4}$ ETH)')

group_width = len(dataframes) * bar_width
center_positions = r + (group_width / 4)

custom_labels = ["0", "0.05", "0.1", "0.15", "0.2"]
plt.xticks(center_positions, custom_labels)
plt.legend()
plt.tight_layout()
plt.grid(axis='y')
plt.show()

simulation_data = {
    'Standard Deviation': [0, 0.05, 0.1, 0.15, 0.2],
    'Naive': naive_profit_means,
    'Adaptive': adapt_profit_means,
    'Last-minute': lastminute_profit_means
}

# Convert to DataFrame
simulation_df = pd.DataFrame(simulation_data)
simulation_df_melted = simulation_df.melt(id_vars='Standard Deviation', var_name='Strategy', value_name='Average Profit')

# Plot using seaborn
plt.figure(figsize=(20, 12))
sns.despine()
ax = sns.barplot(data=simulation_df_melted, x='Standard Deviation', y='Average Profit', hue='Strategy', palette='plasma')
plt.xlabel('standard deviation ($\sigma$)', fontsize = 45)
plt.ylabel('Average Profit ($10^{-4}$ ETH)', fontsize = 45)
plt.xticks(fontsize = 45)
plt.yticks(fontsize = 45)
legend = ax.legend(title=None, fontsize=45, ncol = 3)  # You can specify the fontsize as 'small', 'medium', 'large', or an integer value
legend.set_title(None)  # Remove legend title
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='gray', linestyle='-', linewidth=0.7)
plt.ylim(0,35)
# Show plot
plt.tight_layout()

plt.show()


naive_wins = []
adapt_wins = []
lastminute_wins = []
stealth_wins = []

for df in dataframes:
    naive_win = len(df[(df['winning_agent'] >= 0) & (df['winning_agent'] <= 3) & (df['Delay'] == 1)])/100
    naive_wins.append(naive_win)

    adapt_win = len(df[(df['winning_agent'] >= 4) & (df['winning_agent'] <= 7) & (df['Delay'] == 1)])/100
    adapt_wins.append(adapt_win)

    lastminute_win = len(df[(df['winning_agent'] >= 8) & (df['winning_agent'] <= 11) & (df['Delay'] == 1)])/100
    lastminute_wins.append(lastminute_win)

plt.figure(figsize=(15, 9))


plt.plot(custom_labels, naive_wins, label='Naive', marker='o', linestyle='-', linewidth = 3.3)
plt.plot(custom_labels, adapt_wins, label='Adaptive', marker='o', linestyle='-', linewidth = 3.3)
plt.plot(custom_labels, lastminute_wins, label='Bluff', marker='o', linestyle='-', linewidth = 3.3, color = 'red')
# plt.plot(custom_labels, stealth_wins, label='Stealth Agents Wins', marker='o', linestyle='-')

plt.xlabel('standard deviation ($\sigma$)', fontsize = 35)
plt.ylabel('Win Rate (%)', fontsize = 35)
plt.xticks(custom_labels, fontsize = 35)
plt.yticks(fontsize = 35)
plt.legend(fontsize = 35)
plt.ylim(15,67)
plt.grid(True)

ax_inset = inset_axes(plt.gca(), width='30%', height='30%', loc='center left', bbox_to_anchor=(0.35, 0.15, 0.9, 0.9), bbox_transform=plt.gcf().transFigure)

# Plot the same data on the inset
# Adjust the data range and styling for the inset plot if necessary
ax_inset.plot(custom_labels, naive_wins, marker='o', linestyle='-', linewidth=2)
ax_inset.plot(custom_labels, adapt_wins, marker='o', linestyle='-', linewidth=2)
ax_inset.plot(custom_labels, lastminute_wins, marker='o', linestyle='-', linewidth=2)
ax_inset.set_ylim(16, 20)
inset_y_ticks = [16, 18, 20]
ax_inset.set_yticks(inset_y_ticks)
ax_inset.set_yticklabels(inset_y_ticks, fontsize=24)
ax_inset.set_xticklabels(custom_labels, fontsize=24)

ax_inset.grid(True)

plt.show()