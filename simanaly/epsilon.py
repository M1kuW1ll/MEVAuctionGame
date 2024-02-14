import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Read all your dataframes
all_simulation_results1 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0std0.csv')
all_simulation_results2 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.01std0_delay4.csv')
all_simulation_results3 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.02std0_delay4.csv')
all_simulation_results4 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.03std0_delay4.csv')
all_simulation_results5 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.04std0_delay4.csv')
all_simulation_results6 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.05std0.csv')
all_simulation_results7 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.1std0.csv')
all_simulation_results8 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.15std0.csv')
all_simulation_results9 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.2std0.csv')
all_simulation_results10 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.25std0.csv')
all_simulation_results11 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.3std0.csv')

# Initialize lists to store profit means for each dataframe
naive_profit_means = []
adapt_profit_means = []
lastminute_profit_means = []
stealth_profit_means = []

# Process each dataframe and calculate profit means
dataframes = [all_simulation_results1, all_simulation_results2, all_simulation_results3, all_simulation_results4,
              all_simulation_results5, all_simulation_results6, all_simulation_results7, all_simulation_results8,
              all_simulation_results9, all_simulation_results10, all_simulation_results11]

# Initialize lists to store profit means for each dataframe

for df in dataframes:
    naive_profit = df[(df['winning_agent'] >= 0) & (df['winning_agent'] <= 3) & (df['Delay'] == 4)]['Profit'].sum()
    naive_profit_means.append(naive_profit)

    adapt_profit = df[(df['winning_agent'] >= 4) & (df['winning_agent'] <= 7) & (df['Delay'] == 4)]['Profit'].sum()
    adapt_profit_means.append(adapt_profit)

    lastminute_profit = df[(df['winning_agent'] >= 8) & (df['winning_agent'] <= 11) & (df['Delay'] == 4)]['Profit'].sum()
    lastminute_profit_means.append(lastminute_profit)

# Set up the bar width and positions
bar_width = 0.15
r = np.arange(len(dataframes))  # Number of dataframes as positions

plt.figure(figsize=(12, 7))

# Create bar plots for each strategy
colors = ['blue', 'green', 'red']  # Customize the colors for each dataframe
strategies = ['Naive', 'Adaptive', 'Last-minute/Stealth']

for i, (means, color, strategy) in enumerate(zip([naive_profit_means, adapt_profit_means, lastminute_profit_means], colors, strategies)):
    r_i = [x + i * bar_width for x in r]
    plt.bar(r_i, means, width=bar_width, label=f'{strategy}', color=color)

plt.xlabel('Epsilon (seconds)')
plt.ylabel('Aggregated Profit (ETH)')

plt.title('Impact of Revealing time (Epsilon) on Last-minute/Stealth Strategy Aggregated Profits (Global Delay = 4, std = 0)')

group_width = len(dataframes) * bar_width
center_positions = r + (group_width / 4)

custom_labels = ["0","10","20","30","40", "50", "100", "150", "200", "250", "300"]

plt.xticks(center_positions, custom_labels)
plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
plt.tight_layout()
plt.grid(axis='y')
plt.show()

naive_wins = []
adapt_wins = []
lastminute_wins = []
stealth_wins = []

for df in dataframes:
    naive_win = len(df[(df['winning_agent'] >= 0) & (df['winning_agent'] <= 3) & (df['Delay'] == 4)])/100
    naive_wins.append(naive_win)

    adapt_win = len(df[(df['winning_agent'] >= 4) & (df['winning_agent'] <= 7) & (df['Delay'] == 4)])/100
    adapt_wins.append(adapt_win)

    lastminute_win = len(df[(df['winning_agent'] >= 8) & (df['winning_agent'] <= 11) & (df['Delay'] == 4)])/100
    lastminute_wins.append(lastminute_win)

plt.figure(figsize=(15, 9))

plt.plot(custom_labels, naive_wins, label='Naive', marker='o', linestyle='-', linewidth=3.3 )
plt.plot(custom_labels, adapt_wins, label='Adaptive', marker='o', linestyle='-', linewidth=3.3)
plt.plot(custom_labels, lastminute_wins, label='Bluff', marker='o', linestyle='-', linewidth=3.3, color="red")
# plt.plot(custom_labels, stealth_wins, label='Stealth Agents Wins', marker='o', linestyle='-')

plt.xlabel('Revealing Time $\epsilon$ (ms)', fontsize = 35)
plt.ylabel('Win Rate (%)', fontsize = 35)
plt.xticks(custom_labels, fontsize = 35)
plt.legend(fontsize = 35)
plt.yticks(fontsize = 35)
plt.grid(True)
plt.ylim(20,46)
plt.show()

all_simulation_results1 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0std0.csv')
all_simulation_results2 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.05std0.csv')
all_simulation_results3 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.1std0.csv')
all_simulation_results4 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.15std0.csv')
all_simulation_results5 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.2std0.csv')
all_simulation_results6 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.25std0.csv')
all_simulation_results7 = pd.read_csv('../round4_profitupdate/std0_new/bluff_eps0.3std0.csv')

df2 = [all_simulation_results1, all_simulation_results2, all_simulation_results3, all_simulation_results4,
              all_simulation_results5, all_simulation_results6, all_simulation_results7]
custom_labels = ["0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3"]
median_profits = []

for df in df2:
    median_profit = []
    for delay in range(1, 11) :
        # Filter the dataframe for the specified range of winning_agent
        filtered_df = df[(df['winning_agent'] >= 4) & (df['winning_agent'] <= 7) & (df['Delay'] == delay)]
        median_profit.append(filtered_df['Profit'].median())
    median_profits.append(median_profit)

# Create a line plot for the median profit values
plt.figure(figsize=(10, 6))

for i, median_profit in enumerate(median_profits):
    plt.plot(range(1, 11), median_profit, label=f'Bluff Epsilon {custom_labels[i]}', marker='o', linestyle='-')

plt.title('Median of Profit per Win of Adaptive Players under the Impact of Bluff Epslion on Different Global Delays')
plt.xlabel('Global Delay (Step)')
plt.ylabel('Profit per Win (ETH)')
plt.xticks(range(1, 11))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()