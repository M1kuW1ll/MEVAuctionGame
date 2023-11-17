import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read all your dataframes
all_simulation_results1 = pd.read_csv('round4_profitupdate/eps/bluff_eps0.2std0.csv')
all_simulation_results2 = pd.read_csv('round4_profitupdate/eps/bluff_eps0.2std0.05.csv')
all_simulation_results3 = pd.read_csv('round4_profitupdate/eps/bluff_eps0.2std0.1.csv')
all_simulation_results4 = pd.read_csv('round4_profitupdate/eps/bluff_eps0.2std0.15.csv')
all_simulation_results5 = pd.read_csv('round4_profitupdate/eps/bluff_eps0.2std0.2.csv')


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

plt.xlabel('std of D')
plt.ylabel('Aggregated Profit (ETH)')
plt.title('Impact of Auction Termination time (std) on Bluff Strategy Aggregated Profits (Epsilon = 0.2)')

group_width = len(dataframes) * bar_width
center_positions = r + (group_width / 4)

custom_labels = ["0", "0.05", "0.1", "0.15", "0.2"]
plt.xticks(center_positions, custom_labels)
plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
plt.tight_layout()
plt.grid(axis='y')
plt.show()

naive_wins = []
adapt_wins = []
lastminute_wins = []
stealth_wins = []

for df in dataframes:
    naive_win = len(df[(df['winning_agent'] >= 0) & (df['winning_agent'] <= 3) & (df['Delay'] == 1)])
    naive_wins.append(naive_win)

    adapt_win = len(df[(df['winning_agent'] >= 4) & (df['winning_agent'] <= 7) & (df['Delay'] == 1)])
    adapt_wins.append(adapt_win)

    lastminute_win = len(df[(df['winning_agent'] >= 8) & (df['winning_agent'] <= 11) & (df['Delay'] == 1)])
    lastminute_wins.append(lastminute_win)

plt.figure(figsize=(12, 7))


plt.plot(custom_labels, naive_wins, label='Naive', marker='o', linestyle='-')
plt.plot(custom_labels, adapt_wins, label='Adaptive', marker='o', linestyle='-')
plt.plot(custom_labels, lastminute_wins, label='Bluff', marker='o', linestyle='-')
# plt.plot(custom_labels, stealth_wins, label='Stealth Agents Wins', marker='o', linestyle='-')

plt.xlabel('std of D')
plt.ylabel('Number of Wins')
plt.title('Impact of Auction termination time (std) on Bluff Strategy Win Rate (Epsilon = 0.2)')
plt.xticks(custom_labels)
plt.legend(loc='center right', bbox_to_anchor=(0.95, 0.45))

plt.grid(True)
plt.show()