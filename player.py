import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

all_simulation_results = pd.read_csv('cancellation_4440.csv')
adaptive_results = pd.read_csv('round4_profitupdate/std0_new/last_eps0std0.csv')

num_player0_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 0)])
num_player1_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 1)])
num_player2_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 2)])
num_player3_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 3)])
num_player4_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 4)])
num_player5_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 5)])
num_player6_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 6)])
num_player7_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 7)])
num_player8_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 8)])
num_player9_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 9)])
num_player10_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 10)])
num_player11_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 11)])
# num_player12_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 12)])
# num_player13_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 13)])
# num_player14_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 14)])
# num_player15_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 15)])

# Number of players
num_players = 12

player_winning_counts = [[] for _ in range(num_players)]

for player in range(num_players) :

    for delay in range(1, 11) :
        num_player_winning_delay = len(all_simulation_results[
                                           (all_simulation_results['winning_agent'] == player) &
                                           (all_simulation_results['Delay'] == delay)
                                           ])
        player_winning_counts[player].append(num_player_winning_delay)

        print(f"Player {player} Winning with delay {delay} :", num_player_winning_delay)
    print("\n")
#Win rate per player
player_names = ["Naive Agent 1", "Naive Agent 2", "Naive Agent 3", "Naive Agent 4", "Adaptive Agent 1", "Adaptive Agent 2", "Adaptive Agent 3", "Adaptive Agent 4",
                "Bluff Agent 1", "Bluff Agent 2", "Bluff Agent 3", "Bluff Agent 4"]

plt.figure(figsize=(20, 12))

for player in range(num_players):
    plt.plot(range(1, 11), player_winning_counts[player], label=player_names[player])


plt.title('Winning Counts of All Players on Different Global Delays')
plt.xlabel('Global Delay (Step)')
plt.ylabel('Number of Wins')
plt.legend(loc='upper right', fontsize='small',bbox_to_anchor=(0.1, 0.3))
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.xticks(range(1, 11))
plt.tight_layout()
plt.show()

#Profit Distribution Per player
player_profits = []

for player in range(num_players):
    profits = all_simulation_results[(all_simulation_results['winning_agent'] == player)]['Profit']
    player_profits.append(profits)

blue_palette = plt.get_cmap('Blues', (num_players // 1))
red_palette = plt.get_cmap('Greens', (num_players // 1))
green_palette = plt.get_cmap('Reds', (num_players // 1))

plt.figure(figsize=(15, 8))

# Assign colors based on the player group
colors = []
for player in range(num_players):
    if player < 4:
        color = blue_palette(player + 2)
    elif player < 8:
        color = red_palette(player - 2)
    else:
        color = green_palette(player - 6)
    colors.append(color)

boxplot = plt.boxplot(player_profits, vert=True, patch_artist=True, whis=100)
for patch, color in zip(boxplot['boxes'], colors):
    patch.set(facecolor=color)

for median in boxplot['medians']:
    median.set(color='black', linewidth=2)

plt.title('Profit Distribution by Player, epsilon = 0 & std = 0')
plt.ylabel('Profit')
plt.xticks(range(1, num_players + 1), player_names, rotation=45)
plt.ylim(0.0064, 0.0067)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Extract data for the Naive agent with ID 0
naive_data = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 3)]
adapt_data = all_simulation_results[(all_simulation_results['winning_agent'] >= 4) & (all_simulation_results['winning_agent'] <= 7)]
lastminute_data = all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['winning_agent'] <= 11)]
# adapt2_data = adaptive_results[(adaptive_results['winning_agent'] >= 4) & (adaptive_results['winning_agent'] <= 7)]


naive_profits_by_delay = []
for delay in range(1, 11):
    profits = naive_data[naive_data['Delay'] == delay]['Profit']
    naive_profits_by_delay.append(profits)

adapt_profits_by_delay = []
for delay in range (1,11):
    profits = adapt_data[adapt_data['Delay'] == delay]['Profit']
    adapt_profits_by_delay.append(profits)

lastminute_profits_by_delay = []
for delay in range(1, 11):
    profits = lastminute_data[lastminute_data['Delay'] == delay]['Profit']
    lastminute_profits_by_delay.append(profits)

# adapt2_profits_by_delay = []
# for delay in range (1, 11):
#     profits = adapt2_data[adapt2_data['Delay'] == delay]['Profit']
#     adapt2_profits_by_delay.append(profits)


naive_data['Strategy'] = 'Naive'
adapt_data['Strategy'] = 'Adaptive (Profile 2)'
lastminute_data['Strategy'] = 'Last-minute/Bluff'


# Combine the data into a single DataFrame
combined_data = pd.concat([naive_data, adapt_data, lastminute_data])
num_strategies = combined_data['Strategy'].nunique()
palette = sns.color_palette("viridis", num_strategies)

delays = list(range(1, 11))
xtick_labels = [str(10 * delay) for delay in delays]
# Plotting
plt.figure(figsize=(12, 7))
ax = sns.boxplot(x='Delay', y='True Profit', hue='Strategy', data=combined_data,
                 showcaps=False, showfliers=False, whis=0, palette='viridis', dodge=True)

plt.title('Profit Distribution by Strategy on Different Global Delays (Epsilon=0, std of D=0)')
plt.xlabel('Global Delay (ms)')
plt.ylabel('Profit per Win (ETH)')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
handles, labels = ax.get_legend_handles_labels()

# Create a new patch for the median
median_patch = plt.Line2D([], [], color='black', label='Median')

# Combine current handles and labels with the new median_patch
handles.append(median_patch)
labels.append('Median')

plt.legend(handles, labels)

ax.set_xticklabels(xtick_labels)
sns.despine()
plt.show()


naive_data = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 3)]
adapt_data = all_simulation_results[(all_simulation_results['winning_agent'] >= 4) & (all_simulation_results['winning_agent'] <= 7)]
lastminute_data = all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['winning_agent'] <= 11)]
adapt2_data = adaptive_results[(adaptive_results['winning_agent'] >= 4) & (adaptive_results['winning_agent'] <= 7)]


naive_profits_by_delay = []
for delay in range(1, 11):
    profits = naive_data[naive_data['Delay'] == delay]['Profit'].sum()
    naive_profits_by_delay.append(profits)

adapt_profits_by_delay = []
for delay in range (1,11):
    profits = adapt_data[adapt_data['Delay'] == delay]['Profit'].sum()
    adapt_profits_by_delay.append(profits)

lastminute_profits_by_delay = []
for delay in range(1, 11):
    profits = lastminute_data[lastminute_data['Delay'] == delay]['Profit'].sum()
    lastminute_profits_by_delay.append(profits)

# adapt2_profits_by_delay = []
# for delay in range (1, 11):
#     profits = adapt2_data[adapt2_data['Delay'] == delay]['Profit'].sum()
#     adapt2_profits_by_delay.append(profits)


delays = list(range(1, 11))
data = {
    'Delay': delays * 3,
    'Profit': naive_profits_by_delay + adapt_profits_by_delay + lastminute_profits_by_delay,
    'Strategy': ['Naive']*10  + ['Adaptive']*10 + ['Last-minute']*10
}

df = pd.DataFrame(data)

# Setting the plasma palette
palette = sns.color_palette("plasma", 3)

# Creating the plot
plt.figure(figsize=(20, 10))

ax = sns.barplot(x='Delay', y='Profit', hue='Strategy', data=df, palette=palette)
ax.set_xticklabels(xtick_labels)
plt.xlabel('Global Delay (ms)', fontsize = 25)
plt.ylabel('Average Profit ( $10^{-4}$ ETH)', fontsize = 25)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.grid(True, axis='y', linestyle='--', linewidth=0.7)
plt.legend(fontsize = 25, ncol = 4)
plt.ylim(0,25)
plt.tight_layout()
plt.show()

adapt2_profits_by_delay = []
for delay in range (1, 11):
    profits = adapt2_data[adapt2_data['Delay'] == delay]['Profit'].sum()
    adapt2_profits_by_delay.append(profits)

naive_profits_by_delay = []
for delay in range(1, 11):
    profits = naive_data[naive_data['Delay'] == delay]['Profit'].median()
    naive_profits_by_delay.append(profits)

print(naive_profits_by_delay)
print(adapt2_profits_by_delay)
# naive_0_data = all_simulation_results[(all_simulation_results['winning_agent'] >= 4) & (all_simulation_results['winning_agent'] <= 7)]
#
# profits_by_delay = []
#
# for delay in range (1,11):
#     total_profit = naive_0_data[naive_0_data['Delay'] == delay]['Profit']
#     profits_by_delay.append(total_profit)
#
# plt.figure(figsize=(10, 6))
#
# # Create a bar plot for aggregated profits
# boxplot = plt.boxplot(profits_by_delay, patch_artist=True, boxprops=dict(facecolor="green"), whis = 10)
#
# for median in boxplot['medians']:
#     median.set(color='black', linewidth=2)
#
# plt.title('Profit Distribution of Adaptive Agent 1 on Different Global Delays')
# plt.xlabel('Global Delay (Step)')
# plt.ylabel('Profit (ETH)')
# plt.xticks(range(1, 11))  # Adjust the tick labels to match the delay values
# plt.ylim(0.0064, 0.0069)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
#
# plt.show()
