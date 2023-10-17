import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_simulation_results = pd.read_csv('Round3/16uniform_round3_run1.csv')

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
num_player12_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 12)])
num_player13_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 13)])
num_player14_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 14)])
num_player15_winning = len(all_simulation_results[(all_simulation_results['winning_agent'] == 15)])

# Number of players
num_players = 16

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
                "Last-minute Agent 1", "Last-minute Agent 2", "Last-minute Agent 3", "Last-minute Agent 4", "Stealth Agent 1", "Stealth Agent 2", "Stealth Agent 3", "Stealth Agent 4"]

plt.figure(figsize=(20, 12))

for player in range(num_players):
    plt.plot(range(1, 11), player_winning_counts[player], label=player_names[player])


plt.title('Winning Counts of 16 Players on All Global Delays')
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
    profits = all_simulation_results[all_simulation_results['winning_agent'] == player]['Profit']
    player_profits.append(profits)

plt.figure(figsize=(15, 8))  # Adjust the figure size as per your requirements
boxplot = plt.boxplot(player_profits, vert=True, patch_artist=True, whis=100)
colors = plt.cm.plasma(np.linspace(0, 1, num_players))
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
for median in boxplot['medians']:
    median.set(color='black', linewidth=2)
plt.title('Profit Distribution by Player')
plt.xlabel('Player')
plt.ylabel('Profit')
player_labels = [f'Player {i}' for i in range(num_players)]
plt.xticks(range(1, num_players + 1), player_labels)
plt.ylim(0.0063, 0.007)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()

# Extract data for the Naive agent with ID 0
naive_data = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 3)]
adapt_data = all_simulation_results[(all_simulation_results['winning_agent'] >= 4) & (all_simulation_results['winning_agent'] <= 7)]
lastminute_data = all_simulation_results[(all_simulation_results['winning_agent'] >= 8) & (all_simulation_results['winning_agent'] <= 11)]
stealth_data = all_simulation_results[(all_simulation_results['winning_agent'] >= 12) & (all_simulation_results['winning_agent'] <= 15)]

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

stealth_profits_by_delay = []
for delay in range(1, 11):
    profits = stealth_data[stealth_data['Delay'] == delay]['Profit']
    stealth_profits_by_delay.append(profits)

# plt.figure(figsize=(20, 12))
#
# group_spacing = 0.2
#
# # Compute positions for each of the boxplots
# x_naive = np.arange(1, 11)
# x_adapt = [x + group_spacing for x in x_naive]
# x_lastminute = [x + group_spacing for x in x_adapt]
# x_stealth = [x + group_spacing for x in x_lastminute]
#
# whiskerprops = dict(linestyle='none')
#
# boxplot_naive = plt.boxplot(naive_profits_by_delay, positions=x_naive, widths=0.15, patch_artist=True, whis=100, showfliers=False, whiskerprops=whiskerprops)
# boxplot_adapt = plt.boxplot(adapt_profits_by_delay, positions=x_adapt, widths=0.15, patch_artist=True, whis=100, showfliers=False, whiskerprops=whiskerprops)
# boxplot_lastminute = plt.boxplot(lastminute_profits_by_delay, positions=x_lastminute, widths=0.15, patch_artist=True, whis=100, showfliers=False, whiskerprops=whiskerprops)
# boxplot_stealth = plt.boxplot(stealth_profits_by_delay, positions=x_stealth, widths=0.15, patch_artist=True, whis=100, showfliers=False, whiskerprops=whiskerprops)
#
# for box in boxplot_naive['boxes']:
#     box.set(facecolor='yellow')
# for box in boxplot_adapt['boxes'] :
#     box.set(facecolor='blue')
# for box in boxplot_lastminute['boxes'] :
#     box.set(facecolor='red')
# for box in boxplot_stealth['boxes'] :
#     box.set(facecolor='green')
#
# for median in boxplot_naive['medians']:
#     median.set(color='black', linewidth=2)
# for median in boxplot_adapt['medians']:
#     median.set(color='black', linewidth=2)
# for median in boxplot_lastminute['medians']:
#     median.set(color='black', linewidth=2)
# for median in boxplot_stealth['medians']:
#     median.set(color='black', linewidth=2)
#
#
# plt.title('Profit Distribution by Strategy on Different Global Delays')
# plt.xlabel('Delay (Step)')
# plt.ylabel('Profit (ETH)')
# group_centers = np.arange(1, 11) + 1.5 * group_spacing
# plt.xticks(group_centers, range(1,11))
# plt.ylim(0.0063, 0.0073)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.legend([boxplot_naive["boxes"][0], boxplot_adapt["boxes"][0], boxplot_lastminute["boxes"][0], boxplot_stealth["boxes"][0]], ['Naive', 'Adapt', 'Last-minute', 'Stealth'])
# plt.show()


# plt.figure(figsize=(20, 12))
#
# group_spacing = 0.2
#
#
# x_naive = np.arange(1, 11)
# x_adapt = [x + group_spacing for x in x_naive]
# x_lastminute = [x + group_spacing for x in x_adapt]
# x_stealth = [x + group_spacing for x in x_lastminute]
#
# bar_width = 0.15
#
# # Plot bars for each strategy
# plt.bar(x_naive, naive_profits_by_delay, width=bar_width, color='yellow', label='Naive')
# plt.bar(x_adapt, adapt_profits_by_delay, width=bar_width, color='blue', label='Adapt')
# plt.bar(x_lastminute, lastminute_profits_by_delay, width=bar_width, color='red', label='Last-minute')
# plt.bar(x_stealth, stealth_profits_by_delay, width=bar_width, color='green', label='Stealth')
#
# # Labeling and title
# plt.title('Average Profit on Different Global Delays for Different Strategies')
# plt.xlabel('Delay')
# plt.ylabel('Profit')
# group_centers = np.arange(1, 11) + 1.5 * group_spacing
# plt.xticks(group_centers, range(1,11))
#
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.legend()
# plt.show()

naive_0_data = all_simulation_results[all_simulation_results['winning_agent'] == 12]

profits_by_delay = []

for delay in range (1,11):
    total_profit = naive_0_data[naive_0_data['Delay'] == delay]['Profit']
    profits_by_delay.append(total_profit)

plt.figure(figsize=(10, 6))

# Create a bar plot for aggregated profits
boxplot = plt.boxplot(profits_by_delay, patch_artist=True, boxprops=dict(facecolor="blue"), whis = 100)

for median in boxplot['medians']:
    median.set(color='black', linewidth=2)

plt.title('Profit Distribution of Stealth Player 1 on Different Global Delays')
plt.xlabel('Delay')
plt.ylabel('Profit')
plt.xticks(range(1, 11))  # Adjust the tick labels to match the delay values
plt.ylim(0.00645, 0.007)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()
