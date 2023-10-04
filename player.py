import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_simulation_results = pd.read_csv('Round3/16uniform_round3_run2.csv')

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

plt.figure(figsize=(12, 7))

for player in range(num_players):
    plt.plot(range(1, 11), player_winning_counts[player], label=f'Player {player}')

plt.title('Winning counts of 16 players across delays')
plt.xlabel('Delay')
plt.ylabel('Number of Wins')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.xticks(range(1, 11))
plt.tight_layout()
plt.show()


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

# Display the plot
plt.show()
