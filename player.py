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

player0_winning_counts = []
player1_winning_counts = []
player2_winning_counts = []
player3_winning_counts = []
player4_winning_counts = []
player5_winning_counts = []
player6_winning_counts = []
player7_winning_counts = []
player8_winning_counts = []
player9_winning_counts = []
player10_winning_counts = []
player11_winning_counts = []
player12_winning_counts = []
player13_winning_counts = []
player14_winning_counts = []
player15_winning_counts = []

for delay in range (1, 11):
    num_player0_winning_delay = len(all_simulation_results[
                                      (all_simulation_results['winning_agent'] == 0) &
                                      (all_simulation_results['Delay'] == delay)
                                      ])
    player0_winning_counts.append(num_player0_winning_delay)
    print("Naive Agents Winning with delay", delay, ":", num_player0_winning_delay)

print("\n")

for delay in range (1, 11):
    num_player1_winning_delay = len(all_simulation_results[
                                      (all_simulation_results['winning_agent'] == 1) &
                                      (all_simulation_results['Delay'] == delay)
                                      ])
    player1_winning_counts.append(num_player1_winning_delay)
    print("Naive Agents Winning with delay", delay, ":", num_player1_winning_delay)

print("\n")

for delay in range (1, 11):
    num_player2_winning_delay = len(all_simulation_results[
                                      (all_simulation_results['winning_agent'] == 2) &
                                      (all_simulation_results['Delay'] == delay)
                                      ])
    player2_winning_counts.append(num_player2_winning_delay)
    print("Naive Agents Winning with delay", delay, ":", num_player2_winning_delay)