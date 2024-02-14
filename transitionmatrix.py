import itertools

import pandas as pd
import numpy as np
from open_spiel.python.egt import alpharank
import ast
file_path = 'payofftables/1008profiles_100.csv'
payoff_df = pd.read_csv(file_path)

num_strategies = 3
num_players = 10
delay_groups = [2, 2, 6]

# delay_group_profiles = [list(itertools.combinations_with_replacement(range(num_strategies), num_players)) for num_players in delay_groups]
# full_profiles = list(itertools.product(*delay_group_profiles))
full_profiles = list(itertools.product(range(num_strategies), repeat=sum(delay_groups)))
strategy_to_int = {'Naive': 0, 'Adaptive': 1, 'LastMinute': 2}
#
# Initialize a list of empty 3D numpy arrays, one for each player
payoff_matrices = [np.zeros((num_strategies,) * num_players) for _ in range(sum(delay_groups))]

for _, row in payoff_df.iterrows():
    # Convert the string representation of the 'Profile' and 'Accumulated Profits' columns back to a list and a dictionary
    profile = ast.literal_eval(row['Profile'])
    profits = ast.literal_eval(row['Accumulated Profits'])

    # Flatten the profile to get the strategy of each player
    flattened_profile = [strategy_to_int[strategy] for strategy in profile]
    print(flattened_profile)
    # Calculate the payoff for each player
    for player_id, strategy in enumerate(flattened_profile):
        # Set the value at that index in the matrix to the player's payoff in the current profile
        payoff_matrices[player_id][tuple(flattened_profile)] = profits[player_id]

# Compute the AlphaRank scores
_, _, pi, _, _ = alpharank.compute(payoff_matrices, alpha=1)

# Sort the strategy profiles by their scores in the stationary distribution
ranking = np.argsort(-pi)

# Print the ranked profiles along with their scores
print("Ranking of Strategy Profiles:")
for rank, index in enumerate(ranking, start=1):
    # Look up the strategy profile corresponding to the index
    profile = full_profiles[index]
    print(f"Rank {rank}: Profile {profile}, Alpha-Rank Score: {pi[index]}")
# num_players = 10
# num_strategies = 3
#
# # Initialize a list of empty 3D numpy arrays, one for each player
# payoff_matrices = [np.zeros((num_strategies, num_strategies, num_strategies)) for _ in range(num_players)]
#
# strategy_to_int = {'Naive': 0, 'Adaptive': 1, 'LastMinute': 2}
#
#
# # Initialize a tensor to hold the payoff matrices for each player
# payoff_tensor = np.zeros((num_players, num_strategies, num_strategies, num_strategies))
#
# # Iterate over the DataFrame
# for _, row in payoff_df.iterrows():
#     # Convert the string representation of the 'Profile' and 'Accumulated Profits' columns back to a list and a dictionary
#     profile = ast.literal_eval(row['Profile'])
#     profits = ast.literal_eval(row['Accumulated Profits'])
#
#     # For each player, find the index in the tensor that corresponds to the player's strategy in the current profile
#     for player_id, strategy in enumerate(profile):
#         # Map the strategy to an integer
#         strategy_int = strategy_to_int[strategy]
#         # Create an index for the tensor using the strategies of the players in the low, medium, and high latency groups
#         index = (player_id, strategy_int, strategy_int, strategy_int)
#         # Set the value at that index in the tensor to the player's payoff in the current profile
#         payoff_tensor[index] = profits[player_id]
# # Compute the AlphaRank scores
#
# # Create a list of all possible strategy profiles
# all_profiles = list(itertools.product(range(num_strategies), repeat=num_players))
#
# # Compute the AlphaRank scores
# _, _, pi, _, _ = alpharank.compute(payoff_matrices, alpha=1)
#
# # Sort the strategy profiles by their scores in the stationary distribution
# ranking = np.argsort(-pi)
#
# # Print the ranked profiles along with their scores
# print("Ranking of Strategy Profiles:")
# for rank, index in enumerate(ranking, start=1):
#     # Look up the strategy profile corresponding to the index
#     profile = all_profiles[index]
#     print(f"Rank {rank}: Profile {profile}, Alpha-Rank Score: {pi[index]}")

# meta_games = []
#     for _ in range(num_agents):
#         shape = [3] * num_agents
#         meta_game = np.zeros(shape=shape, dtype=np.float32)
#         meta_games.append(meta_game)
#     for meta_game, payoff in zip(meta_games, listified_values):
#         ist_iter = iter(payoff)
#         for coord in itertools.product(range(3), repeat=num_agents):
#             meta_game[coord] = next(list_iter)
# def global_min_max_normalize(matrix_list) :
#     # Find global minimum and maximum across all matrices
#     global_min = np.min([np.min(matrix) for matrix in matrix_list])
#     global_max = np.max([np.max(matrix) for matrix in matrix_list])
#
#     # Normalize each matrix using the global min and max
#     normalized_matrices = [(matrix - global_min) / (global_max - global_min) for matrix in matrix_list]
#
#     return normalized_matrices
#
#
# payoff_tables = global_min_max_normalize(meta_games)
# #payoff_tables = [meta_game/1000000 for meta_game in meta_games]
#
# # Run AlphaRank
# alpha = 1000
# payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
# _, _, pi, _, _ = alpharank.compute(payoff_tables, alpha=alpha, m=50)
# # pi,alpha = sweep_pi_vs_alpha(payoff_tables,return_alpha=True)