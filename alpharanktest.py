import itertools
import pandas as pd
import numpy as np
from open_spiel.python.egt import alpharank
import ast
file_path = '1008profiles_100.csv'
payoff_df = pd.read_csv(file_path)

num_strategies = 3
num_players = 10
delay_groups = [2, 2, 6]

# delay_group_profiles = [list(itertools.combinations_with_replacement(range(num_strategies), num_players)) for num_players in delay_groups]
# full_profiles = list(itertools.product(*delay_group_profiles))
full_profiles = list(itertools.product(range(num_strategies), repeat=sum(delay_groups)))
strategy_to_int = {'Naive': 0, 'Adaptive': 1, 'LastMinute': 2}
#
payoff_vectors = [np.zeros(len(full_profiles)) for _ in range(sum(delay_groups))]

profile_to_index = {profile: index for index, profile in enumerate(full_profiles)}


for _, row in payoff_df.iterrows():
    # Convert the string representation of the 'Profile' and 'Accumulated Profits' columns back to a list and a dictionary
    profile = ast.literal_eval(row['Profile'])
    profits = ast.literal_eval(row['Accumulated Profits'])

    # Flatten the profile to get the strategy of each player
    flattened_profile = [strategy_to_int[strategy] for strategy in profile]
    index = profile_to_index[tuple(flattened_profile)]
    # Calculate the payoff for each player
    for player_id, strategy in enumerate(flattened_profile):
        # Set the value at that index in the matrix to the player's payoff in the current profile
        payoff_vectors[player_id][index] = profits[player_id]

payoff_matrices = [np.array([vector]) for vector in payoff_vectors]
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