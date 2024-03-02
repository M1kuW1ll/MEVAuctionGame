from sim import Auction, PlayerWithNaiveStrategy, PlayerWithAdaptiveStrategy, PlayerWithLastMinute
import itertools
import pyspiel
import pandas as pd
import numpy as np
from math import factorial
from itertools import permutations
import ast

strategies = ['Naive', 'Adaptive', 'LastMinute']
delays = [1]*10
all_indistinguishable_combinations = list(itertools.combinations_with_replacement(strategies, 10))

# Flatten the tuples and combine them into a full strategy profile for each combination
indistinguishable_profiles = []
for combination in all_indistinguishable_combinations:
    flattened_profile = list(itertools.chain(*combination)) # Flatten the tuple of tuples
    indistinguishable_profiles.append(flattened_profile)

print(f"Total number of profiles: {len(all_indistinguishable_combinations)}")

# Convert indistinguishable_profiles to a set for faster lookup
indistinguishable_profiles_set = set(map(tuple, indistinguishable_profiles))

profile_number = 1
data = []
accumulated_profits_list = []

for profile in all_indistinguishable_combinations:
    print(f"Running simulation with profile {profile_number}: {profile}")
    accumulated_profits = {player_id : 0 for player_id in range(len(profile))}
    for run in range(10) :
        model = Auction(profile, delay=1, rate_public_mean=0.082, rate_public_sd=0, rate_private_mean=0.04,
                        rate_private_sd=0, T_mean=12, T_sd=0.1)
        # Run the model steps as required
        for i in range(int(model.T * 100)) :
            model.step()

        for player_id, profit in model.player_profits.items() :
            accumulated_profits[player_id] += profit

    strategy_groups = {}
    for player_id, (strategy, delay) in enumerate(zip(profile, delays)) :
        key = (strategy, delay)
        if key not in strategy_groups :
            strategy_groups[key] = []
        strategy_groups[key].append(player_id)

    # Calculate the average profit for each group and assign it back
    for group in strategy_groups :
        group_player_ids = strategy_groups[group]
        avg_profit = np.mean([accumulated_profits[id] for id in group_player_ids])
        for player_id in group_player_ids :
            accumulated_profits[player_id] = avg_profit

    print(f"Accumulated profits for profile {profile_number}: {accumulated_profits}")

    data.append({
        "Profile Number" : profile_number,
        "Profile" : str(profile),  # Convert the profile list to a string for easy storage
        "Accumulated Profits" : accumulated_profits  # This stores the final adjusted profits
    })

    # accumulated_profits_list.append(list(accumulated_profits.values()))
    profile_number += 1

# Save the results of all distinguishable profiles to a CSV file
df = pd.DataFrame(data)
csv_file_path = 'test_payoff_66.csv'
df.to_csv(csv_file_path, index=False)

print(f"Results saved to {csv_file_path}")

# all_distinct_combinations = list(itertools.product(strategies, repeat=10))
# print(f"Total number of profiles: {len(all_distinct_combinations)}")
#
# profile_number = 1
# data = []
# accumulated_profits_list = []
#
# for profile in all_distinct_combinations :
#     print(f"Running simulation with profile {profile_number}: {profile}")
#     accumulated_profits = {player_id : 0 for player_id in range(len(profile))}
#     for run in range(10) :  # You may adjust the number of runs per profile as needed
#         model = Auction(profile, delay=1, rate_public_mean=0.082, rate_public_sd=0, rate_private_mean=0.04,
#                         rate_private_sd=0, T_mean=12, T_sd=0.1)
#
#         # Run the model steps as required
#         for i in range(int(model.T * 100)) :
#             model.step()
#
#         # Aggregate profits for this run
#         for player_id, profit in model.player_profits.items() :
#             accumulated_profits[player_id] += profit
#
#     print(f"Accumulated profits for profile {profile_number}: {accumulated_profits}")
#
#     data.append({
#         "Profile Number" : profile_number,
#         "Profile" : str(profile),  # Convert the profile list to a string for easy storage
#         "Accumulated Profits" : accumulated_profits  # This stores the final adjusted profits
#     })
#
#     accumulated_profits_list.append(list(accumulated_profits.values()))
#     profile_number += 1

# df = pd.DataFrame(data)
#
# # Define the path where you want to save the CSV file
# csv_file_path = 'test_payoff.csv'
#
# # Save the DataFrame to a CSV file
# df.to_csv(csv_file_path, index=False)
#
# print(f"Results saved to {csv_file_path}")