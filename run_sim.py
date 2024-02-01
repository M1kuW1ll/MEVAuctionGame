from sim import Auction, PlayerWithNaiveStrategy, PlayerWithAdaptiveStrategy, PlayerWithLastMinute
import itertools
import numpy as np

strategies = ['Naive', 'Adaptive', 'LastMinute']

low_latency_combinations = list(itertools.combinations_with_replacement(strategies, 2))
medium_latency_combinations = list(itertools.combinations_with_replacement(strategies, 2))
high_latency_combinations = list(itertools.combinations_with_replacement(strategies, 6))

all_indistinguishable_combinations = itertools.product(low_latency_combinations, medium_latency_combinations, high_latency_combinations)

# Flatten the tuples and combine them into a full strategy profile for each combination
indistinguishable_profiles = []
for combination in all_indistinguishable_combinations:
    flattened_profile = list(itertools.chain(*combination)) # Flatten the tuple of tuples
    indistinguishable_profiles.append(flattened_profile)

profile_number = 1

for profile in indistinguishable_profiles:
    print(f"Running simulation with profile {profile_number}: {profile}")
    accumulated_profits = {player_id : 0 for player_id in range(len(profile))}
    for run in range(100) :
        model = Auction(profile, delay=1, rate_public_mean=0.082, rate_public_sd=0, rate_private_mean=0.04,
                    rate_private_sd=0, T_mean=12, T_sd=0)
    # Run the model steps as required
        for i in range(int(model.T * 100)) :
            model.step()

        for player_id, profit in model.player_profits.items() :
            accumulated_profits[player_id] += profit

            # Print or store the accumulated profits for this profile
        print(f"Accumulated profits for profile {profile_number} run {run}: {accumulated_profits}")

    profile_number += 1