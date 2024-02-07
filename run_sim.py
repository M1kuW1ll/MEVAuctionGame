from sim import Auction, PlayerWithNaiveStrategy, PlayerWithAdaptiveStrategy, PlayerWithLastMinute
import itertools
import pyspiel
import pandas as pd
import numpy as np

strategies = ['Naive', 'Adaptive', 'LastMinute']
delays = [1]*2 + [3]*2 + [5]*6
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
data = []
accumulated_profits_list = []

for profile in indistinguishable_profiles:
    print(f"Running simulation with profile {profile_number}: {profile}")
    accumulated_profits = {player_id : 0 for player_id in range(len(profile))}
    for run in range(1) :
        model = Auction(profile, delay=1, rate_public_mean=0.082, rate_public_sd=0, rate_private_mean=0.04,
                    rate_private_sd=0, T_mean=12, T_sd=0)
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

    accumulated_profits_list.append(list(accumulated_profits.values()))
    profile_number += 1

df = pd.DataFrame(data)

# Define the path where you want to save the CSV file
csv_file_path = 'test_payoff.csv'

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print(f"Results saved to {csv_file_path}")

