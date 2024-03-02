import itertools
import pandas as pd
import numpy as np
import pyspiel
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import utils
import ast

file_path = 'test_payoff_distinguishable.csv'
df = pd.read_csv(file_path)


def parse_accumulated_profits(profits_str) :
    profits_dict = ast.literal_eval(profits_str)
    return np.array(list(profits_dict.values()))


payoff_values = df['Accumulated Profits'].apply(parse_accumulated_profits)
num_agents = 10
strategies_per_agent = 3
shape = [strategies_per_agent] * num_agents
payoff_matrices = [np.zeros(shape=shape, dtype=np.float32) for _ in range(num_agents)]


def decode_profile(profile_str) :
    # Convert the profile string representation into a tuple
    profile = ast.literal_eval(profile_str)
    strategy_indices = [0 if s == 'Naive' else 1 if s == 'Adaptive' else 2 for s in profile]
    return strategy_indices


for index, row in df.iterrows() :
    profile_indices = decode_profile(row['Profile'])
    payoff_values_list = parse_accumulated_profits(row['Accumulated Profits'])

    for agent_idx, payoff in enumerate(payoff_values_list) :
        indices = list(profile_indices)
        payoff_matrices[agent_idx][tuple(indices)] = payoff
        print(payoff_matrices[agent_idx])

strategy_profiles = list(itertools.product(range(strategies_per_agent), repeat=num_agents))
# Compute the AlphaRank scores
alpha = 1
_, _, pi, _, _ = alpharank.compute(payoff_matrices, alpha=alpha, m=50)

ranking = np.argsort(-pi)

print("Ranking of Strategy Profiles:")
for rank, index in enumerate(ranking, start=1) :
    profile = strategy_profiles[index]
    print(f"Rank {rank}: Profile {profile}, Alpha-Rank Score: {pi[index]}")
