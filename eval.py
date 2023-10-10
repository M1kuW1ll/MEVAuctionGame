import random
import pandas as pd

from main import Auction


def generate_strategies(fixed_strategy=None, manual_values=None):
    if manual_values:
        return manual_values

    strategies = {'N': 0, 'A': 0, 'L': 0, 'S': 0, 'B': 0}
    if fixed_strategy:
        strategies[fixed_strategy] = 1
        remaining_value = 9
    else:
        remaining_value = 10

    for key in random.sample([k for k in strategies.keys() if k != fixed_strategy], len(strategies) - 1):
        value = random.randint(0, remaining_value)
        strategies[key] = value
        remaining_value -= value

    return strategies

def run_simulation(strategies, delay, num_simulations):
    sim_results = pd.DataFrame(columns=['winning_agent', 'winning_bid_value', 'Winner Aggregated Signal','Profit', 'Probability', 'winning_bid_time', 'N', 'A', 'L', 'S', 'B', 'Delay'])  # Add 'Aggregated Signal Max' to columns
    for _ in range(num_simulations):
        N, A, L, S, B = strategies.values()
        model = Auction(N, A, L, S, B, rate_public_mean=0.085, rate_public_sd=0, rate_private_mean=0.04, rate_private_sd=0,
                        T_mean=12, T_sd=0.1, delay=delay)
        for i in range(int(model.T * 100)):
            model.step()
        time_step = int(model.T * 100) - 1
        sim_results.loc[len(sim_results)] = [int(model.winning_agents[-1:][0]), model.max_bids[-1:][0],model.winner_aggregated_signal,
                                             model.winner_profit, model.winner_probability, time_step, N, A, L, S, B, delay]  # Add 'model.aggregated_signal_max' to values

    return sim_results

manual_values = {'N': 15, 'A': 0, 'L': 0, 'S': 0, 'B': 0}
num_simulations = 10000
num_runs = 1
all_results = pd.DataFrame(columns=['Fixed Strategy', 'Chances of Winning', 'Mean Winning Bid Value', 'Delay'])

for run in range(num_runs):
    all_sim_results = []
    for fixed_strategy in [None]:
        print('Run ' + str(run))
        delay = 1
        print('Delay ' + str(delay))
        strategies = generate_strategies(fixed_strategy, manual_values)
        sim_results = run_simulation(strategies, delay, num_simulations)
        all_sim_results.append(sim_results)

    concatenated_sim_results = pd.concat(all_sim_results, ignore_index=True)
    filename = f'EOF.csv'
    concatenated_sim_results.to_csv(filename, index=False)
    print(f'Simulation results saved to {filename}')