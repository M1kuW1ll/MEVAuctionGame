# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import mesa
import pandas as pd
import numpy as np

from scipy.stats import poisson, norm
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from mesa import Agent, Model
from mesa.time import RandomActivation, SimultaneousActivation
import seaborn as sns


class PlayerWithNaiveStrategy(Agent):
    def __init__(self, unique_id, model, pm, delay, probability):
        super().__init__(unique_id, model)
        self.pm = pm
        self.bid = 0
        self.private_signal_value = 0
        self.private_signal = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.probability = probability

    def step(self):

        # Aggregated signal
        self.aggregated_signal = self.model.public_signal_value + self.private_signal_value

        if self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm

        self.bid_queue.append(self.bid)

    def advance(self):
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)


class PlayerWithAdaptiveStrategy(Agent):
    def __init__(self, unique_id, model, pm, delay, probability):
        super().__init__(unique_id, model)
        self.pm = pm
        self.bid = 0
        self.private_signal = 0
        self.private_signal_value = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.probability = probability

    def step(self):

        # Aggregated signal
        self.aggregated_signal = self.model.public_signal_value + self.private_signal_value
        # delta is a small constant value added to the current maximum bid
        delta = 0.01

        if len(self.model.max_bids) >= self.model.global_delay + 1:
            if self.aggregated_signal - self.pm > self.model.max_bids[-1] + delta:
                self.bid = self.model.max_bids[-1] + delta
            elif self.aggregated_signal - self.pm <= self.model.max_bids[
                -1 - self.model.global_delay] + delta and self.aggregated_signal > self.pm:
                self.bid = self.aggregated_signal - self.pm

        self.bid_queue.append(self.bid)

    def advance(self):
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)


class PlayerWithLastMinute(Agent):
    def __init__(self, unique_id, model, pm, time_reveal_delta, time_estimate, delay, probability) :
        super().__init__(unique_id, model)
        self.pm = pm
        self.bid = 0
        self.time_reveal_delta = time_reveal_delta
        self.private_signal = 0
        self.private_signal_value = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.time_estimate = time_estimate
        self.probability = probability

    def step(self):
        t = self.model.schedule.time

        # Aggregated signal
        self.aggregated_signal = self.model.public_signal_value + self.private_signal_value

        time_reveal = self.time_estimate - self.time_reveal_delta - self.individual_delay - self.model.global_delay

        if t < time_reveal:
            self.bid = 0
        elif t >= time_reveal and self.aggregated_signal > self.pm:
            self.bid = self.aggregated_signal - self.pm
        else :
            self.bid = 0

        self.bid_queue.append(self.bid)

    def advance(self):
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay:
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)


class PlayerWithStealthStrategy(Agent):
    def __init__(self, unique_id, model, pm, time_reveal_delta, time_estimate, delay, probability):
        super().__init__(unique_id, model)
        self.pm = pm
        self.bid = 0
        self.time_reveal_delta = time_reveal_delta
        self.private_signal = 0
        self.private_signal_value = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.time_estimate = time_estimate
        self.probability = probability

    def step(self):
        t = self.model.schedule.time  # get current time

        # Aggregated signal
        self.aggregated_signal = self.model.public_signal_value + self.private_signal_value

        time_reveal = self.time_estimate - self.time_reveal_delta - self.individual_delay - self.model.global_delay

        if t < time_reveal and self.model.public_signal_value > self.pm:
            self.bid = self.model.public_signal_value
        elif t >= time_reveal and self.aggregated_signal > self.pm:
            self.bid = self.aggregated_signal - self.pm
        else:
            self.bid = 0

        self.bid_queue.append(self.bid)

    def advance(self):
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay:
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)


class PlayerWithBluffStrategy(Agent):
    def __init__(self, unique_id, model, pm, time_reveal_delta, time_estimate, bluff_value, delay, probability):
        super().__init__(unique_id, model)
        self.pm = pm
        self.bid = 0
        self.time_reveal_delta = time_reveal_delta
        self.bluff_value = bluff_value
        self.private_signal = 0
        self.private_signal_value = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.time_estimate = time_estimate
        self.probability = probability

    def step(self) :
        t = self.model.schedule.time  # get current time

        # Aggregated signal
        self.aggregated_signal = self.model.public_signal_value + self.private_signal_value

        time_reveal = self.time_estimate - self.time_reveal_delta - self.individual_delay - self.model.global_delay

        if t < time_reveal :
            self.bid = self.bluff_value
        elif t >= time_reveal and self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm
        else :
            self.bid = 0

        self.bid_queue.append(self.bid)

    def advance(self):
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)

def get_current_bids(model) :
    return model.show_current_bids

def get_bid_agents(model):
    return model.show_bid_agents

class Auction(Model):
    def __init__(self, N, A, L, S, B, rate_public_mean, rate_public_sd, rate_private_mean, rate_private_sd, T_mean, T_sd, delay) :
        self.num_naive = N
        self.num_adapt = A
        self.num_lastminute = L
        self.num_stealth = S
        self.num_bluff = B
        self.global_delay = delay
        self.max_bids = []
        self.current_bids = []
        self.bid_agents = []
        self.winning_agents = []
        self.show_current_bids = []
        self.show_bid_agents = []

        self.schedule = SimultaneousActivation(self)
        # Initialize public signal
        self.public_signal = 0
        self.public_signal_value = 0
        self.private_signal = 0

        # Intialize auction time
        self.T = norm.rvs(loc=T_mean, scale=T_sd)

        self.public_lambda = norm.rvs(loc=rate_public_mean, scale=rate_public_sd)

        self.private_lambda = norm.rvs(loc=rate_private_mean, scale=rate_private_sd)

        # Create Agents
        for i in range(self.num_naive):
            pm = np.random.uniform(0.001, 0.002)
            delay = random.randint(3, 5)
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithNaiveStrategy(i, self, pm, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_adapt):
            pm = np.random.uniform(0.001, 0.002)
            delay = random.randint(3, 5)
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithAdaptiveStrategy(i + self.num_naive, self, pm, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_lastminute):
            pm = np.random.uniform(0.001, 0.002)
            time_reveal_delta = random.randint(3, 5)
            time_estimate = int(norm.rvs(loc=T_mean, scale=T_sd) * 100)
            delay = random.randint(3, 5)
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithLastMinute(i + self.num_naive + self.num_adapt, self, pm,
                                     time_reveal_delta, time_estimate, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_stealth):
            pm = np.random.uniform(0.001, 0.002)
            time_reveal_delta = random.randint(3, 5)
            time_estimate = int(norm.rvs(loc=T_mean, scale=T_sd) * 100)
            delay = random.randint(3, 5)
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithStealthStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute, self, pm,
                                         time_reveal_delta, time_estimate, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_bluff):
            pm = np.random.uniform(0.001, 0.002)
            time_reveal_delta = random.randint(3, 5)
            time_estimate = int(norm.rvs(loc=T_mean, scale=T_sd) * 100)
            bluff_value = random.randint(400,410)
            delay = random.randint(3, 5)
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithBluffStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute + self.num_stealth,
                                        self, pm, time_reveal_delta, time_estimate, bluff_value, delay, probability)
            self.schedule.add(a)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Current Bids" : get_current_bids,
                "Agents" : get_bid_agents
            },
            agent_reporters={"Bid" : "bid"}
        )

    def step(self):

        # Update public signal
        new_public_signal = poisson.rvs(mu=self.public_lambda)
        self.public_signal += new_public_signal

        for _ in range(new_public_signal) :
            signal_value = np.random.lognormal(mean=0.3, sigma=0.5)
            # Add the value of the current public signal to the total value
            self.public_signal_value += signal_value

        new_private_signal = poisson.rvs(mu=self.private_lambda)
        self.private_signal += new_private_signal

        for agent in self.schedule.agents:
            for _ in range (new_private_signal):
                private_signal_value = np.random.lognormal(mean=0.3, sigma=0.5)
                if random.random() < agent.probability:
                    agent.private_signal_value += private_signal_value


        self.schedule.step()

        if self.current_bids :
            max_bid = max(self.current_bids)
            self.max_bids.append(max_bid)
            winning_agent = self.bid_agents[self.current_bids.index(max_bid)]
            self.winning_agents.append(winning_agent)

        self.show_current_bids = self.current_bids.copy()
        self.show_bid_agents = self.bid_agents.copy()

        self.current_bids.clear()
        self.bid_agents.clear()

        self.datacollector.collect(self)

# Setup and run the model
model = Auction(5, 5, 3, 3, 2, rate_public_mean=0.1, rate_public_sd=0, rate_private_mean=0.1, rate_private_sd=0,
                T_mean=12, T_sd=0.1, delay=3)

for i in range(int(model.T * 100)):
    model.step()
# Data Collection
model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()


print(f"Public signal: {model.public_signal}")
print(f"Public signal Value: {model.public_signal_value}")
print(f"Auction time (seconds): {model.T}")
print(f"Auction time steps: {model.schedule.time}")

print(f"Winning Bid of Each Step: {model.max_bids}")
print(f"Final Winning Bids: {model.max_bids[-1]}")
print(f"Winning Agent of Each Step: {model.winning_agents}")
print(f"Final Winning Agent: {model.winning_agents[-1]}")

# Print table for each step
for time_step in range(int(model.T * 100)):
    current_bids_at_time_step = model_data.loc[time_step, 'Current Bids']
    bid_agents_at_time_step = model_data.loc[time_step, 'Agents']

    df = pd.DataFrame({
        'Bids' : current_bids_at_time_step,
        'Agents' : bid_agents_at_time_step
    })
    print(f"Data for time step {time_step}:")
    print(df)
    print("\n")

print('Winning Agent ID: ' + str(model.winning_agents[-1:][0]))
print('Winning bid value: ' + str(model.max_bids[-1:][0]))
print('Winning bid time: ' + str(time_step) + ' ms')

dfs = []

for i in range(len(model_data)) :
    current_bids = model_data["Current Bids"].iloc[i]
    current_agents = model_data["Agents"].iloc[i]
    bid_dict = {str(agent) : bid for agent, bid in zip(current_agents, current_bids)}

    # Convert the dictionary to a DataFrame and store it in the list
    dfs.append(pd.DataFrame(bid_dict, index=[i]))

all_bids = pd.concat(dfs)

# Plot the data
plt.figure(figsize=(20, 12))
plt.gca().set_prop_cycle('color', plt.cm.inferno(np.linspace(0, 1, len(all_bids.columns))))

fontsize = 12
for column in all_bids.columns:
    plt.plot(all_bids.index, all_bids[column], label=column,linewidth=2)

plt.xlabel('Time Step', fontsize=fontsize)
plt.ylabel('Bid Value', fontsize=fontsize)
plt.legend(title='Agent ID', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsize)
plt.title('Bids Received by Relay Across All Time Steps', fontsize=fontsize)
plt.grid(False)
plt.xticks(np.arange(0, 1300, 100), fontsize=fontsize)
plt.yticks(np.arange(0, 500, 50), fontsize=fontsize)
plt.axvline(1200, color='k')
plt.xlim(0), plt.ylim(-1)
plt.show()


# for i in range(int(model.T * 20)):
#     print(f"Data at time step {i}")
#     print(f"Bids placed by each agent")
#     print(agent_data.loc[i, 'Bid'])
#     print("\n")

#
# plt.plot(model.max_bids)
# plt.show()


# agent_data_reset = agent_data.reset_index()
#
# agent_data_pivot = agent_data_reset.pivot(index='Step', columns='AgentID', values='Bid')
#
# plt.figure(figsize=(20, 12))

# for column in agent_data_pivot.columns:
#     plt.plot(agent_data_pivot[column], markersize = 3, label='Agent ' + str(column))
#
# plt.title('Bids Made by Agent Across All Time Steps')
# plt.xlabel('Time step')
# plt.ylabel('Bid')
# plt.legend(loc='upper left')
# plt.grid(True)
# plt.xticks(np.arange(0, int(model.T*20), 10))  # Set x-axis to be 0, 10, 20, 30... etc
# plt.yticks(np.arange(0, 100, 10))
# plt.show()

