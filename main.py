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


class PlayerWithNaiveStrategy(Agent) :
    def __init__(self, unique_id, model, pm, rate_private, delay) :
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.private_signal = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))

    def step(self) :
        t = self.model.schedule.time  # get current time
        # Private signal

        self.private_signal += poisson.rvs(mu=self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal

        if self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm

        self.bid_queue.append(self.bid)

    def advance(self) :
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)


class PlayerWithAdaptiveStrategy(Agent) :
    def __init__(self, unique_id, model, pm, rate_private, delay) :
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.private_signal = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))

    def step(self) :
        t = self.model.schedule.time  # get current time
        # Private signal

        self.private_signal += poisson.rvs(mu=self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal
        # delta is a small constant value added to the current maximum bid
        delta = 0.01

        if len(self.model.max_bids) >= self.model.global_delay + 1 :
            if self.aggregated_signal - self.pm > self.model.max_bids[-1] + delta :
                self.bid = self.model.max_bids[-1] + delta
            elif self.aggregated_signal - self.pm <= self.model.max_bids[
                -1 - self.model.global_delay] + delta and self.aggregated_signal > self.pm :
                self.bid = self.aggregated_signal - self.pm

        self.bid_queue.append(self.bid)

    def advance(self) :
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)


class PlayerWithLastMinute(Agent) :
    def __init__(self, unique_id, model, pm, rate_private, time_reveal_delta, time_estimate, delay) :
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.time_reveal_delta = time_reveal_delta
        self.private_signal = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.time_estimate = time_estimate

    def step(self) :
        t = self.model.schedule.time  # get current time
        # Private signal
        self.private_signal += poisson.rvs(mu=self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal

        time_reveal = self.time_estimate - self.time_reveal_delta - self.individual_delay - self.model.global_delay

        if t < time_reveal :
            self.bid = 0
        elif t >= time_reveal and self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm
        else :
            self.bid = 0

        self.bid_queue.append(self.bid)

    def advance(self) :
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)


class PlayerWithStealthStrategy(Agent) :
    def __init__(self, unique_id, model, pm, rate_private, time_reveal_delta, time_estimate, delay) :
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.time_reveal_delta = time_reveal_delta
        self.private_signal = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.time_estimate = time_estimate

    def step(self) :
        t = self.model.schedule.time  # get current time
        # Private signal

        self.private_signal += poisson.rvs(mu=self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal

        time_reveal = self.time_estimate - self.time_reveal_delta - self.individual_delay - self.model.global_delay

        if t < time_reveal and self.model.public_signal > self.pm :
            self.bid = self.model.public_signal
        elif t >= time_reveal and self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm
        else :
            self.bid = 0

        self.bid_queue.append(self.bid)

    def advance(self) :
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)


class PlayerWithBluffStrategy(Agent) :
    def __init__(self, unique_id, model, pm, rate_private, time_reveal_delta, time_estimate, bluff_value, delay):
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.time_reveal_delta = time_reveal_delta
        self.bluff_value = bluff_value
        self.private_signal = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.time_estimate = time_estimate

    def step(self) :
        t = self.model.schedule.time  # get current time
        # Private signal

        self.private_signal += poisson.rvs(mu=self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal

        time_reveal = self.time_estimate - self.time_reveal_delta - self.individual_delay - self.model.global_delay

        if t < time_reveal :
            self.bid = self.bluff_value
        elif t >= time_reveal and self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm
        else :
            self.bid = 0

        self.bid_queue.append(self.bid)

    def advance(self) :
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)

def get_current_bids(model):
    return model.show_current_bids

def get_bid_agents(model):
    return model.show_bid_agents

class Auction(Model):
    def __init__(self, N, A, L, S, B, rate_public, T_mean, T_sd, delay) :
        self.num_naive = N
        self.num_adapt = A
        self.num_lastminute = L
        self.num_stealth = S
        self.num_bluff = B
        self.public_lambda = rate_public
        self.global_delay = delay
        self.max_bids = []
        self.current_bids = []
        self.bid_agents = []
        self.winning_agents = []
        self.show_current_bids = []
        self.show_bid_agents = []

        self.schedule = SimultaneousActivation(self)
        # Initialize public signal
        self.current_signal = 0
        self.public_signal = 0

        # Intialize auction time
        self.T = norm.rvs(loc=T_mean, scale=T_sd)

        # Create Agents
        for i in range(self.num_naive):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0.01, 0.02)
            delay = random.randint(3, 5)
            a = PlayerWithNaiveStrategy(i, self, pm, rate_private, delay)
            self.schedule.add(a)

        for i in range(self.num_adapt):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0.01, 0.02)
            delay = random.randint(3, 5)
            a = PlayerWithAdaptiveStrategy(i + self.num_naive, self, pm, rate_private, delay)
            self.schedule.add(a)

        for i in range(self.num_lastminute):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0.01, 0.02)
            time_reveal_delta = random.randint(3, 5)
            time_estimate = int(norm.rvs(loc=T_mean, scale=T_sd)*20)
            delay = random.randint(3, 5)
            a = PlayerWithLastMinute(i + self.num_naive + self.num_adapt, self, pm, rate_private,
                                     time_reveal_delta, time_estimate, delay)
            self.schedule.add(a)

        for i in range(self.num_stealth):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0.01, 0.02)
            time_reveal_delta = random.randint(3, 5)
            time_estimate = int(norm.rvs(loc=T_mean, scale=T_sd) * 20)
            delay = random.randint(3, 5)
            a = PlayerWithStealthStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute, self, pm,
                                          rate_private, time_reveal_delta, time_estimate, delay)
            self.schedule.add(a)

        for i in range(self.num_bluff):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0.01, 0.02)
            time_reveal_delta = random.randint(3, 5)
            time_estimate = int(norm.rvs(loc=T_mean, scale=T_sd) * 20)
            bluff_value = random.randint(1500,1600)
            delay = random.randint(3, 5)
            a = PlayerWithBluffStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute + self.num_stealth,
                                        self, pm, rate_private, time_reveal_delta, time_estimate, bluff_value, delay)
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
        self.current_signal = poisson.rvs(mu=self.public_lambda * self.schedule.time)
        self.public_signal += self.current_signal

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
model = Auction(5, 5, 3, 3, 2, rate_public=0.02, T_mean=12, T_sd=0.1, delay=3)

for i in range(int(model.T * 20)):
    model.step()

model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()


print(f"Public signal: {model.public_signal}")
print(f"Auction time: {model.T}")
print(f"Auction time: {model.schedule.time}")

print(model.max_bids)
print(len(model.max_bids))
print(model.winning_agents)
print(len(model.winning_agents))


for time_step in range(int(model.T * 20)):
    current_bids_at_time_step = model_data.loc[time_step, 'Current Bids']
    bid_agents_at_time_step = model_data.loc[time_step, 'Agents']

    df = pd.DataFrame({
        'Bids' : current_bids_at_time_step,
        'Agents' : bid_agents_at_time_step
    })
    print(f"Data for time step {time_step}:")
    print(df)
    print("\n")

# for i in range(int(model.T * 20)):
#     print(f"Data at time step {i}")
#     print(f"Bids placed by each agent")
#     print(agent_data.loc[i, 'Bid'])
#     print("\n")

#
# plt.plot(model.max_bids)
# plt.show()
#
# for time_step in range (int(model.T *20)):

# exploded_data = model_data.explode('Current Bids')
# exploded_data = exploded_data.explode('Agents')
#
# # Let's create a list of colors (you can adjust it according to the number of your agents)
#
# colormap = cm.get_cmap('tab10', 18)
#
# # Assigning colors for each agent
# color_dict = {agent: colormap(i) for i, agent in enumerate(exploded_data['Agents'].unique())}
#
#
# # Create a figure and axis
# fig, ax = plt.subplots()
#
# # Use scatter plot for each bid with color corresponding to agent
# for agent, color in color_dict.items():
#     agent_data = exploded_data[exploded_data['Agents'] == agent]
#     ax.scatter(agent_data.index, agent_data['Current Bids'], color=color, label=agent)
#
# ax.set_xlabel('Time Step')
# ax.set_ylabel('Bid')
# ax.set_title('Bids over Time')
# ax.legend()
# plt.show()

# agent_data_reset = agent_data.reset_index()
#
# agent_data_pivot = agent_data_reset.pivot(index='Step', columns='AgentID', values='Bid')
#
# plt.figure(figsize=(20, 12))
#
# for column in agent_data_pivot.columns:
#     plt.plot(agent_data_pivot[column], markersize = 3, label='Agent ' + str(column))
#
# plt.title('Bids Made by Agent Across All Time Steps')
# plt.xlabel('Time step')
# plt.ylabel('Bid')
# plt.legend(loc='upper left')
# plt.grid(True)
# plt.xticks(np.arange(0, int(model.T*20), 10))  # Set x-axis to be 0, 10, 20, 30... etc
# plt.yticks(np.arange(0, 1500, 100))
# plt.show()


dfs = []

# Loop over each row
for i in range(len(model_data)) :
    # Get current bids and agents from the row
    current_bids = model_data["Current Bids"].iloc[i]
    current_agents = model_data["Agents"].iloc[i]

    # Create a dictionary of bids
    bid_dict = {str(agent) : bid for agent, bid in zip(current_agents, current_bids)}

    # Convert the dictionary to a DataFrame and store it in the list
    dfs.append(pd.DataFrame(bid_dict, index=[i]))

# Concatenate all the dataframes in the list
all_bids = pd.concat(dfs)

# Plot the data
plt.figure(figsize=(20, 12))
for column in all_bids.columns :
    plt.plot(all_bids.index, all_bids[column], label=column)

plt.xlabel('Time Step')
plt.ylabel('Bid Value')
plt.legend(title='Agent ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Bids Received by Relay Across All Time Steps')
plt.grid(True)
plt.xticks(np.arange(0, int(model.T*20), 10))  # Set x-axis to be 0, 10, 20, 30... etc
plt.yticks(np.arange(0, 1500, 100))
plt.show()