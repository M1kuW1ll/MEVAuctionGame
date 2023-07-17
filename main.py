# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import mesa
import pandas as pd
import numpy as np

from scipy.stats import poisson, norm
from collections import deque
import matplotlib.pyplot as plt
import random
from mesa import Agent, Model
from mesa.time import RandomActivation, SimultaneousActivation


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

        self.private_signal = poisson.rvs(mu=self.private_lambda * t)
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

        self.private_signal = poisson.rvs(mu=self.private_lambda * t)
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
    def __init__(self, unique_id, model, pm, rate_private, time_reveal, delay) :
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.time_reveal = time_reveal
        self.private_signal = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))

    def step(self) :
        t = self.model.schedule.time  # get current time
        # Private signal

        self.private_signal = poisson.rvs(mu=self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal

        if t < self.time_reveal :
            self.bid = 0
        elif t >= self.time_reveal and self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm
        else :
            self.bid = 0

        self.bid_queue.append(self.bid)

    def advance(self) :
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)


class PlayerWithStealthStrategy(Agent) :
    def __init__(self, unique_id, model, pm, rate_private, time_reveal, delay) :
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.time_reveal = time_reveal
        self.private_signal = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))

    def step(self) :
        t = self.model.schedule.time  # get current time
        # Private signal

        self.private_signal = poisson.rvs(mu=self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal

        if t < self.time_reveal and self.model.public_signal > self.pm :
            self.bid = self.model.public_signal
        elif t >= self.time_reveal and self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm
        else :
            self.bid = 0

        self.bid_queue.append(self.bid)

    def advance(self) :
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)


class PlayerWithBluffStrategy(Agent) :
    def __init__(self, unique_id, model, pm, rate_private, time_reveal, bluff_value, delay) :
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.time_reveal = time_reveal
        self.bluff_value = bluff_value
        self.private_signal = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))

    def step(self) :
        t = self.model.schedule.time  # get current time
        # Private signal

        self.private_signal = poisson.rvs(mu=self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal

        if t < self.time_reveal :
            self.bid = self.bluff_value
        elif t >= self.time_reveal and self.aggregated_signal > self.pm :
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
        self.current_signal = poisson.rvs(mu=self.public_lambda * self.schedule.time)
        self.public_signal = 0

        # Intialize auction time
        self.T = norm.rvs(loc=T_mean, scale=T_sd)

        # Create Agents
        for i in range(self.num_naive):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0.01, 0.02)
            delay = random.randint(1, 3)
            a = PlayerWithNaiveStrategy(i, self, pm, rate_private, delay)
            self.schedule.add(a)

        for i in range(self.num_adapt):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0.01, 0.02)
            delay = random.randint(1, 3)
            a = PlayerWithAdaptiveStrategy(i + self.num_naive, self, pm, rate_private, delay)
            self.schedule.add(a)

        for i in range(self.num_lastminute):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0.01, 0.02)
            time_reveal = random.randint(230, 235)
            delay = random.randint(1, 3)
            a = PlayerWithLastMinute(i + self.num_naive + self.num_adapt, self, pm, rate_private,
                                     time_reveal, delay)
            self.schedule.add(a)

        for i in range(self.num_stealth):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0.01, 0.02)
            time_reveal = random.randint(230, 235)
            delay = random.randint(1, 3)
            a = PlayerWithStealthStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute, self, pm,
                                          rate_private, time_reveal, delay)
            self.schedule.add(a)

        for i in range(self.num_bluff):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0.01, 0.02)
            time_reveal = random.randint(230, 235)
            bluff_value = random.randint(14, 16)
            delay = random.randint(1, 3)
            a = PlayerWithBluffStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute + self.num_stealth,
                                        self, pm, rate_private, time_reveal, bluff_value, delay)
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
model = Auction(10, 10, 5, 5, 2, rate_public=0.02, T_mean=12, T_sd=0.1, delay=1)

for i in range(int(model.T * 20)):
    model.step()

model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()
#
# if model_data.empty:
#     print("The DataFrame is empty.")
# else:
#     print("The DataFrame is not empty.")


print(f"Public signal: {model.public_signal}")
print(f"Auction time: {model.T}")
print(f"Auction time: {model.schedule.time}")

print(model.max_bids)
print(len(model.max_bids))
print(model.winning_agents)
print(len(model.winning_agents))

# for i in range(int(model.T * 20)):
#     print(f"Data at time step {i}")
#     print(f"Bids placed by each agent")
#     print(agent_data.loc[i, 'Bid'])
#     print("\n")

for time_step in range(int(model.T * 20)) :
    current_bids_at_time_step = model_data.loc[time_step, 'Current Bids']
    bid_agents_at_time_step = model_data.loc[time_step, 'Agents']

    df = pd.DataFrame({
        'Current Bids at this step' : current_bids_at_time_step,
        'Agents at this step' : bid_agents_at_time_step
    })
    print(f"Data for time step {time_step}:")
    print(df)
    print("\n")
#
# plt.plot(model.max_bids)
# plt.show()

