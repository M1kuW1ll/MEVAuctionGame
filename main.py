# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import mesa
import pandas as pd
import numpy as np

from scipy.stats import poisson, norm

import random
from mesa import Agent, Model
from mesa.time import RandomActivation, SimultaneousActivation


class PlayerWithNaiveStrategy(Agent):
    def __init__(self, unique_id, model, pm, rate_private) :
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.private_signal = 0
        self.aggregated_signal = 0

    def step(self):
        t = self.model.schedule.time  # get current time
        # Private signal
        self.private_signal = poisson.rvs(self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal

        if self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm

    def advance(self):
        self.model.current_bids.append(self.bid)

class PlayerWithAdaptiveStrategy(Agent):
    def __init__(self, unique_id, model, pm, rate_private):
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.private_signal = 0
        self.aggregated_signal = 0

    def step(self):
        t = self.model.schedule.time  # get current time
        # Private signal
        self.private_signal = poisson.rvs(self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal
        # delta is a small constant value added to the current maximum bid
        delta = 0.01

        if self.model.max_bids:
            if self.aggregated_signal - self.pm > self.model.max_bids[-1] + delta:
                self.bid = self.model.max_bids[-1] + delta
            elif self.aggregated_signal - self.pm <= self.model.max_bids[-1] + delta and self.aggregated_signal > self.pm:
                self.bid = self.aggregated_signal - self.pm

    def advance(self):
        self.model.current_bids.append(self.bid)


class PlayerWithLastMinute(Agent):
    def __init__(self, unique_id, model, pm, rate_private, time_reveal) :
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.time_reveal = time_reveal
        self.private_signal = 0
        self.aggregated_signal = 0

    def step(self) :
        t = self.model.schedule.time  # get current time
        # Private signal
        self.private_signal = poisson.rvs(self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal

        if t < self.time_reveal:
            self.bid = 0
        elif t >= self.time_reveal and self.aggregated_signal > self.pm:
            self.bid = self.aggregated_signal - self.pm
        else:
            self.bid = 0

    def advance(self):
        self.model.current_bids.append(self.bid)


class PlayerWithStealthStrategy(Agent) :
    def __init__(self, unique_id, model, pm, rate_private, time_reveal) :
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.time_reveal = time_reveal
        self.private_signal = 0
        self.aggregated_signal = 0

    def step(self) :
        t = self.model.schedule.time  # get current time
        # Private signal
        self.private_signal = poisson.rvs(self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal

        if t < self.time_reveal and self.model.public_signal > self.pm :
            self.bid = self.model.public_signal
        elif t >= self.time_reveal and self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm
        else:
            self.bid = 0

    def advance(self):
        self.model.current_bids.append(self.bid)


class PlayerWithBluffStrategy(Agent):
    def __init__(self, unique_id, model, pm, rate_private, time_reveal, bluff_value):
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.time_reveal = time_reveal
        self.bluff_value = bluff_value
        self.private_signal = 0
        self.aggregated_signal = 0

    def step(self) :
        t = self.model.schedule.time  # get current time
        # Private signal
        self.private_signal = poisson.rvs(self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal

        if t < self.time_reveal:
            self.bid = self.bluff_value
        elif t >= self.time_reveal and self.aggregated_signal > self.pm:
            self.bid = self.aggregated_signal - self.pm
        else:
            self.bid = 0

    def advance(self):
        self.model.current_bids.append(self.bid)


class Auction(Model):
    def __init__(self, N, A, L, S, B, rate_public, T_mean, T_sd) :
        self.num_naive = N
        self.num_adapt = A
        self.num_lastminute = L
        self.num_stealth = S
        self.num_bluff = B
        self.public_lambda = rate_public

        self.max_bids = []
        self.current_bids = []


        # Create Agents
        self.schedule = SimultaneousActivation(self)
        for i in range(self.num_naive):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0, 0.02)
            a = PlayerWithNaiveStrategy(i, self, pm, rate_private)
            self.schedule.add(a)

        for i in range(self.num_adapt):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0, 0.02)
            a = PlayerWithAdaptiveStrategy(i + self.num_naive, self, pm, rate_private)
            self.schedule.add(a)

        for i in range(self.num_lastminute):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0, 0.02)
            time_reveal = random.randint(1150, 1190)
            a = PlayerWithLastMinute(i + self.num_naive + self.num_adapt, self, pm, rate_private,
                                     time_reveal)
            self.schedule.add(a)

        for i in range(self.num_stealth):
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0, 0.02)
            time_reveal = random.randint(1170, 1190)
            a = PlayerWithStealthStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute, self, pm,
                                          rate_private, time_reveal)
            self.schedule.add(a)

        for i in range(self.num_bluff) :
            pm = np.random.uniform(0.001, 0.002)
            rate_private = np.random.uniform(0, 0.02)
            time_reveal = random.randint(1170, 1190)
            bluff_value = random.randint(150, 160)
            a = PlayerWithBluffStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute + self.num_stealth,
                                        self, pm, rate_private, time_reveal, bluff_value)
            self.schedule.add(a)

        # Initialize public signal
        self.public_signal = poisson.rvs(self.public_lambda * self.schedule.time)

        # Intialize auction time
        self.T = norm.rvs(loc=T_mean, scale=T_sd)
    def step(self) :
        self.schedule.step()

        # Update public signal
        self.public_signal = poisson.rvs(self.public_lambda * self.schedule.time)

        if self.current_bids:
            self.max_bids.append(max(self.current_bids))

        self.current_bids.clear()


# Setup and run the model
model = Auction(100, 100, 20, 20, 2, rate_public=0.08, T_mean=12, T_sd=0.1)

for i in range(int(model.T * 100)) :
    model.step()

# Print winning bidder and highest bid
highest_bid = max([a.bid for a in model.schedule.agents])
winner = [a.unique_id for a in model.schedule.agents if a.bid == highest_bid]
print(f"Winning bidder: {winner, highest_bid}")
print(f"Public signal: {model.public_signal}")
print(f"Auction time: {model.T}")
print(f"Auction time: {model.schedule.time}")

for a in model.schedule.agents:
    print(f"Agent {a.unique_id} final bid: {a.bid}")
