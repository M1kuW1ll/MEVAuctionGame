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
from collections import deque

class PlayerWithNaiveStrategy(Agent):
    def __init__(self, unique_id, model, pm, rate_private, delay):
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        self.private_signal = 0
        self.aggregated_signal = 0
        self.delay = delay
        self.bid_queue = deque(maxlen=delay)

    def step(self):
        t = self.model.schedule.time  # get current time
        # Private signal
        self.private_signal = poisson.rvs(self.private_lambda * t)
        # Aggregated signal
        self.aggregated_signal = self.model.public_signal + self.private_signal
        self.bid_queue.append(self.bid)

        if self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm

    def advance(self):
        if len(self.bid_queue) == self.delay:
            self.model.current_bids.append(self.bid_queue.popleft())


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
        self.bid_
        if self.aggregated_signal - self.pm > self.model.current_max_bid + delta:
            self.bid = self.model.current_max_bid + delta
            self.model.current_max_bid = self.bid
        else:
            self.bid = self.aggregated_signal - self.pm
            self.model.num_bid += 1
            if self.bid > self.model.current_max_bid :
                self.model.current_max_bid = self.bid
                self.model.winning_player = self


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

        if t >= self.time_reveal and self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm
            self.model.num_bid += 1
            if self.bid > self.model.current_max_bid :
                self.model.current_max_bid = self.bid
                self.model.winning_player = self


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
            self.model.num_bid += 1
            if self.bid > self.model.current_max_bid :
                self.model.current_max_bid = self.bid
                self.model.winning_player = self
        elif t >= self.time_reveal and self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm
            self.model.num_bid += 1
            if self.bid > self.model.current_max_bid :
                self.model.current_max_bid = self.bid
                self.model.winning_player = self


# class PlayerWithBluffStrategy(Agent):
#     def __init__(self, unique_id, model, pm, rate_private, time_reveal, bluff_value):
#         super().__init__(unique_id, model)
#         self.private_lambda = rate_private
#         self.pm = pm
#         self.bid = 0
#         self.time_reveal = time_reveal
#         self.bluff_value = bluff_value
#         self.private_signal = 0
#         self.aggregated_signal = 0
#
#     def step(self) :
#         t = self.model.schedule.time  # get current time
#         # Private signal
#         self.private_signal = poisson.rvs(self.private_lambda * t)
#         # Aggregated signal
#         self.aggregated_signal = self.model.public_signal + self.private_signal
#
#         if t < self.time_reveal:
#             self.bid = self.bluff_value
#             self.model.num_bid += 1
#             if self.bid > self.model.current_max_bid :
#                 self.model.current_max_bid = self.bid
#                 self.model.winning_player = self
#         elif t >= self.time_reveal and self.aggregated_signal > self.pm:
#             self.bid = self.aggregated_signal - self.pm
#             self.model.num_bid += 1
#             if self.bid > self.model.current_max_bid:
#                 self.model.current_max_bid = self.bid
#                 self.model.winning_player = self


class Auction(Model):
    def __init__(self, N, A, L, S, rate_public, T_mean, T_sd, global_delay) :
        self.num_naive = N
        self.num_adapt = A
        self.num_lastminute = L
        self.num_stealth = S

        self.public_lambda = rate_public

        # Initialize public signal
        self.public_signal = poisson.rvs(self.public_lambda * self.schedule.time)

        # Intialize auction time
        self.T = norm.rvs(loc=T_mean, scale=T_sd)

        self.global_delay = global_delay

        self.current_max_bid = 0
        self.current_bids = []
        self.bid_history = deque(maxlen = global_delay)


        # Create Agents
        self.schedule = RandomActivation(self)
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
            time_reveal = random.randint(1150, 1190)
            a = PlayerWithStealthStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute, self, pm,
                                          rate_private, time_reveal)
            self.schedule.add(a)

        # for i in range(self.num_bluff) :
        #     pm = np.random.uniform(5, 10)
        #     rate_private = np.random.uniform(0, 1.0)
        #     time_reveal = random.randint(1150, 1190)
        #     bluff_value = random.randint(2300, 2400)
        #     a = PlayerWithBluffStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute + self.num_stealth,
        #                                 self, pm, rate_private, time_reveal, bluff_value)
        #     self.schedule.add(a)


    def step(self) :

        # Update public signal
        self.public_signal = poisson.rvs(self.public_lambda * self.schedule.time)

        self.schedule.step()

        self.bid_history.append(self.current_max_bid)

        if len(self.bid_history) == self.global_delay:
            self.current_max_bid = max(self.bid_history.popleft(), *self.current_bids)

        self.current_bids.clear()


# Setup and run the model
model = Auction(100, 100, 20, 20, rate_public=0.08, T_mean=12, T_sd=0.1)

for i in range(int(model.T * 100)) :
    model.step()

# Print winning bidder and highest bid
print(f"Number of Bids: {model.num_bid}")
print(f"Winning bidder: {model.winning_player.unique_id, model.winning_player.aggregated_signal, model.winning_player.bid}")
print(f"Highest bid: {model.current_max_bid}")
print(f"Profit margin of winner: {model.winning_player.pm}")
print(f"Public signal: {model.public_signal}")
print(f"Private lambda: {model.winning_player.private_lambda, model.winning_player.private_signal}")
print(f"Auction time: {model.T}")
print(f"Auction time: {model.schedule.time}")
