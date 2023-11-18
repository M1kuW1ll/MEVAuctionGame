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
        self.bid_count = 0

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

        if len(self.bid_queue) == 1:
            self.bid_count += 1
        elif self.bid_queue[-1] != self.bid_queue[-2]:
            self.bid_count +=1


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
        self.bid_count = 0

    def step(self):

        # Aggregated signal
        self.aggregated_signal = self.model.public_signal_value + self.private_signal_value
        # delta is a small constant value added to the current maximum bid
        delta = 0.0002

        if len(self.model.max_bids) >= 1:
            if self.aggregated_signal - self.pm > self.model.max_bids[-1] + delta:
                self.bid = self.model.max_bids[-1] + delta
            elif self.aggregated_signal - self.pm <= self.model.max_bids[-1] + delta and self.aggregated_signal > self.pm:
                self.bid = self.aggregated_signal - self.pm

        self.bid_queue.append(self.bid)

    def advance(self):
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)

        if len(self.bid_queue) == 1:
            self.bid_count += 1
        elif self.bid_queue[-1] != self.bid_queue[-2]:
            self.bid_count +=1


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
        self.bid_count = 0

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

        if len(self.bid_queue) == 1:
            self.bid_count += 1
        elif self.bid_queue[-1] != self.bid_queue[-2]:
            self.bid_count +=1


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
        self.bid_count = 0

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

        if len(self.bid_queue) == 1:
            self.bid_count += 1
        elif self.bid_queue[-1] != self.bid_queue[-2]:
            self.bid_count +=1


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
        self.bid_count = 0

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

        if len(self.bid_queue) == 1:
            self.bid_count += 1
        elif self.bid_queue[-1] != self.bid_queue[-2]:
            self.bid_count +=1


def get_current_bids(model):
    return model.show_current_bids


def get_bid_agents(model):
    return model.show_bid_agents


def get_public_signal(model):
    return model.public_signal_value


def get_private_signal_max(model):
    return model.private_signal_max


def get_aggregated_signal_max(model):
    return model.aggregated_signal_max


class Auction(Model):
    def __init__(self, N, A, L, S, B, rate_public_mean, rate_public_sd, rate_private_mean, rate_private_sd, T_mean, T_sd, delay) :
        self.num_naive = N
        self.num_adapt = A
        self.num_lastminute = L
        self.num_stealth = S
        self.num_bluff = B
        self.global_delay = delay

        # Initialize bids
        self.max_bids = []
        self.current_bids = []
        self.bid_agents = []
        self.winning_agents = []
        self.show_current_bids = []
        self.show_bid_agents = []

        self.schedule = SimultaneousActivation(self)

        # Initialize signals
        self.public_signal = 0
        self.public_signal_value = 0
        self.private_signal = 0
        self.private_signal_max = 0
        self.aggregated_signal_max = 0
        self.probability_mean = 0
        self.probability_std = 0

        # Initialize winner's info
        self.winner_profit = 0
        self.winner_aggregated_signal = 0
        self.winner_probability = 0


        # Initialize auction time and rate parameters
        self.T = norm.rvs(loc=T_mean, scale=T_sd)

        self.public_lambda = norm.rvs(loc=rate_public_mean, scale=rate_public_sd)

        self.private_lambda = norm.rvs(loc=rate_private_mean, scale=rate_private_sd)

        probabilities = [i * 0.01 + 0.8 for i in range(15)]

        # Create Agents
        for i in range(self.num_naive):
            pm = norm.rvs(loc=0.00659, scale=0.0001)
            delay = i+1
            probability = probabilities[i]
            a = PlayerWithNaiveStrategy(i, self, pm, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_adapt):
            pm = norm.rvs(loc=0.00659, scale=0.0001)
            delay = i+3
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithAdaptiveStrategy(i + self.num_naive, self, pm, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_lastminute):
            pm = norm.rvs(loc=0.00659, scale=0.0001)
            time_reveal_delta = random.randint(10, 20)
            time_estimate = 1200
            delay = i+5
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithLastMinute(i + self.num_naive + self.num_adapt, self, pm,
                                     time_reveal_delta, time_estimate, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_stealth):
            pm = norm.rvs(loc=0.00659, scale=0.0001)
            time_reveal_delta = random.randint(10, 20)
            time_estimate = 1200
            delay = i+7
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithStealthStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute, self, pm,
                                         time_reveal_delta, time_estimate, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_bluff):
            pm = norm.rvs(loc=0.00659, scale=0.0001)
            time_reveal_delta = random.randint(10, 20)
            time_estimate = 1200
            bluff_value = np.random.uniform(0.2, 0.21)
            delay = i+9
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithBluffStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute + self.num_stealth,
                                        self, pm, time_reveal_delta, time_estimate, bluff_value, delay, probability)
            self.schedule.add(a)

        # Initialize data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Current Bids": get_current_bids,
                "Agents": get_bid_agents,
                "Public Signal": get_public_signal,
                "Private Signal Max": get_private_signal_max,
                "Aggregated Signal Max": get_aggregated_signal_max
            },
            agent_reporters={"Bid": "bid",
                             "Probability": "probability",
                             "Bid Count": "bid_count"
                             }
        )

    def step(self):

        # Update public signal
        new_public_signal = poisson.rvs(mu=self.public_lambda)
        self.public_signal += new_public_signal

        for _ in range(new_public_signal) :
            signal_value = np.random.lognormal(mean=-11.66306, sigma=3.05450)
            # Add the value of the current public signal to the total value
            self.public_signal_value += signal_value

        # Update private signal
        new_private_signal = poisson.rvs(mu=self.private_lambda)
        self.private_signal += new_private_signal

        for _ in range (new_private_signal):
            private_signal_value = np.random.lognormal(mean=-8.41975, sigma =1.95231)
            self.private_signal_max += private_signal_value
            for agent in self.schedule.agents:
                if random.random() < agent.probability:
                    agent.private_signal_value += private_signal_value

        self.aggregated_signal_max = self.public_signal_value + self.private_signal_max

        self.schedule.step()

        # Select the winner of the step
        if self.current_bids:
            max_bid = max(self.current_bids)
            self.max_bids.append(max_bid)
            winner_id = self.bid_agents[self.current_bids.index(max_bid)]
            self.winning_agents.append(winner_id)
            for agent in self.schedule.agents:
                if agent.unique_id == winner_id:
                    self.winner_profit = agent.aggregated_signal - max_bid
                    self.winner_aggregated_signal = agent.aggregated_signal
                    self.winner_probability = agent.probability

        self.show_current_bids = self.current_bids.copy()
        self.show_bid_agents = self.bid_agents.copy()

        self.current_bids.clear()
        self.bid_agents.clear()

        # Collect data at the end of the step
        self.datacollector.collect(self)


# Setup and run the model
model = Auction(3, 3, 3, 3, 3, rate_public_mean=0.08, rate_public_sd=0, rate_private_mean=0.04, rate_private_sd=0,
                T_mean=12, T_sd=0, delay=1)

for i in range(int(model.T * 100+1)):
    model.step()

# Data Collection
model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()

# Print the simulation details
print(f"Public signal number: {model.public_signal}")
print(f"Public signal value: {model.public_signal_value}")
print(f"Private signal number: {model.private_signal}")
print(f"Auction time (seconds): {model.T}")
print(f"Auction time steps: {model.schedule.time}")
print("\n")
print(agent_data.loc[i, 'Probability'])
print("\n")
print(agent_data.loc[i, 'Bid Count'])
print("\n")
print(f"Winning Bid of Each Step: {model.max_bids}")
print(f"Winning Agent of Each Step: {model.winning_agents}")


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
print(f"Winner Profit: {model.winner_profit}")
print(f"Winner Total Signal: {model.winner_aggregated_signal}")
print(f"Winner Probility: {model.winner_probability}")
print('Winning bid time: ' + str(time_step) + ' ms')


dfs = []

for i in range(len(model_data)) :
    current_bids = model_data["Current Bids"].iloc[i]
    current_agents = model_data["Agents"].iloc[i]

    bid_dict = {str(agent) : bid for agent, bid in zip(current_agents, current_bids)}

    # Convert the dictionary to a DataFrame and store it in the list
    dfs.append(pd.DataFrame(bid_dict, index=[i]))
all_bids = pd.concat(dfs)
public_signals = model_data["Public Signal"]
private_signal_max = model_data["Private Signal Max"]
aggregated_signal_max = model_data["Aggregated Signal Max"]

# Plot the data
agent_types = ['Naive', 'Naive','Naive', 'Adaptive', 'Adaptive','Adaptive', 'Last-minute', 'Last-minute', 'Last-minute', 'Stealth', 'Stealth','Stealth','Bluff', 'Bluff', 'Bluff']  # Extend this list based on your agents
agent_names = [f"{agent_type} Agent {i+1}" for i, agent_type in enumerate(agent_types)]
all_bids.columns = agent_names


# Plotting using Seaborn for a better appearance
sns.set(style="white", palette="inferno")
all_bids_reset_index = all_bids.reset_index(drop=True)
# Create a figure with a single subplot
f, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plotting the main data
for column in all_bids.columns:
    sns.lineplot(data=all_bids, x=all_bids.index, y=column, ax=ax, label=column, linewidth = 3)

# Overlaying the zoomed area with a secondary axis
axins = ax.inset_axes([0.5, 0.45, 0.35, 0.35])  # X, Y, width, height in the figure proportion
for column in all_bids.columns:
    sns.lineplot(data=all_bids, x=all_bids.index, y=column, ax=axins, linewidth = 2.5)

# Remove the legend from the inset plot
axins.legend_ = None

# Subplot title
axins.set_title('Zoomed in at t = 1200', pad=10, fontsize=16)

# Set the range of the zoomed area
axins.set_xlim(1170, 1201)
axins.set_ylim(0, 0.23)
axins.tick_params(labelsize=16)

ax.axvline(1200, color='k', linestyle='--', linewidth=2)
axins.axvline(1200, color='k', linestyle='--', linewidth=2)
# Set font size of the main legend
ax.legend(fontsize=16)

# Set font sizes of labels and title
ax.set_xlabel('Time (ms)', fontsize=16)
ax.set_ylabel('Bid Value (ETH)', fontsize=16)
ax.title.set_size(16)

# Set font size of tick labels
ax.tick_params(labelsize=16)

# Set the ylim for the main plot
ax.set_xlim(0, 1250)
ax.set_ylim(0, 0.23)
ax.set_yticks(np.arange(0, 0.23, 0.02))

# Bring back the border for the inset plot and remove the connecting lines
axins.set_frame_on(True)



# Display the plot
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

