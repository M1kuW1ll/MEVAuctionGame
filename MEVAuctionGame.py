#!/usr/bin/env python
# coding: utf-8

# In[19]:


import mesa
import pandas as pd
import numpy as np


# In[20]:


from scipy.stats import poisson, norm


# In[21]:


import random
from mesa import Agent, Model
from mesa.time import RandomActivation


# In[78]:


class Player(Agent):
    def __init__(self, unique_id, model, pm, rate_private):
        super().__init__(unique_id, model)
        self.private_lambda = rate_private
        self.pm = pm
        self.bid = 0
        
    def step(self):
        t = self.model.schedule.time # get current time
        # Private signal 
        private_signal = poisson.rvs(self.private_lambda * t)
        # Aggregated signal
        aggregated_signal = self.model.public_signal + private_signal
        
        if aggregated_signal > self.model.current_max_bid + self.pm:
            self.bid = aggregated_signal
            self.model.current_max_bid = self.bid
            self.model.winning_player = self   
            self.model.num_bid += 1
            
class Auction(Model):
    def __init__(self, N, rate_public, T_mean, T_sd):
        self.num_agents = N
        self.public_lambda = rate_public
        
        #Create Agents
        self.schedule = RandomActivation(self)
        for i in range(self.num_agents):
            pm = np.random.uniform(0, 10.0)
            rate_private = np.random.uniform(0, 1.0)
            a = Player(i, self, pm, rate_private)
            self.schedule.add(a)
        
        # Initialize highest bid
        self.current_max_bid = 0
        self.num_bid = 0
        self.winning_player = None
        
        # Initialize public signal
        self.public_signal = poisson.rvs(self.public_lambda * self.schedule.time)
        
        #Intialize auction time
        self.T = norm.rvs(loc=T_mean, scale=T_sd)
        
    def step(self):
        
        #Update public signal
        self.public_signal = poisson.rvs(self.public_lambda * self.schedule.time)
        
        self.schedule.step()
        

# Setup and run the model
model = Auction(100, rate_public=0.8, T_mean=12, T_sd=0.1)

for i in range(int(model.T*100)): 
    model.step()

# Print winning bidder and highest bid
print(f"Number of Bids: {model.num_bid}")
print(f"Winning bidder: {model.winning_player.unique_id}")
print(f"Highest bid: {model.current_max_bid}")
print(f"Profit margin of winner: {model.winning_player.pm}")
print(f"Public signal: {model.public_signal}")
print(f"Private lambda: {model.winning_player.private_lambda}")
print(f"Auction time: {model.T}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




