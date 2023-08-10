# import matplotlib.pyplot as plt
# from scipy.stats import poisson, lognorm
import random

import numpy as np
#
# def P(lambda_):
#     return poisson.rvs(mu=lambda_)
#
# lambda_ = 0.1
# times = np.arange(0, 240)
#
# random_variates = [P(lambda_) for t in times]
# cumulative_variates = np.cumsum(random_variates)
# plt.figure(figsize=(10, 6))
# plt.plot(times, cumulative_variates)
# plt.plot(times, random_variates)
# plt.xlabel('Time')
# plt.ylabel('E(t)')
# plt.title('Private Lognorm Process')
# plt.grid(True)
# plt.show()
# signal_value = np.random.lognormal(mean=0.00027, sigma=0.0977)
# print(signal_value)


# def E(sigma, mean):
#     return poisson.rvs(sigma, mean)
#
# sigma = 0.5
# mean = 0.1
#


# random_variates = [E(sigma, mean) for t in times]
# cumulative_variates = np.cumsum(random_variates)
# plt.figure(figsize=(10, 6))
# plt.plot(times, cumulative_variates)
# plt.plot(times, random_variates)
# plt.xlabel('Time')
# plt.ylabel('E(t)')
# plt.title('Private Lognorm Process')
# plt.grid(True)
# plt.show()
import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm

# Given mean and standard deviation
target_mean = 0.00027
target_std = 0.097


def loss(params) :
    # Unpack parameters
    mu, sigma = params

    # Compute the mean and standard deviation for the log-normal distribution
    mean = np.exp(mu + sigma ** 2 / 2.0)
    std = ((np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)) ** 0.5

    # Return the squared difference between the target and computed values
    return (mean - target_mean) ** 2 + (std - target_std) ** 2


# Initial guess
init_params = [-5, 0.1]

# Use the minimize function from scipy.optimize
result = minimize(loss, init_params, method='L-BFGS-B', bounds=[(-20, 20), (0, 10)])

# Extract the best parameters
mu_best, sigma_best = result.x

print(f"Best µ: {mu_best:.5f}, Best σ: {sigma_best:.5f}")


