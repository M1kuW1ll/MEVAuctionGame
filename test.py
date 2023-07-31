import matplotlib.pyplot as plt
from scipy.stats import poisson, lognorm
import numpy as np

def P(lambda_):
    return poisson.rvs(mu=lambda_)

lambda_ = 0.1
times = np.arange(0, 240)

random_variates = [P(lambda_) for t in times]
cumulative_variates = np.cumsum(random_variates)
plt.figure(figsize=(10, 6))
plt.plot(times, cumulative_variates)
plt.plot(times, random_variates)
plt.xlabel('Time')
plt.ylabel('E(t)')
plt.title('Private Lognorm Process')
plt.grid(True)
plt.show()

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