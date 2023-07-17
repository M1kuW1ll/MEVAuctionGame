import matplotlib.pyplot as plt
from scipy.stats import poisson
import numpy as np


def P(t, lambda_):
    return poisson.rvs(mu=lambda_ * t)

lambda_ = 0.01

times = 240

for t in range (times+1):



# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(times, total_signal)
plt.xlabel('Time')
plt.ylabel('P(t)')
plt.title('Public Poisson Process')
plt.grid(True)
plt.show()

def E(t, lambda_):
    return poisson.rvs(mu=lambda_ * t)

lambda_ = 0.1

times = np.arange(0, 240)

random_variates = [E(t, lambda_) for t in times]

plt.figure(figsize=(10, 6))
plt.plot(times, random_variates)
plt.xlabel('Time')
plt.ylabel('E(t)')
plt.title('Private Poisson Process')
plt.grid(True)
plt.show()