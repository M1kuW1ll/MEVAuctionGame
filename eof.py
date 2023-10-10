import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

all_simulation_results = pd.read_csv('EOF.csv')

probabilities = np.arange(0.8, 0.95, 0.01)
winning_by_probability = []

for probability in probabilities:
    winning = len(all_simulation_results[
        (all_simulation_results['winning_agent'] >= 0) &
        (all_simulation_results['winning_agent'] <= 14) &
        (np.isclose(all_simulation_results['Probability'], probability))  # Using np.isclose because of floating point precision issues
    ])
    winning_by_probability.append(winning)

plt.figure(figsize=(10, 6))
plt.plot(probabilities, winning_by_probability, label='Winning Count', marker='o')
plt.xlabel('Probability')
plt.ylabel('Winning Count')
plt.title('Winning Count at Different Probabilities')
plt.xticks(probabilities, [f"{prob:.2f}" for prob in probabilities])
plt.grid(True)
plt.show()


grouped = all_simulation_results[(all_simulation_results['winning_agent'] >= 0) & (all_simulation_results['winning_agent'] <= 14)].groupby('Probability')
profit_by_probability = [grouped.get_group(prob)['Profit'].values for prob in grouped.groups]

plt.figure(figsize=(10, 6))
# Setting the desired x-axis tick labels
xticks_labels = np.arange(0.8, 0.95, 0.01)

# Plotting the box plot
boxplot = plt.boxplot(profit_by_probability, vert=True, patch_artist=True, whis=100)

for median in boxplot['medians']:
    median.set(color='black', linewidth=2)


# Setting x-axis labels
plt.xticks(range(1, len(xticks_labels) + 1), [f"{tick:.2f}" for tick in xticks_labels])
plt.ylim(0.0063, 0.007)
# Labeling and title
plt.title('Box Plot of Profit Distributions by Probability')
plt.xlabel('Probability')
plt.ylabel('Profit')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Display the plot
plt.show()


profits_by_probability = []

for probability in probabilities:
    profits = all_simulation_results[
        (all_simulation_results['winning_agent'] >= 0) &
        (all_simulation_results['winning_agent'] <= 14) &
        (np.isclose(all_simulation_results['Probability'], probability))  # Using np.isclose because of floating point precision issues
    ]['Profit'].sum()
    profits_by_probability.append(profits)

plt.figure(figsize=(10, 6))

# Create a bar plot for aggregated profits
xticks_positions = range(len(probabilities))
bars = plt.bar(xticks_positions, profits_by_probability, color='purple')

plt.xticks(xticks_positions, [f"{prob:.2f}" for prob in probabilities])
plt.title('Aggregated Profit by Probabilities')
plt.xlabel('Probability')
plt.ylabel('Aggregated Profit')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()
