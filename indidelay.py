import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

all_simulation_results = pd.read_csv('round4_profitupdate/eof_10indi_0.8eof.csv')

delays = np.arange(1, 11, 1)

winning_agent_counts = all_simulation_results['winning_agent'].value_counts().reset_index()/100
winning_agent_counts.columns = ['winning_agent', 'count']


# Create the barplot using seaborn with the 'inferno' palette
plt.figure(figsize=(15, 9))
bar = sns.barplot(x='winning_agent', y='count', data=winning_agent_counts, palette='inferno')
bar.set_xticklabels([f'{delay*10:.0f}' for delay in delays])
# Set titles and labels
plt.xticks(fontsize = 40)
plt.yticks(fontsize = 40)
plt.xlabel('Individual Delays (ms)', fontsize=40)
plt.ylabel('Win Rate (%)', fontsize=40)
plt.grid(axis='y', linestyle='--', color='grey', linewidth=0.7)
# Show the plot
plt.show()



data = pd.read_csv('round4_profitupdate/eof_10indi_0.8eof.csv')



# Create the boxplot using seaborn with the 'viridis' palette
plt.figure(figsize=(15, 9))
box = sns.boxplot(x='winning_agent', y='Profit', data=data, palette='viridis', showfliers=False)

# Assuming 'delays' is defined as the unique sorted 'winning_agent' values divided by 10
# If 'delays' is not defined in the data, we need to calculate it

# Set custom x-tick labels based on the delays
box.set_xticklabels([f'{int(delay*10)}' for delay in delays])

# Set the title and labels with updated font sizes
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Individual Delay (ms)', fontsize=20)
plt.ylabel('Profit per win ($10^{-3}$ ETH)', fontsize=20)  # Updated unit to ETH

# Add y-axis gridlines in grey color
plt.grid(axis='y', linestyle='--', color='grey', linewidth=0.7)

# Show the plot
plt.show()


# Now that we have the correct column name, we can group by 'winning_agent' instead of 'Agent_ID' and sum the profits.

# Group the data by 'winning_agent' and sum the 'Profit' for each agent
profit_sum_by_agent = data.groupby('winning_agent')['Profit'].sum().reset_index()

# Create the barplot using seaborn with the 'viridis' palette
plt.figure(figsize=(15, 9))
bar2 = sns.barplot(x='winning_agent', y='Profit', data=profit_sum_by_agent, palette='plasma')
bar2.set_xticklabels([f'{delay*10:.0f}' for delay in delays])
# Set the title and labels
plt.xticks(fontsize = 40)
plt.yticks(fontsize = 40)
plt.xlabel('Individual Delay (10ms)', fontsize=40)
plt.ylabel('Average Profit ($10^{-4}$ ETH)', fontsize=40)
plt.grid(axis='y', linestyle='--', color='grey', linewidth=0.7)
# Show the plot
plt.show()

# Calculate the total profit for each agent
total_profit_by_agent = all_simulation_results.groupby('winning_agent')['Profit'].sum().reset_index()

# Print the total profits by winning agent
print("Total Profits by Winning Agent:")
print(total_profit_by_agent)

winning_agent_counts = all_simulation_results['winning_agent'].value_counts().reset_index()/100
print(winning_agent_counts)

