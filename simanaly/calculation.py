
import pandas as pd

# Provided data
data = pd.DataFrame({
    'Probability': list(range(0, 10)),
    'Profit': [2.283153, 2.959528, 3.467853, 4.504326, 4.921320, 5.774697, 7.792085, 9.442124, 10.884404, 13.758760]
})

# Calculate the differences
data['Prob_Diff'] = data['Probability'].diff()
data['Profit_Diff'] = data['Profit'].diff()

# Calculate the exponential growth rate for each pair
data['Growth_Rate'] = (data['Profit'] / data['Profit'].shift(1)) ** (1 / data['Prob_Diff']) - 1

# Calculate the average growth rate
average_growth_rate = data['Growth_Rate'].mean()

print("Average Exponential Growth Rate for 1% Increase in Probability:", average_growth_rate)

data2 = pd.DataFrame({
    'Probability': list(range(0, 10)),
    'Profit': [5.809993, 6.248952, 6.212433, 6.340805, 6.472442, 6.682846, 6.833104, 6.596732, 7.335992, 7.283263]
})

data2['Prob_Diff'] = data2['Probability'].diff()
data2['Profit_Diff'] = data2['Profit'].diff()

# Calculate the exponential growth rate for each pair
data2['Growth_Rate'] = (data2['Profit'] / data2['Profit'].shift(1)) ** (1 / data2['Prob_Diff']) - 1

# Calculate the average growth rate
average_growth_rate = data2['Growth_Rate'].mean()

print("Average Exponential Growth Rate for 1 step decrease in Individual Delay:", average_growth_rate)


data3 = pd.DataFrame({
    'Probability': list(range(0, 10)),
    'Profit': [0.630318, 1.161784, 1.397363, 2.077355, 2.947237, 4.215904, 6.403034, 9.509995, 14.601553, 22.250567]
})

# Calculate the differences
data3['Prob_Diff'] = data3['Probability'].diff()
data3['Profit_Diff'] = data3['Profit'].diff()

# Calculate the exponential growth rate for each pair
data3['Growth_Rate'] = (data3['Profit'] / data3['Profit'].shift(1)) ** (1 / data3['Prob_Diff']) - 1

# Calculate the average growth rate
average_growth_rate = data3['Growth_Rate'].mean()

print("Average Exponential Growth Rate for 1% Increase in Probability:", average_growth_rate)

#
# 0            0.0   0.630318
# 1            1.0   1.161784
# 2            2.0   1.397363
# 3            3.0   2.077355
# 4            4.0   2.947237
# 5            5.0   4.215904
# 6            6.0   6.403034
# 7            7.0   9.509995
# 8            8.0  14.601553
# 9            9.0  22.250567