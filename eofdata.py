import pandas as pd
import pyarrow
import gzip

# Step 1: Decompress and read the .csv.gz file into a pandas DataFrame
# List of file names
file_names = [f'{i}.csv.gz' for i in range(10, 24)]

# Initialize an empty list to store the dataframes
dfs = []

# Loop through each file
for file_name in file_names:
    file_path = f'mempooldata/{file_name}'
    with gzip.open(file_path, 'rt') as f:
        # Read the file and append the dataframe to the list
        df = pd.read_csv(f, sep='\t', low_memory=False)
        dfs.append(df)

# Concatenate all dataframes in the list
combined_df = pd.concat(dfs, ignore_index=True)

total_private_tx = combined_df[(combined_df['timepending'] == 0)]
total_private_tx_num = len(total_private_tx)
print("Total Private Transactions:", total_private_tx_num)
confirmed_private_tx = combined_df[(combined_df['timepending'] == 0) & (combined_df['status'] == 'confirmed')]
confirmed_private_tx_num = len(confirmed_private_tx)
print("Confirmed Private Transactions:", confirmed_private_tx_num)
print("Percentage of Private Transactions Confirmed:", (confirmed_private_tx_num/total_private_tx_num)*100)

total_transactions = len(combined_df)
print("Total Transactions:", total_transactions)

total_public_tx = combined_df[(combined_df['timepending'] != 0)]
total_public_tx_num = len(total_public_tx)
print("Total Public Transactions:", total_public_tx_num)

confirmed_public_tx = combined_df[(combined_df['timepending'] != 0) & (combined_df['status'] == 'confirmed')]
confirmed_public_tx_num = len(confirmed_public_tx)
print("Confirmed Public Transactions:", confirmed_public_tx_num)
print("Percentage of Public Transactions Confirmed:", (confirmed_public_tx_num/total_public_tx_num)*100)

min_curblocknumber = combined_df['curblocknumber'].min()
max_curblocknumber = combined_df['curblocknumber'].max()

print("Minimum 'curblocknumber':", min_curblocknumber)
print("Maximum 'curblocknumber':", max_curblocknumber)
print("Block Number:", max_curblocknumber - min_curblocknumber + 1)