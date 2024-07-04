from scipy.io import loadmat
import pandas as pd

data = loadmat("/home/voxar/Desktop/pkb/L2CS-Net-1/metadata.mat")
print(data.keys())

# Assuming your data is stored in a variable named 'data'
# You need to identify the specific key that contains the data of interest
# For example, if the data is in a key named 'my_data', you can access it like this:

my_data = data['recordings']

# Convert the data to a Pandas DataFrame (assuming it's a 2D array)
df = pd.DataFrame(my_data)

# Save the DataFrame to a CSV file
df.to_csv("output.csv", index=False)

# Print the DataFrame to see the contents (optional)
print(df)