import pandas as pd
import numpy as np

# Load the primary DataFrame
df1 = pd.read_csv('/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/DMD/BASELINE.csv')

# List of files for comparison
file_list = ['/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/DMD/Cel.csv', '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/DMD/Cel+MG.csv','/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/DMD/DP2.csv', '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/DMD/MESH_02.csv', '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/DMD/MESH_03.csv','/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/DMD/MG.csv', '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/DMD/MG+Cel.csv']  # Replace with your actual file names

# Initialize a counter for removed rows
total_removed_count = 0

# Iterate over each file in the list
for file_name in file_list:
    # Load the current DataFrame
    df2 = pd.read_csv(file_name)

    # Remove rows with -10 in any column from both DataFrames
    df1_filtered = df1[~df1.isin([-10]).any(axis=1)]
    df2_filtered = df2[~df2.isin([-10]).any(axis=1)]

    # Count removed rows for the current file
    removed_count = len(df1) - len(df1_filtered)
    total_removed_count += removed_count

    # Extract the relevant columns from filtered DataFrames
    pitch1 = df1_filtered.iloc[:, 1]
    yaw1 = df1_filtered.iloc[:, 2]

    pitch2 = df2_filtered.iloc[:, 1]
    yaw2 = df2_filtered.iloc[:, 2]

    # Calculate Mean Absolute Error (MAE) for pitch
    mae_pitch = np.mean(np.abs(pitch1 - pitch2))

    # Calculate Mean Absolute Error (MAE) for yaw
    mae_yaw = np.mean(np.abs(yaw1 - yaw2))

    # Round the MAE values to 4 decimal places
    mae_pitch = round(mae_pitch, 4)
    mae_yaw = round(mae_yaw, 4)

    # Print the results for the current file
    print(f"Results for {file_name}:")
    print(f"Mean Absolute Error for Pitch: {mae_pitch}")
    print(f"Mean Absolute Error for Yaw: {mae_yaw}")
    print(f"Number of rows removed: {removed_count}")
    print()

# Print the total number of rows removed across all files
print(f"Total number of rows removed across all files: {total_removed_count}")


