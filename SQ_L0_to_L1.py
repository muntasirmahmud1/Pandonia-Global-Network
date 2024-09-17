import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File path
file_path = '/Users/muntasirmahmud/Library/CloudStorage/OneDrive-UMBC/4. SciGlob/Pandora_2_L0/Pandora2_L0_L1_SQ_20240815.xlsx'

# Load the data
data = pd.read_excel(file_path, header=None)

# Extract L0_raw and L1_raw
L0_raw = data.iloc[0:6, :]
L1_raw = data.iloc[7:12, :]

# Process L0 to L1
L0_ = L0_raw.iloc[:, 5:2053]  # Extract the raw counts for each pixel
L0_blind = L0_raw.iloc[:, 2054:2057].mean(axis=1).values.reshape(-1, 1)  # Mean of last three blind pixels
L0_blind_corrected = L0_ - L0_blind  # Subtract the mean blind pixels
L0_scaled = L0_blind_corrected.divide(L0_raw.iloc[:, 4], axis=0)  # divide by the column = Scale factor for data
L1_calculated = L0_scaled.iloc[:-1, :] - L0_scaled.iloc[-1, :]  # Subtract Dark measurement

# L1 from PGN
L1_ = L1_raw.iloc[:, 5:2053]  # Extract columns 6 to 2053
L1 = L1_.divide(L1_raw.iloc[:, 4], axis=0)  # divide by the column = Scale factor for data

# Select the row to plot
row_to_plot = 0  # Change this value to plot a different row

# Function to normalize
def normalize(data):
    return (data/np.max(data))

# Plotting the normalized values of one row of L1 calculated and L1 together
plt.figure(figsize=(20, 10))
plt.plot(normalize(L1_calculated.iloc[row_to_plot, :]), label='SQ calculated from L0')
plt.plot(normalize(L1.iloc[row_to_plot, :]), label='SQ from L1')
plt.xlabel('Pixel')
plt.ylabel('Normalized Value')
plt.title('Normalized Comparison of L1 calculated and actual L1')
plt.legend()
plt.show()
