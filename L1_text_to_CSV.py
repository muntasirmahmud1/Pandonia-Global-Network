import pandas as pd
import os

file_path = '/Users/muntasirmahmud/Library/CloudStorage/OneDrive-UMBC/4. SciGlob/Pandora_2_L0/Pandora2s1_GreenbeltMD_20240725_L1_smca1c9d20220412p1-8.txt'

base_name = os.path.basename(file_path)
csv_file_name = os.path.splitext(base_name)[0] + '.csv'

# Read the entire file
with open(file_path, 'r', encoding='latin1') as file:
    lines = file.readlines()

# Extract the line containing the nominal wavelengths
nominal_wavelengths_line = lines[22].strip()  # Line 23 in the file (index 22)

# Split the line into individual wavelengths and convert them to float
wavelength = [float(value) for value in nominal_wavelengths_line.split(': ')[1].split()]

# Skip the initial 89 rows
data_lines = lines[89:]

# Process each line to split into rows and columns
data_list = []
for line in data_lines:
    # Strip the newline character and split by tabs
    rows = line.strip().split('\t')
    for row in rows:
        # Split each row by spaces
        columns = row.split()
        data_list.append(columns)

# Convert the list of lists into a DataFrame
data1 = pd.DataFrame(data_list)
# Convert the necessary columns to numeric before saving
data1.iloc[:, 2:6205] = data1.iloc[:, 2:6205].apply(pd.to_numeric, errors='coerce')

# Define the column names
column_names = [
    'routine', 'time (yyyymmddThhmmssZ)', 'Fractional days', 'Routine count', 'Repetition count', 'measurement duration (s)',
    'Data processing', 'Integration time [ms]', 'Number of BC cycles', 'Number of DC cycles', 'Saturation index', 'filterwheel 1', 'filterwheel 2',
    'Pointing zenith angle (degree)', 'Zenith pointing mode', 'Pointing azimuth (degree)', 'Azimuth pointing mode', 'Target distance', 'Sum over 2^i',
    'Dark correction method', 'L1 data quality', 'DQ1 limit', 'DQ2 limit', 'Wavelength effective temperature',
    'No of pixels, DC higher than BC', 'No of pixels, DQ1_DC_BC', 'No of pixels, DQ2_DC_BC', 'highest corrected counts',
    'Mean over blind,oversampled pixels in BC', 'Standard error over blind,oversampled pixels in BC', 'Mean over blind,oversampled pixels in DC', 'Standard error over blind,oversampled pixels in DC',
    'Stray light correction', 'Est avg residual SL', 'Est SL before correction at 300nm', 'Est SL before correction at 302.5nm', 'Est SL before correction at 305nm',
    'Est SL before correction at 310nm', 'Est SL before correction at 320nm', 'Est SL before correction at 350nm', 'Est SL before correction at 400nm',
    'Wavelength change fitting result', 'Normalized rms of fitting residuals', 'No of pixels in wavelength change retrieval', 'No of pixels in wavelength change correction',
    'Mean wavelength correction', 'SD of Mean wavelength correction', 'Min wavelength correction', 'Max Mean wavelength correction', 'Expected wavelength shift based on effective temperature',
    'Retrieved wavelength change, order 0', 'Retrieved wavelength change, order 1', 'Temperature at electronics board', 'Spectrometer control temperature',
    'Auxiliary spectrometer temperature', 'Temperature in head sensor (C)', 'Humidity in head sensor (%)', 'Pressure in head sensor (hPa)',
    'Scale factor for data', 'Uncertainty indicator', 'L1 data type',
] + [f'L1 pixel {i}' for i in range(1, 2049)] + [f'Atmospheric variability {i}' for i in range(1, 2049)] + [f'Uncertainty {i}' for i in range(1, 2049)]

# Assign column names to the DataFrame
data1.columns = column_names

save_folder = "/Users/muntasirmahmud/Library/CloudStorage/OneDrive-UMBC/4. SciGlob/Pandora_2_L0"

# Save the DataFrame to a CSV file with the same name as the initial text file
data1.to_csv(os.path.join(save_folder, csv_file_name), index=False)

print(f"Data saved to {os.path.join(save_folder, csv_file_name)}")
