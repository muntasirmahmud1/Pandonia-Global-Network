import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

############################## Modify the below for a better fit ##################################

# Parameters for peak detection
min_height = 0.005  # adjust based on your data
min_distance = 20
min_width = 3
# Define pixel ranges and assigned wavelengths
wavelength_assignments = [
    (700, 800, 375),
    (900, 1000, 405),
    (1250, 1350, 445),
    (1600, 1700, 488),
    (1950, 2048, 540)
]

window_size = 70  # ±70 pixels around the peak

Dispersion_Polynomial_degree = 1
A2_poly_degree = 1
A3_poly_degree = 1

center_wavelengths = [280, 380, 480]  # in nm

###################################################################################################

######################## Automatically find the latest file and Load data #########################

# Get list of files and sort by modification time
# data_dir = r"C:\Blick\data\tmp"
data_dir = r"/Users/muntasirmahmud/Library/CloudStorage/OneDrive-UMBC/4. SciGlob/Panndora_456/Lab_pandora/25_May_23_Pandora59/"
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
latest_file = max(files, key=os.path.getmtime)
print(f"Data Load: Latest file found: {os.path.basename(latest_file)}")

# Read only the last 2 data rows (after header)
with open(latest_file, 'r', encoding='latin1') as f:
    lines = f.readlines()

# Read last 2 lines only
last_two_lines = lines[-2:]

# Process each line to split into rows and columns
data_list = []
for line in last_two_lines:
    # Strip the newline character and split by tabs
    rows = line.strip().split('\t')
    for row in rows:
        # Split each row by spaces
        columns = row.split()
        data_list.append(columns)

# Convert the list of lists into a DataFrame
data1 = pd.DataFrame(data_list)
# Convert the necessary columns to numeric before saving
data1.iloc[:, 2:] = data1.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

def normalize(data):
    n = (data - min(data)) / (max(data)- min(data))
    return n

#### BC - DC ####
pixels = np.arange(0, 2048)
all_lsf = ((data1.iloc[0, 26:2074]).div(data1.iloc[0, 24], axis=0)).values - ((data1.iloc[1, 26:2074]).div(data1.iloc[1, 24], axis=0)).values
all_lsf = normalize(all_lsf)
int_all_lsf = np.array(data1.iloc[0, 5])
print("Data Load: Done")

plt.figure(figsize=(20, 10))
plt.yscale('log')
plt.xlim(0,2048)
plt.plot(all_lsf, label= f'All lasers, int {int_all_lsf} ms')
plt.xticks(np.arange(0, 2048, 100),fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Pixels',fontsize=16)
plt.ylabel('Normalized Counts',fontsize=16)
plt.title('BC-DC',fontsize=16)
# plt.ylim(0.8e-4,1.5)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################################################################

##################################### Dispersion Polynomial ######################################

# Find peaks in the single LSF signal
peaks, props = find_peaks(all_lsf, height=min_height, distance=min_distance, width=min_width)

assigned_wavelengths = []
for peak in peaks:
    for pix_start, pix_end, wl in wavelength_assignments:
        if pix_start <= peak <= pix_end:
            assigned_wavelengths.append((peak, wl))
            break  # assume one match per peak

# Extract pixel-wavelength pairs
pixels = [p for p, _ in assigned_wavelengths]
wavelengths = [w for _, w in assigned_wavelengths]

# Fit X order polynomial: λ = f(pixel)
disp_poly_coeff = np.polyfit(pixels, wavelengths, Dispersion_Polynomial_degree)
disp_poly = np.poly1d(disp_poly_coeff)
print("Dispersion Polynomial Coefficients:", disp_poly_coeff)

# Plot fit
plt.figure(figsize=(12, 5))
px_vals = np.linspace(0, 2048, 1000)
plt.plot(pixels, wavelengths, 'ro', label='Assigned Wavelengths')
plt.plot(px_vals, disp_poly(px_vals), label='Dispersion Polynomial')
plt.xlabel('Pixel')
plt.ylabel('Wavelength (nm)')
plt.title('Wavelength Calibration')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

############################### Normalized Overlay of Laser Peaks ###############################

plt.figure(figsize=(10, 6))

for px, wl in assigned_wavelengths:
    # Define window
    start = max(0, px - window_size)
    end = min(len(all_lsf), px + window_size + 1)
    
    local_signal = all_lsf[start:end]
    
    # Normalize this peak segment only
    local_signal = (local_signal - np.min(local_signal)) / (np.max(local_signal) - np.min(local_signal))
    
    # Convert pixel indices to wavelength offset
    dispersion_nm_per_pixel = np.polyder(disp_poly)(px)
    x = (np.arange(start, end) - px) * dispersion_nm_per_pixel
    
    plt.plot(x, local_signal, alpha=0.8, label=f'{wl} nm')

plt.xlabel('Wavelength Offset from Peak (nm)')
plt.ylabel('Normalized Intensity')
plt.title('Normalized Overlay of Laser Peaks')
plt.xlim(-5, 5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

##################################################################################################

##################################### Slit function fitting ######################################

def slit_func(x, A2, A3, C1):
    return np.exp(-np.abs(x / A2) ** A3) + C1

A2_list, A3_list, C1_list = [], [], []
peak_wavelengths = []

for px, wl in assigned_wavelengths:
    start = max(0, px - window_size)
    end = min(len(all_lsf), px + window_size + 1)
    
    lsf = all_lsf[start:end]
    lsf_norm = (lsf - np.min(lsf)) / (np.max(lsf) - np.min(lsf))

    dispersion_nm_per_pixel = np.polyder(disp_poly)(px)
    x = (np.arange(start, end) - px) * dispersion_nm_per_pixel

    try:
        popt, _ = curve_fit(
            slit_func, x, lsf_norm,
            bounds=([0.01, 1.5, 0], [5.0, 10.0, 0.1])
        )
        A2_list.append(popt[0])
        A3_list.append(popt[1])
        C1_list.append(popt[2])
        peak_wavelengths.append(wl)
    except RuntimeError:
        print(f"Fit failed for peak at {wl} nm")

# === Fit A2 and A3 polynomials to wavelength ===
wavelengths_um = np.array(peak_wavelengths) / 1000.0

A2_poly = np.polyfit(wavelengths_um, A2_list, A2_poly_degree)
A3_poly = np.polyfit(wavelengths_um, A3_list, A3_poly_degree)
C1_poly = [np.mean(C1_list)]

# === Print final coefficients ===
print("Slit function fitting method -> Symmetric modified Gaussian")
print("A2 polynomial (Width) ->", ' '.join(f"{c:.6e}" for c in A2_poly))
print("A3 polynomial (Shape) ->", ' '.join(f"{c:.6e}" for c in A3_poly))
print("C1 (Constant offset)  ->", f"{C1_poly[0]:.6e}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(wavelengths_um, A2_list, 'ro', label='Measured A2')
plt.plot(wavelengths_um, np.polyval(A2_poly, wavelengths_um), 'b-', label='Fitted A2')
plt.xlabel("Wavelength (μm)")
plt.ylabel("A2 (Width)")
plt.title("A2 vs Wavelength")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(wavelengths_um, A3_list, 'ro', label='Measured A3')
plt.plot(wavelengths_um, np.polyval(A3_poly, wavelengths_um), 'b-', label='Fitted A3')
plt.xlabel("Wavelength (μm)")
plt.ylabel("A3 (Shape)")
plt.title("A3 vs Wavelength")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


def generate_adaptive_x(A2, spacing=0.01):
    half_width = 3 * A2
    num_points = int(2 * half_width / spacing) + 1
    return np.linspace(-half_width, half_width, num_points)

def compute_fwhm(x, y):
    y = y - np.min(y)        # Remove baseline
    y = y / np.max(y)        # Normalize to 1

    half_max = 0.5
    above = np.where(y >= half_max)[0]

    if len(above) < 2:
        return 0.0  # not enough points above half max

    left_idx = above[0]
    right_idx = above[-1]

    # Linear interpolation to get exact crossing points
    def interp_half(i1, i2):
        return x[i1] + (x[i2] - x[i1]) * (half_max - y[i1]) / (y[i2] - y[i1])

    x_left = interp_half(left_idx - 1, left_idx) if left_idx > 0 else x[left_idx]
    x_right = interp_half(right_idx, right_idx + 1) if right_idx < len(x) - 1 else x[right_idx]

    return np.abs(x_right - x_left)


plt.figure(figsize=(10, 6))
for λ0 in center_wavelengths:
    λ0_um = λ0 / 1000.0
    A2 = np.clip(np.polyval(A2_poly, λ0_um), 0.2, 5.0)
    A3 = np.polyval(A3_poly, λ0_um)
    C1 = C1_poly[0]

    x = generate_adaptive_x(A2)
    S = np.exp(-np.abs(x / A2) ** A3) + C1
    # S /= np.sum(S)  # normalize

    fwhm = compute_fwhm(x, S)
    print(f"λ₀ = {λ0} nm → A2 = {A2:.3f}, A3 = {A3:.2f}, FWHM = {fwhm:.4f} nm")

    plt.plot(x, S, label=f'λ₀ = {λ0} nm, FWHM = {fwhm:.3f} nm')

plt.title("Slit Function with FWHM")
plt.xlabel("Wavelength Offset from Center (nm)", fontsize=12)
plt.ylabel("Normalized Intensity", fontsize=12)
plt.xticks(np.arange(-1, 1.1, 0.25), fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
