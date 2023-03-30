import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_medium(nx, ny, min_val=0.5, max_val=10, sigma=3):
    random_field = np.random.rand(nx, ny) * (max_val - min_val) + min_val
    smooth_field = gaussian_filter(random_field, sigma=sigma)
    return smooth_field

# Define the simulation parameters
nx, ny = 100, 100

# Generate the medium
medium = generate_medium(nx, ny)

# Visualize the medium
fig, ax = plt.subplots()
cax = ax.imshow(medium, cmap='viridis', extent=[0, nx, 0, ny])
plt.colorbar(cax, label="Medium value")
plt.show()