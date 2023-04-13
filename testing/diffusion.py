import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numba
from scipy.ndimage import gaussian_filter
from time import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


# Define the reaction diffusion function
@numba.jit
def reaction_diffusion(u, D, dt, dx, dy, fire_threshold, fire_duration, burned):
    # Calculate the Laplacian
    u_x = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dx ** 2
    u_y = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dy ** 2

    # Update u with reaction and diffusion
    u += dt * D * (u_x + u_y)

    # Find locations that are above the fire_threshold and have not burned before
    on_fire = (u >= fire_threshold) & ~burned

    # Keep track of the fire duration at each location
    fire_counter = np.zeros_like(u)
    fire_counter[on_fire] = fire_duration

    # Update the burned array
    burned[on_fire] = True

    return u, fire_counter, burned


# Define the Gaussian function
def gaussian(x, y, x0, y0, A, sigma):
    return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


# Define simulation parameters
width, height = 100, 100  # Define the size of the domain
res_width, res_height = 400, 400  # Define the spatial resolution
init_magnitude = 500  # Initial magnitude of the Gaussian heat source
sigma = 2  # Standard deviation of the Gaussian heat source
N = 2  # Maximum value of the medium
down = 0.95  # Lower threshold for medium values
up = 1.1  # Upper threshold for medium values
b_width = 2  # Width of the boundary near the borders
fire_threshold = 240  # Threshold value for fire
fire_duration = 1  # Duration of fire
frames = 1000  # Number of frames in animation

# Calculate dx and dy based on the domain size and desired resolution
dx = width / res_width
dy = height / res_height

# Define the initial condition
u = np.zeros((res_height, res_width))

# Add initial conditions in each region
initial_conditions = []
x, y = np.meshgrid(np.arange(res_width), np.arange(res_height))
x0 = 50
y0 = 50
initial_condition = gaussian(x, y, x0, y0, init_magnitude, sigma)
u += initial_condition
initial_conditions.append(initial_condition)

# Define the medium with random values between 0 and N
D = np.random.rand(res_height, res_width) * N

# Apply a Gaussian filter to smooth the medium
D = gaussian_filter(D, sigma=4)

# Apply the rule to set values below R to zero
D[D < 0.98] -= 0.7 
D[D < 0.9] = 0

# Create a boundary of width b_width near the borders
D[:, :b_width] = 0  # Left border
D[:, -b_width:] = 0  # Right border
D[:b_width, :] = 0  # Top border
D[-b_width:, :] = 0  # Bottom border

# Initialize the burned array
burned = np.zeros((res_height, res_width), dtype=bool)

# Compute the simulation
u_states = [u.copy()]

# Set the parameters for the simulation
dt = 1 / (2 * (np.max(D) / dx ** 2 + np.max(D) / dy ** 2))

# Run the simulation
for frame in range(frames):
    u, fire_counter, burned = reaction_diffusion(u, D, dt, dx, dy, fire_threshold, fire_duration, burned)

    # Update heat sources based on fire_counter
    u[fire_counter > 0] += init_magnitude  # Add heat from burning trees
    fire_counter[fire_counter > 0] -= 1  # Decrease fire duration counter
    u_states.append(u.copy())

# Plot the medium
fig1, ax1 = plt.subplots()
im1 = ax1.imshow(D, cmap='viridis', interpolation='nearest', animated=True)
plt.colorbar(im1)

# Set up the plot for the simulation
fig2, ax2 = plt.subplots()
im2 = ax2.imshow(u_states[0], cmap='viridis', interpolation='nearest', animated=True)
plt.colorbar(im2)

# Define the update function for the animation
def update(frame):
    im2.set_array(u_states[frame])
    return im2,

# Create the animation
ani = animation.FuncAnimation(fig2, update, frames=frames, interval=5, blit=True)

plt.show()
