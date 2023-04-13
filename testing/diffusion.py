import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numba
import warnings
from scipy.ndimage import gaussian_filter
warnings.filterwarnings("ignore") 
from time import time
t0=time() 


@numba.jit
def reaction_diffusion(u, D, dt, dx, dy, fire_threshold, fire_duration, burned):
    u_x = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dx**2
    u_y = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dy**2
    u += dt * D * (u_x + u_y)

    # Find locations that are above the fire_threshold and have not burned before
    on_fire = (u >= fire_threshold) & ~burned

    # Keep track of the fire duration at each location
    fire_counter = np.zeros_like(u)
    fire_counter[on_fire] = fire_duration

    # Update the burned array
    burned[on_fire] = True

    return u, fire_counter, burned


def gaussian(x, y, x0, y0, A, sigma):
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

# Define the size of the domain
width, height = 100, 100

# Define the spatial resolution
res_width, res_height = 500,500

# Calculate dx and dy based on the domain size and desired resolution
dx = width / res_width
dy = height / res_height

# Define the initial condition
u = np.zeros((res_height, res_width))

# Add initial conditions in each region
 
init_magnitude = 5
sigma = 2
 
initial_conditions = [] 
x, y = np.meshgrid(np.arange(res_width), np.arange(res_height))
x0=50; y0=50; 
initial_condition = gaussian(x, y, x0, y0, init_magnitude, sigma)
u += initial_condition
initial_conditions.append(initial_condition)

 


# Define the medium with random values between 0 and N
N = 2
D = np.random.rand(res_height, res_width) * N

# Apply a Gaussian filter to smooth the medium
D = gaussian_filter(D, sigma=3)
# Set a threshold value R
down = 0.95
up = 1.1
# Apply the rule to set values below R to zero
D[D < down] = 0.0
D[D > up] = 1

# Create a boundary of width b_width near the borders
b_width = 2
D[:, :b_width] = 0  # Left border
D[:, -b_width:] = 0  # Right border
D[:b_width, :] = 0  # Top border
D[-b_width:, :] = 0  # Bottom border

# Plot the medium
fig, ax = plt.subplots()
im = ax.imshow(D, cmap='viridis', interpolation='nearest', animated=True)
plt.colorbar(im)
#plt.show()


# Set the parameters for the simulation
dt = 1 / (2 * (np.max(D) / dx**2 + np.max(D) / dy**2))
 
# Set the parameters for the simulation
fire_threshold = 2.1  # Adjust the fire threshold as needed
fire_duration = 10  # Adjust the fire duration as needed
  

# Initialize the burned array
burned = np.zeros((res_height, res_width), dtype=bool)

# Compute the simulation
frames = 500
u_states = [u.copy()]

for frame in range(frames):
    u, fire_counter, burned = reaction_diffusion(u, D, dt, dx, dy, fire_threshold, fire_duration, burned)

    # Update heat sources based on fire_counter
    u[fire_counter > 0] += init_magnitude  # Add heat from burning trees
    fire_counter[fire_counter > 0] -= 1  # Decrease fire duration counter
    if np.max(u) > 10:
        print(frame)

    u_states.append(u.copy()) 

# Set up the plot
fig, ax = plt.subplots()
im = ax.imshow(u_states[0], cmap='viridis', interpolation='nearest', animated=True)
plt.colorbar(im)

def update(frame):
    im.set_array(u_states[frame])
    return im,

print(time()-t0)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=5, blit=True)
print(time()-t0)
plt.show() 
