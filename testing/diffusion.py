import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numba
import warnings

warnings.filterwarnings("ignore")

from time import time
t0=time()

@numba.jit
def reaction_diffusion(u, D, dt, dx, dy):
    u_x = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dx**2
    u_y = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dy**2
    u += dt * D * (u_x + u_y)
    return u

def gaussian(x, y, x0, y0, A, sigma):
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

# Define the size of the domain
width, height = 100, 100

# Define the spatial resolution
res_width, res_height = 300, 300

# Calculate dx and dy based on the domain size and desired resolution
dx = width / res_width
dy = height / res_height

# Define the initial condition
u = np.zeros((res_height, res_width))

# Add initial conditions in each region
num_regions = 10
num_initial_conditions = 5
init_magnitude = 5**10
sigma = 2

for region in range(num_regions):
    for i in range(num_initial_conditions):
        x0 = (res_width // num_regions) * region + res_width // (2 * num_regions)
        y0 = (res_height // num_initial_conditions) * i + res_height // (2 * num_initial_conditions)
        x, y = np.meshgrid(np.arange(res_width), np.arange(res_height))
        u += gaussian(x, y, x0, y0, init_magnitude, sigma)

# Define the medium with 10 vertical regions
D = np.ones((res_height, res_width))

diffusion_values = np.linspace(0, 10, num_regions) 
for region in range(num_regions):
    start_x = (res_width // num_regions) * region
    end_x = (res_width // num_regions) * (region + 1)
    D[:, start_x:end_x] = diffusion_values[region]

# Set the parameters for the simulation
dt = 1 / (2 * (np.max(D) / dx**2 + np.max(D) / dy**2))

# Compute the simulation
frames = 460
u_states = [u.copy()]

for _ in range(frames):
    u = reaction_diffusion(u, D, dt, dx, dy)
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