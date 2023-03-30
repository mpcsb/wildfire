import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter 

def generate_medium(nx, ny, min_val=-8, max_val=10, sigma=3):
    random_field = np.random.rand(nx, ny) * (max_val - min_val) + min_val
    smooth_field = gaussian_filter(random_field, sigma=sigma)

    # Apply a nonlinear scaling to increase low values and create a few spikes
    smooth_field = np.where(smooth_field < 0.0, smooth_field * 2, smooth_field)
    smooth_field = np.where( np.logical_and(smooth_field < 1, smooth_field >0), smooth_field * 0.5, smooth_field)
    smooth_field = np.where(smooth_field > 1, smooth_field * 2, smooth_field)

    return smooth_field


# Define the simulation parameters
nx, ny = 200, 200  # Increase the spatial resolution
domain_size_x, domain_size_y = 100, 100
dx, dy = domain_size_x / nx, domain_size_y / ny
dt = 0.1
print(dx, dy, dt)

smooth_field = generate_medium(nx, ny)
max_medium_value = np.max(smooth_field)

c = 0.5  # Constant propagation speed (modify this based on your medium)

#dt = min(dx, dy) / (c * max_medium_value) * 0.5
 
timesteps = 200


# Visualize the medium
fig, ax = plt.subplots()
cax = ax.imshow(smooth_field, cmap='viridis', extent=[0, nx, 0, ny])
plt.colorbar(cax, label="Medium value")
plt.savefig('medium.png')

def medium(x, y):
    i, j = int(x), int(y)
    return smooth_field[i, j]
 
# Define the initial wave
def wave(x, y):
    return np.exp(-((x - 50)**2 + (y - 50)**2) / (2 * 5**2))
 
 

# Create the x, y coordinates
x = np.arange(0, nx * dx, dx)
y = np.arange(0, ny * dy, dy)
X, Y = np.meshgrid(x, y)

# Initialize the wave field
u = np.zeros((nx, ny, 3))
u[:, :, 0] = wave(X, Y)

fig, ax = plt.subplots()
cax = ax.imshow(u[:, :, 0], vmin=-1, vmax=1, cmap='viridis', extent=[0, nx * dx, 0, ny * dy])

def update(frame):
    global u
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            m = medium(x[i], y[j])
            cx, cy = c * m * dt / dx, c * m * dt / dy

            u[i, j, 2] = (2 * (1 - cx**2 - cy**2)) * u[i, j, 1] - u[i, j, 0] + (cx**2) * (u[i + 1, j, 1] + u[i - 1, j, 1]) + (cy**2) * (u[i, j + 1, 1] + u[i, j - 1, 1])

    u[:, :, 0], u[:, :, 1], u[:, :, 2] = u[:, :, 1], u[:, :, 2], u[:, :, 0]
    cax.set_data(u[:, :, 0])

    ax.set_title(f"Timestep: {frame}")  # Add this line to display the timestep as the title

    return [cax]

ani = animation.FuncAnimation(fig, update, frames=range(timesteps), interval=50, blit=True)
plt.show()
