import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter 
from numba import njit
import numpy as np

def generate_medium(nx, ny, min_val=-25, max_val=30, sigma=6):
    random_field = np.random.rand(nx, ny) * (max_val - min_val) + min_val
    smooth_field = gaussian_filter(random_field, sigma=sigma)

    # Apply a nonlinear scaling to increase low values and create a few spikes
    smooth_field = np.where(smooth_field < 0.0, smooth_field * 0, smooth_field)
    smooth_field = np.where( np.logical_and(smooth_field < 2, smooth_field >0), smooth_field * 0.01, smooth_field)
    smooth_field = np.where( np.logical_and(smooth_field < 2.5, smooth_field >2), smooth_field * 0.1, smooth_field) 
    smooth_field = np.where(smooth_field > 2, smooth_field * 2.5, smooth_field)
    #smooth_field = np.where(smooth_field > 0, smooth_field * 0.5, smooth_field)
    return smooth_field


# Define the simulation parameters
nx, ny = 200, 200  # Increase the spatial resolution
domain_size_x, domain_size_y = 200, 200
dx, dy = domain_size_x / nx, domain_size_y / ny 

#smooth_field = generate_medium(nx, ny)
smooth_field = generate_medium(ny, nx)  # Swap nx and ny

max_medium_value = np.max(smooth_field)

c = 0.5  # Constant propagation speed (modify this based on your medium)
print(dx, dy,  min(dx, dy) / (c * max_medium_value) * 0.5)
dt = min(dx, dy) / (c * max_medium_value) * 0.5
 
timesteps = 500
 

# @njit
# def medium(x, y):
#     i, j = int(x), int(y)
#     i = min(i, smooth_field.shape[0] - 1) 
#     j = min(j, smooth_field.shape[1] - 1) 
#     return smooth_field[i, j]
@njit
def medium(x, y):
    j, i = int(x), int(y)  # Swap i and j
    i = min(i, smooth_field.shape[0] - 1) 
    j = min(j, smooth_field.shape[1] - 1) 
    return smooth_field[i, j]

@njit
def medium_2d(X, Y):
    nx, ny = X.shape
    result = np.empty((nx, ny))
    for i in range(nx):
        for j in range(ny):
            result[i, j] = medium(X[i, j], Y[i, j])
    return result

 
def wave(x, y, amp, sigma, px=50, py=50):
    return np.exp(-((x - px)**2 + (y - py)**2) / (sigma**2)) * amp



def generate_wave_speed_field(medium, c):
    wave_speed_field = c * medium
    return wave_speed_field

wave_speed_field = generate_wave_speed_field(smooth_field, c)
 
# Create the x, y coordinates
x = np.arange(0, nx * dx, dx)
y = np.arange(0, ny * dy, dy)
X, Y = np.meshgrid(x, y)  

# Initialize the wave field
A = 3
u = np.zeros((nx, ny, A), dtype=np.float64, order='C')
u[:, :, 0] = wave(X, Y, amp=A, sigma=1)  # Adjust the 'sigma' value as needed

wave_field_data = np.zeros((timesteps, nx, ny))

@njit
def simulation(timesteps, u):
    wave_field_data = np.zeros((timesteps, nx, ny))

    for t in range(timesteps):
        if t%100 == 0:
            print(t)
        m = medium_2d(X[1:-1, 1:-1], Y[1:-1, 1:-1])
        cx = c * m * dt / dx
        cy = c * m * dt / dy

        u[1:-1, 1:-1, 2] = (2 * (1 - cx**2 - cy**2)) * u[1:-1, 1:-1, 1] - u[1:-1, 1:-1, 0] + (cx**2) * (u[2:, 1:-1, 1] + u[:-2, 1:-1, 1]) + (cy**2) * (u[1:-1, 2:, 1] + u[1:-1, :-2, 1])

        u[:, :, 0], u[:, :, 1], u[:, :, 2] = u[:, :, 1], u[:, :, 2], u[:, :, 0]

        wave_field_data[t] = u[:, :, 0]

    return wave_field_data

import time
t0 = time.time()
wave_field_data = simulation(timesteps, u)

def visualize_wave_propagation(wave_field_data):
    fig, ax = plt.subplots() 
    #cax = ax.imshow(wave_field_data[0], vmin=-1, vmax=1, cmap='viridis', extent=[0, nx * dx, 0, ny * dy])
    cax = ax.imshow(wave_field_data[0], vmin=-1, vmax=1, cmap='viridis', extent=[0, nx * dx, 0, ny * dy], origin='lower')

    def update(frame):
        cax.set_data(wave_field_data[frame])
        ax.set_title(f"Timestep: {frame}")

        return [cax]

    ani = animation.FuncAnimation(fig, update, frames=range(timesteps), interval=50, blit=True)
    plt.show()
 

print(time.time() - t0)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(smooth_field, cmap='viridis', extent=[0, nx, 0, ny], origin='lower')
axs[0].set_title("Medium field")
plt.colorbar(axs[0].imshow(smooth_field, cmap='viridis', extent=[0, nx, 0, ny], origin='lower'), ax=axs[0], label="Medium value")

axs[1].imshow(wave_speed_field, cmap='viridis', extent=[0, nx, 0, ny], origin='lower')
axs[1].set_title("Wave speed field")
plt.colorbar(axs[1].imshow(wave_speed_field, cmap='viridis', extent=[0, nx, 0, ny], origin='lower'), ax=axs[1], label="Wave speed")

plt.tight_layout()
plt.savefig('medium.png')

visualize_wave_propagation(wave_field_data)
