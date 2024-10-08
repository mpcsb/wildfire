import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter 
from numba import njit

def generate_medium(nx, ny, min_val=-25, max_val=30, sigma=12):
    random_field = np.random.rand(nx, ny) * (max_val - min_val) + min_val
    smooth_field = gaussian_filter(random_field, sigma=sigma)

    # Apply a nonlinear scaling to increase low values and create a few spikes
    smooth_field = np.where(smooth_field < 0.0, smooth_field * 0, smooth_field)
    smooth_field = np.where( np.logical_and(smooth_field < 2, smooth_field >0), smooth_field * 0.3, smooth_field)
    smooth_field = np.where( np.logical_and(smooth_field < 2.5, smooth_field >2), smooth_field * 0.7, smooth_field) 
    smooth_field = np.where(smooth_field > 3, smooth_field * 2.5, smooth_field)
    return smooth_field


# Define the simulation parameters
nx, ny = 10, 10  # Increase the spatial resolution
domain_size_x, domain_size_y = 200, 200
dx, dy = domain_size_x / nx, domain_size_y / ny 

smooth_field = generate_medium(nx, ny)
max_medium_value = np.max(smooth_field)

c = 11  # Constant propagation speed (modify this based on your medium)
print(dx, dy,  min(dx, dy) / (c * max_medium_value) * 0.5)
dt = min(dx, dy) / (c * max_medium_value) * 0.5
 
timesteps = 500 

# Visualize the medium
fig, ax = plt.subplots()
cax = ax.imshow(smooth_field, cmap='viridis', extent=[0, nx, 0, ny]) 
plt.colorbar(cax, label="Medium value")
plt.savefig('medium.png')

@njit
def medium(x, y):
    i, j = int(x), int(y)
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
 

def wave(x, y, px, py, amp, sigma):
    return np.exp(-((x - px)**2 + (y - py)**2) / (sigma**2)) * amp


# Create the x, y coordinates
x = np.arange(0, nx * dx, dx)
y = np.arange(0, ny * dy, dy)
X, Y = np.meshgrid(x, y) 
 

# Initialize the wave field
A =2
u = np.zeros((nx, ny, A), dtype=np.float64, order='C')
 
sigma= 156
u[:, :, 0] = wave(X, Y, 
                  px=44,#int(domain_size_x/2), 
                  py=33,#int(domain_size_y/2), 
                  amp=A, sigma=sigma)    
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
    cax = ax.imshow(wave_field_data[0], vmin=-1, vmax=1, cmap='viridis', extent=[0, nx * dx, 0, ny * dy])

    def update(frame):
        cax.set_data(wave_field_data[frame])
        ax.set_title(f"Timestep: {frame}")

        return [cax]

    ani = animation.FuncAnimation(fig, update, frames=range(timesteps), interval=50, blit=True)
    plt.show()
 

print(time.time() - t0)
visualize_wave_propagation(wave_field_data)
 