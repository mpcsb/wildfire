import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter 
from numba import njit
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# 1. Create a medium in the same way, but for thermal conductivity
def generate_medium(nx, ny, min_val=0.1, max_val=6, sigma=3):
    random_field = np.random.rand(nx, ny) * (max_val - min_val) + min_val
    smooth_field = gaussian_filter(random_field, sigma=sigma)
    return smooth_field

# Define the simulation parameters
nx, ny = 200, 200
domain_size_x, domain_size_y = 200, 200
dx, dy = domain_size_x / nx, domain_size_y / ny 

# Generate the medium field
thermal_conductivity_field = generate_medium(nx, ny)

# 2. Create the simulation of the heat diffusion equations
alpha = 2 # Thermal diffusivity constant
dt = min(dx, dy)**2 / (4 * alpha)  # Time step size
timesteps = 500

def heat_diffusion(u, k, timesteps, dt, dx, dy, alpha):
    for t in range(timesteps):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (k[1:-1, 1:-1] * alpha * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 + k[1:-1, 1:-1] * alpha * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2)
        u = u_new.copy()
    return u

# Initialize the temperature field
initial_temperature = 25  # Background temperature
heat_source_temperature = 100  # Heat source temperature
heat_source_x, heat_source_y = nx // 2, ny // 2  # Heat source position

temperature_field = np.full((nx, ny), initial_temperature, dtype=np.float64)
temperature_field[heat_source_x, heat_source_y] = heat_source_temperature

# Run the heat diffusion simulation
temperature_data = np.zeros((timesteps, nx, ny))
for t in range(timesteps):
    temperature_field = heat_diffusion(temperature_field, thermal_conductivity_field, 1, dt, dx, dy, alpha)
    temperature_data[t] = temperature_field

# 3. Create an animation in the exact same way as in the code shared
def visualize_heat_diffusion(temperature_data):
    fig, ax = plt.subplots()
    cax = ax.imshow(temperature_data[0], cmap='inferno', extent=[0, nx * dx, 0, ny * dy], origin='lower')

    def update(frame):
        cax.set_data(temperature_data[frame])
        ax.set_title(f"Timestep: {frame}")
        return [cax]

    ani = animation.FuncAnimation(fig, update, frames=range(timesteps), interval=50, blit=True)
    plt.show()

# Visualize the thermal conductivity field
fig, ax = plt.subplots()
ax.imshow(thermal_conductivity_field, cmap='viridis', extent=[0, nx, 0, ny], origin='lower')
ax.set_title("Thermal Conductivity field")
plt.colorbar(ax.imshow(thermal_conductivity_field, cmap='viridis', extent=[0, nx, 0, ny], origin='lower'), ax=ax, label="Thermal conductivity")
plt.savefig('thermal_conductivity.png')

#Visualize the heat diffusion
visualize_heat_diffusion(temperature_data)



  