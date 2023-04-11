import numpy as np

def create_medium(size_x, size_y, k_min, k_max, threshold):
    np.random.seed(42)
    medium = np.random.uniform(k_min, k_max, (size_x, size_y))
    medium[medium < threshold] = k_min
    medium[medium >= threshold] = k_max
    return medium

def heat_diffusion(u, k, alpha, dt, dx, dy, timesteps):
    for _ in range(timesteps):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (k[1:-1, 1:-1] * alpha * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 + k[1:-1, 1:-1] * alpha * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2)
        u = u_new
    return u

# Parameters
size_x, size_y = 100, 100
k_min, k_max = 0.1, 0.5
threshold = 0.5
alpha = 1.0
dt = 0.01
dx = dy = 1.0
timesteps = 100

# # Create medium
# k = create_medium(size_x, size_y, k_min, k_max, threshold)

# # Initial temperature distribution
# u = np.zeros((size_x, size_y))
# u[size_x // 2, size_y // 2] = 100  # Heat source at the center

# # Heat diffusion
# u_final = heat_diffusion(u, k, alpha, dt, dx, dy, timesteps)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Functions from previous response
# create_medium()
# heat_diffusion()

def update(frame, u, k, alpha, dt, dx, dy):
    u_new = heat_diffusion(u, k, alpha, dt, dx, dy, timesteps=1)
    im.set_array(u_new)
    u[:] = u_new[:]
    return [im]

# Parameters
size_x, size_y = 100, 100
k_min, k_max = 0.1, 0.5
threshold = 0.5
alpha = 50.0
dt = 0.01
dx = dy = 1.0
n_frames = 200

# Create medium
k = create_medium(size_x, size_y, k_min, k_max, threshold)

# Initial temperature distribution
u = np.zeros((size_x, size_y))
u[size_x // 2, size_y // 2] = 50**2  # Heat source at the center

# Set up the plot
fig, ax = plt.subplots()
im = ax.imshow(u, cmap='inferno', vmin=0, vmax=100)
plt.colorbar(im)

# Animate
animation = FuncAnimation(fig, update, frames=n_frames, fargs=(u, k, alpha, dt, dx, dy), interval=50, blit=True)

# Save animation
#animation.save('heat_diffusion_animation.mp4', writer='ffmpeg')

plt.show()
