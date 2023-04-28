import pickle
import numpy as np
import matplotlib.pyplot as plt

from data_load import  process_wf_data, load_data
from terrain import find_points, visualize, get_elevation
from terrain import merge_elevation_data
from terrain import calculate_slope_aspect, visualize_terrain
 
import numpy as np
from scipy.interpolate import griddata

process_wf_data(1000)
initial_coordinates, polygon_coordinates, dates, areas = load_data()

 
for idx, a in enumerate(areas):
    if a < 0.0015 and a > 0.001: 
        print(a, idx)
        break

initial_coords, polygon, date, area = initial_coordinates[idx], polygon_coordinates[idx], dates[idx], areas[idx]
 
points = find_points(polygon, resolution_param=100.0, distance_param=1000.0)
print(len(points)) 
visualize(polygon, points);    

KEY = 'AIzaSyD8UCI6nHZzSyBRqoAGjVDpBNZdlzZnYfc'
points = [(lon, lat) for lat, lon in points]
elevations = get_elevation(points, KEY=KEY)


merged_data = merge_elevation_data(points, elevations)
# write merged_data to pickle
with open(f'merged_data_{idx}.pkl', 'wb') as f: 
    pickle.dump(merged_data, f)

# load merged_data from pickle
with open(f'merged_data_{idx}.pkl', 'rb') as f:
    merged_data = pickle.load(f)

 


# Find unique latitudes and longitudes and calculate the number of rows and columns
unique_lats = np.unique([coord[0] for coord in merged_data])
unique_lons = np.unique([coord[1] for coord in merged_data])
num_rows = len(unique_lats)
num_cols = len(unique_lons)
 

# Create an empty numpy array with the appropriate shape
terrain = np.empty((num_rows, num_cols, 3))

# Generate a grid of latitudes and longitudes
lat_grid, lon_grid = np.meshgrid(unique_lats, unique_lons, indexing='ij')

# Perform interpolation using griddata
points = np.array([(coord[0], coord[1]) for coord in merged_data])
values = np.array([coord[2] for coord in merged_data])
elevation_grid = griddata(points, values, (lat_grid, lon_grid), method='linear')

# Combine the latitude, longitude, and elevation data into the terrain array
terrain[:, :, 0] = lat_grid
terrain[:, :, 1] = lon_grid
terrain[:, :, 2] = elevation_grid

# Apply the same mask as before to the terrain
mask = np.isnan(terrain[:, :, 2])
filtered_terrain = np.ma.masked_array(terrain, mask=np.repeat(mask[:, :, np.newaxis], 3, axis=2))

slope, aspect = calculate_slope_aspect(filtered_terrain)

# Visualize the smoothed terrain surface and the calculated slope and aspect
  
polygon_coords = np.array(polygon)
# Call the visualize_terrain function with the polygon_coords parameter
visualize_terrain(filtered_terrain, slope, aspect, polygon_coords)

# %%


from weather import calculate_wind_gradient

wind_speed = calculate_wind_gradient(filtered_terrain, 5, 120, 0.2)

# wind_spd = calculate_wind_gradient(filtered_terrain, ref_wind_speed, ref_height, alpha)

# Plot the wind speed
plt.imshow(wind_speed, origin='lower')
plt.colorbar(label='Wind Speed')
plt.title('Wind Speed Gradient')
plt.show()