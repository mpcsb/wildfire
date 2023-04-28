import ee
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
ee.Authenticate()
ee.Initialize()


def classify_land_cover(results):
    land_cover = []

    for feature in results:
        coordinates = feature['geometry']['coordinates']
        properties = feature['properties']
        
        # Check if all required properties are present
        if 'B2' not in properties or 'B3' not in properties or 'B4' not in properties or 'B8' not in properties:
            continue  # Skip this point if any property is missing
        
        blue = properties['B2']
        green = properties['B3']
        red = properties['B4']
        nir = properties['B8']

        # Calculate indices
        ndvi = (nir - red) / (nir + red)
        ndwi = (green - nir) / (green + nir)
        bui = (nir + red) - (blue + green)
        bsi = ((red - blue) / (red + blue)) - ((nir - green) / (nir + green))

        # Classify land cover based on indices
        if ndvi > 0.3:
            cover_type = "Vegetation"
        elif ndwi > 0.3:
            cover_type = "Water"
        elif bui > 0:
            cover_type = "Roads/Buildings"
        elif bsi > 0:
            cover_type = "Bare Soil/Rocks"
        else:
            cover_type = "Unknown"

        land_cover.append((coordinates, cover_type))

    return land_cover

 
def original_terrain_cover_type(central_point, buffer_distance=500, width=1000, height=2000):
    center_point = ee.Geometry.Point(central_point)

    # Create a rectangle around the center point
    rectangle = center_point.buffer(buffer_distance).bounds()

    # Calculate the number of points in the grid (resolution: 100 meters)
    grid_resolution = 100
    num_points_x = int(width / grid_resolution)
    num_points_y = int(height / grid_resolution)

    # Create a grid of points within the rectangle
    points = ee.FeatureCollection.randomPoints(rectangle, num_points_x * num_points_y)

    # Filtering Sentinel-2 collection
    sentinel2_collection = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterDate('2020-01-01', '2020-12-31') \
        .filterBounds(rectangle) \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 0.1)

    # Select the bands and compute the median
    sentinel2_median = sentinel2_collection.select(['B2', 'B3', 'B4', 'B8']).median()

    # Sample the data at the points
    terrain_data = sentinel2_median.sampleRegions(collection=points, geometries=True, scale=10)  # scale set to 10 meters to match Sentinel-2 resolution

    # Retrieve the results as a list of dictionaries
    results = terrain_data.getInfo()['features']

    classified_land_cover = classify_land_cover(results)

    # Define a dictionary to map land cover types to colors
    land_cover_colors = {
        "Water": "blue",
        "Unknown": "gray",
        "Roads/Buildings": "black",
        "Vegetation": "green",
        'Bare Soil/Rocks': "brown"
    }

    # Initialize lists to store latitude, longitude, and colors
    lats = []
    lons = []
    colors = []

    # Populate the lists with the data from classified_land_cover
    for coords, cover_type in classified_land_cover:
        lat, lon = coords[::-1]
        lats.append(lat)
        lons.append(lon)
        colors.append(land_cover_colors.get(cover_type, "red"))  # Use red for unknown cover types


    return classified_land_cover

 
def process_multiple_points(central_points, buffer_distance=500, width=2000, height=2000):
    combined_land_cover = [] 

    for point in tqdm(central_points, desc='Processing Points'):
        classified_land_cover = original_terrain_cover_type(point, 
                                                            buffer_distance=buffer_distance, 
                                                            width=width, 
                                                            height=height)
        combined_land_cover.extend(classified_land_cover)
    
    return combined_land_cover

#%%

# Define the starting point (bottom-left corner of the grid)
start_point = [-123.5987, 40.90715]
start_point = [-124.060543,41.492324]
import math

start_point = [-124.060543, 41.492324]
 

# Define the grid size (number of rows and columns)
n = 15
m = 15
 

spacing_lon = 0.015
spacing_lat = 0.013

# Generate an nxm grid of central points
central_points = []
for i in range(n):
    for j in range(m):
        lon = start_point[0] + (j - (m // 2)) * spacing_lon
        lat = start_point[1] + (i - (n // 2)) * spacing_lat
        central_points.append([lon, lat])


# Process and combine the land cover data from all central points
combined_land_cover = process_multiple_points(central_points, 
                                              buffer_distance=500, 
                                              width=1000, 
                                              height=1000)




def plot(classified_land_cover): 
    import matplotlib.ticker as ticker

    # Define a dictionary to map land cover types to colors
    land_cover_colors = {
        "Water": "blue",
        "Unknown": "gray",
        "Roads/Buildings": "black",
        "Vegetation": "green",
        'Bare Soil/Rocks': "brown"
    }

    # Initialize lists to store latitude, longitude, and colors
    lats = []
    lons = []
    colors = []

    # Populate the lists with the data from classified_land_cover
    for coords, cover_type in classified_land_cover:
        lat, lon = coords[::-1]
        lats.append(lat)
        lons.append(lon)
        colors.append(land_cover_colors.get(cover_type, "red"))  # Use red for unknown cover types

    # Create a scatter plot using the latitude, longitude, and colors
    fig, ax = plt.subplots()
    ax.scatter(lons, lats, c=colors, alpha=0.5)

    # Format xticks to display the values in degrees instead of scientific notation
    formatter = ticker.FormatStrFormatter('%0.4f')
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', rotation=45)

    # Set the title, axis labels, and display the plot
    ax.set_title("Land Cover")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude")  

# Plot the combined land cover data
plot(combined_land_cover)
plt.scatter([start_point[0]], [start_point[1]], c='red', s=100)

set([cover_type for _, cover_type in combined_land_cover])

# %%
