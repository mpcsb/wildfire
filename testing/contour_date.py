import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

file_path = '/Users/miguel.batista/Documents/projects/test/wildfire/WFIGS_Interagency_Fire_Perimeters.geojson'

with open(file_path, 'r') as f:
    data = json.load(f)


initial_coordinates = []
polygon_coordinates = [] 
distance_threshold = 0.0001

# Iterate through the features
for i, feature in enumerate(data['features']):
    properties = feature['properties']
    geometry = feature['geometry']
    try:
        # Get initial coordinates and create a shapely Point
        initial_lat = properties['attr_InitialLatitude']
        initial_lon = properties['attr_InitialLongitude']
        initial_point = Point(initial_lon, initial_lat)
    except:
        continue

    # Get polygon coordinates and create a shapely Polygon
    for polygon in geometry['coordinates']:
        for coordinates in polygon:
            try:
                coords = np.array(coordinates)
                shapely_polygon = Polygon(coords)
                area_deg_squared = shapely_polygon.area
            except:
                continue

            # Check if the distance between the initial point and the contour is within the threshold
            if initial_point.distance(shapely_polygon) <= distance_threshold:
                initial_coordinates.append(np.array([initial_lat, initial_lon]),)
                plt.scatter(initial_lon, initial_lat, c='red', marker='o', label='Initial Point')
                

                polygon_coordinates.append(coords)
                plt.plot(coords[:, 0], coords[:, 1], 'b-', label='Contour')
                # plt.title(str(properties['poly_GISAcres']), str(area_deg_squared))
                plt.title( str(area_deg_squared))

                plt.show()
                break
    if i > 1000:
        break

 
