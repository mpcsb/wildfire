import json

import matplotlib.pyplot as plt
import numpy as np
import requests
from pyproj import Geod
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

geod = Geod(ellps='WGS84')

def find_points(contour_param, resolution_param, distance_param=100.0):
    contour = Polygon(contour_param)
    minx, miny, maxx, maxy = contour.bounds

    # Calculate the number of points in x and y direction based on resolution_param in meters
    x_dist = geod.inv(minx, miny, maxx, miny)[-1]
    y_dist = geod.inv(minx, miny, minx, maxy)[-1]
    x_points = int(x_dist / resolution_param) + 1
    y_points = int(y_dist / resolution_param) + 1

    # Create a grid of points with the specified resolution in meters
    x_range = np.linspace(minx - distance_param/111111.0 , maxx + distance_param/111111.0 , x_points)
    y_range = np.linspace(miny - distance_param/111111.0 , maxy + distance_param/111111.0 , y_points)
    grid = np.meshgrid(x_range, y_range)

    points = [Point(x,y) for x,y in zip(grid[0].flatten(), grid[1].flatten())]

    result = []
    for point in points:
        nearest_point_on_contour = nearest_points(point, contour)[1]
        dist_to_contour_meters = geod.inv(point.x, point.y,
                                          nearest_point_on_contour.x,
                                          nearest_point_on_contour.y)[-1]
        if contour.contains(point) or dist_to_contour_meters <= distance_param:
            result.append((round(point.x,5), round(point.y,5)))

    return result

def visualize(contour_param, points):
    contour = Polygon(contour_param)
    x,y = contour.exterior.xy
    plt.plot(x,y,color='red')

    x,y = zip(*points)
    plt.scatter(x,y,s=2)

    plt.show()

# square_contour = [(40.712776, -74.005974), (40.712776, -73.005974), (41.712776, -73.005974), (41.712776, -74.005974)]
# triangle_contour = [(40.712776, -74.005974), (41.712776, -74.005974), (41.212776, -73.505974)]
# square_result = find_points(square_contour, 1800)
# triangle_result = find_points(triangle_contour, 3000) 
# visualize(square_contour, square_result )
# visualize(triangle_contour, triangle_result )

def get_api_secret():
    with open('keys', 'r') as f:
        for line in f:
            if line.startswith('KEY:'):
                key = line.strip()[4:]
                break
    return key
  

def get_elevation(points):
    KEY = get_api_secret()
    URL = 'https://maps.googleapis.com/maps/api/elevation/json'
    BATCH_SIZE = 512
    result = []
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
        locations = '|'.join([f'{lat},{lon}' for lat, lon in batch])
        params = {'locations': locations, 'key': KEY}
        response = requests.get(URL, params=params)
        data = json.loads(response.text)
        if 'results' in data:
            result.extend([r['elevation'] for r in data['results']])
    return result