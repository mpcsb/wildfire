
#%%
import json
import numpy as np 
from shapely.geometry import Point, Polygon

file_path = '/Users/miguel.batista/Documents/projects/test/wildfire/WFIGS_Interagency_Fire_Perimeters.geojson'



def process_wf_data(N=100):
    with open(file_path, 'r') as f:
        data = json.load(f)

    initial_coordinates = []
    polygon_coordinates = []
    dates = []
    areas = []
    distance_threshold = 0.0001 # tolerance in degrees  

    # Iterate through the features
    for i, feature in enumerate(data['features']):
        properties = feature['properties']
        geometry = feature['geometry']
        try:
            # Get initial coordinates and create a shapely Point
            initial_lat = properties['attr_InitialLatitude']
            initial_lon = properties['attr_InitialLongitude']
            initial_point = Point(initial_lon, initial_lat)
            DateCurrent = properties['poly_DateCurrent']
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
                    initial_coordinates.append(np.array([initial_lat, initial_lon]).tolist())
                    polygon_coordinates.append(coords.tolist())
                    dates.append(DateCurrent)
                    areas.append(area_deg_squared)
                    break
        if i > N:
            break
    
    # Convert arrays to lists and write to file 
    output_data = {
        "initial_coordinates": initial_coordinates,
        "polygon_coordinates": polygon_coordinates,
        "dates": dates,
        "area": areas
    }
    with open('output.json', 'w') as f:
        json.dump(output_data, f)
    
    
 
def load_data():
    with open('output.json', 'r') as f:
        data = json.load(f)

    # Convert lists to NumPy arrays or matrices
    initial_coordinates = np.array(data['initial_coordinates'])
    polygon_coordinates = np.array(data['polygon_coordinates'])
    dates = np.array(data['dates'])
    areas = np.array(data['area'])
    return initial_coordinates, polygon_coordinates, dates, areas
