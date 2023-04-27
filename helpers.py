from math import cos, sin, radians

def convert_to_cartesian(point): 
    R = 6371 # approximate radius of earth in km
    lat, lon, alt = point
    lat, lon = radians(lat), radians(lon)
    x = (R + alt) * cos(lat) * cos(lon)
    y = (R + alt) * cos(lat) * sin(lon)
    z = (R + alt) * sin(lat)
    return round(x,1), round(y,1), round(z) 