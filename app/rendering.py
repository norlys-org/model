import math
from config import config

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert coordinates from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

def update_points_within_radius(points, center_lat, center_lon, radius, score, status):
    for point in points:
        if haversine(center_lat, center_lon, point['lat'], point['lon']) <= radius:
            point['n'] = point['n'] + 1
            point['score'] = point['score'] + score
            point['status'] = status

def create_matrix(scores):
    lats = [lat / 10 for lat in reversed(range(560, 810, 5))]
    lons = range(1, 40)
    matrix = [{ 
        'lat': lat,
        'lon': lon,
        'n': 0,
        'score': 0,
        'status': 'clear'
    } for lon in lons for lat in lats]

    for key in scores:
        station = config['magnetometres'][key]
        score, status = scores[key]
        update_points_within_radius(matrix, station['lat'], station['lon'], 100, score, status)

    for row in matrix:
        if row['n'] == 0:
            continue
        row['score'] = row['score'] / row['n']

    for row in matrix:
        del row['n']

    return matrix