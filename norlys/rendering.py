import math

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

def update_points_within_radius(points, center_lat, center_lon, radius, new_value):
    for point in points:
        lat, lon = point[0]
        if haversine(center_lat, center_lon, lat, lon) <= radius:
            point[1] = new_value

# Example usage
lats = [lat / 10 for lat in reversed(range(560, 810, 5))]
lons = range(1, 40)
points = [[[lat, lon], 0] for lon in lons for lat in lats]

# Given point and value
center_lat = 60  # example latitude
center_lon = 20  # example longitude
radius = 300  # radius in kilometers
new_value = 10  # new value to set

# Update points within the radius
update_points_within_radius(points, center_lat, center_lon, radius, new_value)
