import numpy as np

def calc_angular_distance_and_bearing(
    latlon1: np.ndarray, latlon2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the angular distance and bearing between two points.

    This function calculates the angular distance in radians
    between two latitude and longitude points. It also calculates
    the bearing (direction) from point 1 to point 2 going from the
    cartesian x-axis towards the cartesian y-axis.

    Parameters
    ----------
    latlon1 : ndarray of shape (N, 2)
        Array of N (latitude, longitude) points.
    latlon2 : ndarray of shape (M, 2)
        Array of M (latitude, longitude) points.

    Returns
    -------
    theta : ndarray of shape (N, M)
        Angular distance in radians between each point in latlon1 and latlon2.
    alpha : ndarray of shape (N, M)
        Bearing from each point in latlon1 to each point in latlon2.
    """
    latlon1_rad = np.deg2rad(latlon1)
    latlon2_rad = np.deg2rad(latlon2)

    lat1 = latlon1_rad[:, 0][:, None]
    lon1 = latlon1_rad[:, 1][:, None]
    lat2 = latlon2_rad[:, 0][None, :]
    lon2 = latlon2_rad[:, 1][None, :]

    cos_lat1 = np.cos(lat1)
    sin_lat1 = np.sin(lat1)
    cos_lat2 = np.cos(lat2)
    sin_lat2 = np.sin(lat2)

    dlon = lon2 - lon1
    cos_dlon = np.cos(dlon)
    sin_dlon = np.sin(dlon)

    # theta == angular distance between two points
    theta = np.arccos(sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_dlon)

    # alpha == bearing, going from point1 to point2
    #          angle (from cartesian x-axis (By), going towards y-axis (Bx))
    # Used to rotate the SEC coordinate frame into the observation coordinate
    # frame.
    # SEC coordinates are: theta (colatitude (+ away from North Pole)),
    #                      phi (longitude, + east), r (+ out)
    # Obs coordinates are: X (+ north), Y (+ east), Z (+ down)
    x = cos_lat2 * sin_dlon
    y = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_dlon
    alpha = np.pi / 2 - np.arctan2(x, y)

    return theta, alpha

def format_array(arr):
    """Format a numpy array for Rust test case comparison."""
    if arr.ndim == 2:
        return [format_array(row) for row in arr]
    else:
        return [round(float(x), 4) for x in arr]

# Test 1: Basic case - origin to cardinal points
print("\nTest Case 1: Basic cardinal points")
latlon1 = np.array([[0.0, 0.0]])
latlon2 = np.array([[0.0, 90.0], [90.0, 0.0]])
theta, alpha = calc_angular_distance_and_bearing(latlon1, latlon2)
print(f"Theta: {format_array(theta)}")
print(f"Alpha: {format_array(alpha)}")

# Test 2: Realistic coordinates
print("\nTest Case 2: Realistic coordinates")
latlon1 = np.array([[40.71, -74.0], [34.05, -118.2]])
latlon2 = np.array([[48.85, 2.35]])
theta, alpha = calc_angular_distance_and_bearing(latlon1, latlon2)
print(f"Theta: {format_array(theta)}")
print(f"Alpha: {format_array(alpha)}")

# Test 3: Antipodal points
print("\nTest Case 3: Antipodal points")
latlon1 = np.array([[90.0, 0.0]])
latlon2 = np.array([[-90.0, 0.0]])
theta, alpha = calc_angular_distance_and_bearing(latlon1, latlon2)
print(f"Theta: {format_array(theta)}")
print(f"Alpha: {format_array(alpha)}")

# Test 4: Multiple points grid
print("\nTest Case 4: Multiple points grid")
latlon1 = np.array([[10.0, 10.0], [20.0, 20.0]])
latlon2 = np.array([[10.0, 11.0], [21.0, 20.0], [15.0, 15.0]])
theta, alpha = calc_angular_distance_and_bearing(latlon1, latlon2)
print(f"Theta: {format_array(theta)}")
print(f"Alpha: {format_array(alpha)}")

# Test 5: Edge cases with poles and meridian
print("\nTest Case 5: Edge cases with poles and meridian")
latlon1 = np.array([[45.0, 45.0], [89.5, 10.0], [0.0, 179.5]])
latlon2 = np.array([[45.0, 45.0], [-89.5, -170.0], [0.0, -179.5]])
theta, alpha = calc_angular_distance_and_bearing(latlon1, latlon2)
print(f"Theta: {format_array(theta)}")
print(f"Alpha: {format_array(alpha)}")
