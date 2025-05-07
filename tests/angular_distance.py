import numpy as np

def calc_angular_distance(latlon1: np.ndarray, latlon2: np.ndarray) -> np.ndarray:
    """Calculate the angular distance between a set of points.

    This function calculates the angular distance in radians
    between any number of latitude and longitude points.

    Parameters
    ----------
    latlon1 : ndarray (n, 2 [lat, lon])
        An array of n (latitude, longitude) points in degrees.

    latlon2 : ndarray (m, 2 [lat, lon])
        An array of m (latitude, longitude) points in degrees.

    Returns
    -------
    ndarray (n, m)
        The array of angular distances in radians between the input arrays.
    """
    lat1 = np.deg2rad(latlon1[:, 0])[:, np.newaxis]
    lon1 = np.deg2rad(latlon1[:, 1])[:, np.newaxis]
    lat2 = np.deg2rad(latlon2[:, 0])[np.newaxis, :]
    lon2 = np.deg2rad(latlon2[:, 1])[np.newaxis, :]

    dlon = lon2 - lon1

    # Clip the argument of arccos to the valid range [-1, 1] to avoid NaN due to precision errors
    # This is important especially for identical or antipodal points.
    argument = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
    argument = np.clip(argument, -1.0, 1.0)

    # theta == angular distance between two points
    theta = np.arccos(argument)
    return theta

# --- Define Selected Test Cases ---

# Test Case 1: Identical and Antipodal Points
test1_latlon1 = np.array([
    [0.0, 0.0],      # Origin
    [90.0, 0.0],     # North Pole
    [45.0, 45.0],    # Point P
])
test1_latlon2 = np.array([
    [0.0, 0.0],      # Identical to Origin
    [-90.0, 0.0],    # South Pole (antipodal to North Pole)
    [-45.0, -135.0], # Antipodal to Point P
    [45.0, 45.0],    # Identical to Point P
])
test1_name = "Identical and Antipodal Points"
test1_notes = "Expect distances (rad): 0 for identical, pi for antipodal."

# Test Case 2: Points on Equator and Meridians
test2_latlon1 = np.array([
    [0.0, 0.0],   # Origin
    [0.0, 0.0],   # Origin
])
test2_latlon2 = np.array([
    [0.0, 90.0],  # 90 deg East on Equator
    [0.0, 180.0], # 180 deg East/West on Equator
    [90.0, 0.0],  # North Pole from Origin
])
test2_name = "Equator and Meridian Distances"
test2_notes = "Expect (rad): pi/2 (90deg lon diff), pi (180deg lon diff), pi/2 (origin to pole)."

# Test Case 3: Multiple Realistic Start Points to Single End Point
test3_latlon1 = np.array([
    [40.71, -74.0],  # Approx NYC
    [34.05, -118.2], # Approx LA
    [51.50, -0.1],   # Approx London
])
test3_latlon2 = np.array([
    [48.85, 2.35]    # Approx Paris
])
test3_name = "Realistic Points (NYC, LA, London -> Paris)"
test3_notes = "Calculate angular distances from 3 cities towards Paris."


# Test Case 4: Grid Calculation with various points
test4_latlon1 = np.array([
    [10.0, 10.0],
    [-20.0, -30.0],
])
test4_latlon2 = np.array([
    [10.0, 11.0],    # Close to first point
    [80.0, 10.0],    # Far north of first point
    [-20.001, -30.001] # Very close to second point
])
test4_name = "Grid Calculation - Various Distances"
test4_notes = "Calculates 2x3 angular distance matrix."


# --- Run Tests and Print Results ---

test_cases = [
    (test1_name, test1_latlon1, test1_latlon2, test1_notes),
    (test2_name, test2_latlon1, test2_latlon2, test2_notes),
    (test3_name, test3_latlon1, test3_latlon2, test3_notes),
    (test4_name, test4_latlon1, test4_latlon2, test4_notes),
]

print("Running Angular Distance Calculations...\n")

for name, latlon1, latlon2, notes in test_cases:
    print(f"--- Test Case: {name} ---")
    print(f"latlon1 (shape {latlon1.shape}):\n{latlon1}")
    print(f"latlon2 (shape {latlon2.shape}):\n{latlon2}")
    print(f"Notes: {notes}")

    try:
        result_rad = calc_angular_distance(latlon1, latlon2)
        result_deg = np.rad2deg(result_rad) # Also show in degrees for easier intuition

        print(f"\nResult (shape {result_rad.shape}):")
        print("Radians:")
        with np.printoptions(precision=4, suppress=True):
             print(result_rad)
        print("\nDegrees (for intuition):")
        with np.printoptions(precision=2, suppress=True):
             print(result_deg)

    except Exception as e:
        print(f"\nError during calculation: {e}")

    print("-" * (len(name) + 25)) # Adjusted separator length
    print()
