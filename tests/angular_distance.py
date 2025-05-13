import numpy as np
from tests.secs import calc_angular_distance

# Test Case 1: Identical and Antipodal Points
test1_latlon1 = np.array(
    [
        [0.0, 0.0],
        [90.0, 0.0],
        [45.0, 45.0],
    ]
)
test1_latlon2 = np.array(
    [
        [0.0, 0.0],
        [-90.0, 0.0],
        [-45.0, -135.0],
        [45.0, 45.0],
    ]
)
test1_name = "Identical and Antipodal Points"
test1_notes = "Expect distances (rad): 0 for identical, pi for antipodal."

# Test Case 2: Points on Equator and Meridians
test2_latlon1 = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
    ]
)
test2_latlon2 = np.array(
    [
        [0.0, 90.0],
        [0.0, 180.0],
        [90.0, 0.0],
    ]
)
test2_name = "Equator and Meridian Distances"
test2_notes = (
    "Expect (rad): pi/2 (90deg lon diff), pi (180deg lon diff), pi/2 (origin to pole)."
)

# Test Case 3: Multiple Realistic Start Points to Single End Point
test3_latlon1 = np.array(
    [
        [40.71, -74.0],
        [34.05, -118.2],
        [51.50, -0.1],
    ]
)
test3_latlon2 = np.array([[48.85, 2.35]])
test3_name = "Realistic Points (NYC, LA, London -> Paris)"
test3_notes = "Calculate angular distances from 3 cities towards Paris."


test4_latlon1 = np.array(
    [
        [10.0, 10.0],
        [-20.0, -30.0],
    ]
)
test4_latlon2 = np.array([[10.0, 11.0], [80.0, 10.0], [-20.001, -30.001]])
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
        result_deg = np.rad2deg(result_rad)

        print(f"\nResult (shape {result_rad.shape}):")
        print("Radians:")
        with np.printoptions(precision=4, suppress=True):
            print(result_rad)
        print("\nDegrees (for intuition):")
        with np.printoptions(precision=2, suppress=True):
            print(result_deg)

    except Exception as e:
        print(f"\nError during calculation: {e}")

    print("-" * (len(name) + 25))
    print()
