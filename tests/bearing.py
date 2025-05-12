import numpy as np
from tests.secs import calc_bearing

# Test Case 1: Basic Cardinal Directions from Origin
test1_latlon1 = np.array([[0.0, 0.0]])  # Origin
test1_latlon2 = np.array(
    [
        [1.0, 0.0],  # North
        [0.0, 1.0],  # East
        [-1.0, 0.0],  # South
        [0.0, -1.0],  # West
    ]
)
test1_name = "Basic Cardinal Directions"
test1_notes = "Expect bearings (rad): N(pi/2), E(0), S(3pi/2 or -pi/2), W(pi). Degrees: N(90), E(0), S(270), W(180)."

# Test Case 2: Multiple Realistic Start Points to Single End Point
test2_latlon1 = np.array(
    [
        [40.71, -74.0],  # Approx NYC
        [34.05, -118.2],  # Approx LA
        [51.50, -0.1],  # Approx London
    ]
)
test2_latlon2 = np.array(
    [
        [48.85, 2.35]  # Approx Paris
    ]
)
test2_name = "Realistic Points (NYC, LA, London -> Paris)"
test2_notes = "Calculate bearings from 3 cities towards Paris."

# Test Case 3: Multiple Start to Multiple End (Grid Calculation)
test3_latlon1 = np.array(
    [
        [10.0, 10.0],  # Point A
        [20.0, 20.0],  # Point B
    ]
)
test3_latlon2 = np.array(
    [
        [10.0, 11.0],  # East of A
        [21.0, 20.0],  # North of B
        [15.0, 15.0],  # Between A and B
    ]
)
test3_name = "Multiple Start to Multiple End (Grid)"
test3_notes = "Calculates 2x3 bearing matrix."

# Test Case 4: Edge Cases (Identical, Near Pole, Antimeridian)
test4_latlon1 = np.array(
    [
        [45.0, 45.0],  # Point P
        [89.5, 10.0],  # Near North Pole
        [0.0, 179.5],  # Near Antimeridian
    ]
)
test4_latlon2 = np.array(
    [
        [45.0, 45.0],  # Identical to P
        [-89.5, -170.0],  # Near South Pole (roughly antipodal to Near NP point)
        [0.0, -179.5],  # Across Antimeridian from 3rd point
    ]
)
test4_name = "Edge Cases (Identical, Near Pole, Antimeridian)"
test4_notes = "Tests identical points (expect pi/2 rad or 90 deg), near poles, and crossing +/-180 longitude."

# --- Run Tests and Print Results ---

test_cases = [
    (test1_name, test1_latlon1, test1_latlon2, test1_notes),
    (test2_name, test2_latlon1, test2_latlon2, test2_notes),
    (test3_name, test3_latlon1, test3_latlon2, test3_notes),
    (test4_name, test4_latlon1, test4_latlon2, test4_notes),
]

print("Running Bearing Calculations...\n")

for name, latlon1, latlon2, notes in test_cases:
    print(f"--- Test Case: {name} ---")
    print(f"latlon1 (shape {latlon1.shape}):\n{latlon1}")
    print(f"latlon2 (shape {latlon2.shape}):\n{latlon2}")
    print(f"Notes: {notes}")

    try:
        result_rad = calc_bearing(latlon1, latlon2)
        result_deg = np.rad2deg(result_rad)

        print(f"\nResult (shape {result_rad.shape}):")
        print("Radians:")
        with np.printoptions(precision=4, suppress=True):
            print(result_rad)
        print("\nDegrees:")
        with np.printoptions(precision=2, suppress=True):
            print(result_deg)

    except Exception as e:
        print(f"\nError during calculation: {e}")

    print("-" * (len(name) + 18))
    print()
