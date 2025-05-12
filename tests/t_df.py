import numpy as np
from secs import T_df


R_EARTH = 6371e3
MU0 = 4 * np.pi * 1e-7

test1_obs_loc = np.array([[0.0, 0.0, R_EARTH]])
test1_sec_loc = np.array(
    [
        [10.0, 0.0, R_EARTH + 100e3],
        [
            0.0,
            10.0,
            R_EARTH - 50e3,
        ],
    ]
)
test1_name = "Single Observer, Two SECs (Above & Below R_obs)"
test1_notes = (
    "obs_loc shape: (1,3), sec_loc shape: (2,3)\n"
    "SEC1: obs_r < sec_r (standard case).\n"
    "SEC2: obs_r > sec_r ('under_locs' case for this interaction).\n"
    "Expected T shape: (1, 3, 2)."
)

test2_obs_loc = np.array(
    [
        [0.0, 0.0, R_EARTH],
        [20.0, 0.0, R_EARTH + 10e3],
    ]
)
test2_sec_loc = np.array([[10.0, 10.0, R_EARTH + 110e3]])
test2_name = "Two Observers, Single SEC (SEC Above All)"
test2_notes = (
    "obs_loc shape: (2,3), sec_loc shape: (1,3)\n"
    "All interactions: obs_r < sec_r.\n"
    "Expected T shape: (2, 3, 1)."
)

test3_obs_loc = np.array([[45.0, 45.0, R_EARTH]])
test3_sec_loc = np.array([[45.0, 45.0, R_EARTH + 200e3]])
test3_name = "Special Case: Observer & SEC at Same Lat/Lon"
test3_notes = (
    "obs_loc shape: (1,3), sec_loc shape: (1,3)\n"
    "Angular distance theta should be 0. sin(theta) = 0.\n"
    "Btheta should be 0 due to handler for sin(theta)=0.\n"
    "T[0,0,0] (North) and T[0,1,0] (East) should be 0.\n"
    "T[0,2,0] (Down = -Br) should be non-zero.\n"
    "Expected T shape: (1, 3, 1)."
)

test4_obs_loc = np.array(
    [
        [10.0, 10.0, R_EARTH],
        [-10.0, -10.0, R_EARTH + 20e3],
    ]
)
test4_sec_loc = np.array(
    [
        [15.0, 15.0, R_EARTH + 50e3],
        [-15.0, -15.0, R_EARTH - 30e3],
    ]
)
test4_name = "Complex Interaction: 2 Observers, 2 SECs (Mixed Radii)"
test4_notes = (
    "obs_loc shape: (2,3), sec_loc shape: (2,3)\n"
    "Interactions and r comparisons:\n"
    "  (obs1, sec1): R_EARTH < R_EARTH + 50e3 (standard)\n"
    "  (obs1, sec2): R_EARTH > R_EARTH - 30e3 (under_locs)\n"
    "  (obs2, sec1): R_EARTH + 20e3 < R_EARTH + 50e3 (standard)\n"
    "  (obs2, sec2): R_EARTH + 20e3 > R_EARTH - 30e3 (under_locs)\n"
    "Expected T shape: (2, 3, 2)."
)

# --- Run Tests and Print Results ---

tdf_test_cases = [
    (test1_name, test1_obs_loc, test1_sec_loc, test1_notes),
    (test2_name, test2_obs_loc, test2_sec_loc, test2_notes),
    (test3_name, test3_obs_loc, test3_sec_loc, test3_notes),
    (test4_name, test4_obs_loc, test4_sec_loc, test4_notes),
]

print("Running T_df Magnetic Field Transfer Function Calculations...\n")

for name, obs_loc, sec_loc, notes in tdf_test_cases:
    print(f"--- Test Case: {name} ---")
    print(f"Observer Locations (obs_loc, shape {obs_loc.shape}):\n{obs_loc}")
    print(f"SEC Locations (sec_loc, shape {sec_loc.shape}):\n{sec_loc}")
    print(f"Notes:\n{notes}")

    try:
        result_T_df = T_df(obs_loc, sec_loc)

        print(f"\nResulting T_df matrix (shape {result_T_df.shape}):")
        with np.printoptions(precision=5, suppress=True, linewidth=120):
            print(result_T_df)

    except Exception as e:
        print(f"\nError during T_df calculation: {e}")
        import traceback

        traceback.print_exc()

    print("-" * (len(name) + 20))
    print("\n")
