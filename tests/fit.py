import numpy as np
from secs import SECS
from t_df import R_EARTH


def run_test_scenario(name, sec_df_loc, obs_loc, obs_B, epsilon, r_earth=R_EARTH):
    print(f"--- Test Case: {name} ---")
    # Ensure inputs are numpy arrays for consistency
    sec_df_loc = np.asarray(sec_df_loc)
    obs_loc = np.asarray(obs_loc)
    obs_B = np.asarray(obs_B)

    print(f"SEC DF Locations (lat, lon, r=R_earth+alt):\n{sec_df_loc}")
    print(f"Observation Locations (lat, lon, r=R_earth+alt):\n{obs_loc}")
    print(f"Observed B-field (Bx, By, Bz) (shape {obs_B.shape}):\n{obs_B}")
    print(f"Epsilon: {epsilon}\n")

    try:
        # Use original, potentially buggy SECS __init__ and methods
        secs_system = SECS(sec_df_loc=sec_df_loc, sec_cf_loc=None)

        current_obs_B = obs_B  # Copy or ensure correct shape for fit
        nobs = obs_loc.shape[0]
        if nobs > 0:
            if current_obs_B.ndim == 1 and current_obs_B.shape[0] == 3 and nobs == 1:
                current_obs_B = current_obs_B[np.newaxis, np.newaxis, :]  # (1,1,3)
            elif (
                current_obs_B.ndim == 2
                and current_obs_B.shape[0] == nobs
                and current_obs_B.shape[1] == 3
            ):
                current_obs_B = current_obs_B[np.newaxis, :, :]  # (1,nobs,3)
        elif current_obs_B.size > 0:  # No obs loc but B data given? Problem. Clear B.
            print("Warning: Obs locations empty but B data provided. Clearing B data.")
            current_obs_B = np.empty((1, 0, 3))  # Shape (1, 0, 3)

        # Call fit, catching the expected UnboundLocalError for the 0-SEC case
        try:
            secs_system.fit(obs_loc, current_obs_B, epsilon=epsilon, mode="relative")
        except UnboundLocalError as e:
            # Check if this error is expected (nsec == 0)
            if secs_system.nsec == 0:
                print(
                    f"Known Issue: Original Python SECS._calc_T failed as expected with UnboundLocalError for 0 SECs."
                )
                print(
                    "Rust implementation should handle this case gracefully (e.g., return empty results)."
                )
                print("Skipping Python results printout for this scenario.")
                print("-" * 40 + "\n")
                return  # Stop processing this specific scenario
            else:
                # Unexpected UnboundLocalError, re-raise
                print("!!! Unexpected UnboundLocalError !!!")
                raise e
        except ValueError as e:
            # Catch potential ValueErrors from checks within fit/helpers
            print(f"!!! ValueError during fit: {e}")
            # Print traceback for debugging if needed
            # import traceback
            # traceback.print_exc()
            print("-" * 40 + "\n")
            return

        # Print results if fit completed without the expected error
        print("Resulting sec_amps (1, nsec):")
        with np.printoptions(precision=8, suppress=True, floatmode="fixed"):
            print(secs_system.sec_amps)
        print("\nResulting sec_amps_var (1, nsec):")
        with np.printoptions(precision=8, suppress=True, floatmode="fixed"):
            print(secs_system.sec_amps_var)

    except Exception as e:
        # Catch any other errors during setup or execution
        print(f"!!! Error during test scenario '{name}': {e}")
        # Optional traceback
        import traceback

        traceback.print_exc()

    print("-" * 40 + "\n")


# --- Test Scenarios Definitions (Using np.empty((0,3)) for empty case) ---

# Scenario 1: Simple Case - 1 SEC, 1 Observation
sc1_sec_df_loc = np.array([[0.0, 0.0, R_EARTH + 110e3]])
sc1_obs_loc = np.array([[0.0, 0.0, R_EARTH + 0.0]])
sc1_obs_B = np.array([[0.0, 0.0, 10.0]])  # B=(0,0,10) for the single obs
sc1_epsilon = 0.01
run_test_scenario(
    "Scenario 1: 1 SEC, 1 Obs (Directly Below)",
    sc1_sec_df_loc,
    sc1_obs_loc,
    sc1_obs_B,
    sc1_epsilon,
)

# Scenario 2: Multiple SECs, Multiple Observations
sc2_sec_df_loc = np.array([[10.0, 0.0, R_EARTH + 110e3], [-10.0, 5.0, R_EARTH + 120e3]])
sc2_obs_loc = np.array(
    [
        [5.0, 1.0, R_EARTH + 0.0],
        [-5.0, 2.0, R_EARTH + 10e3],  # Obs at 10km altitude
    ]
)
sc2_obs_B = np.array(
    [
        [1.0, 2.0, 5.0],  # For obs point 1
        [-1.0, 1.5, 6.0],  # For obs point 2
    ]
)
sc2_epsilon = 0.05
run_test_scenario(
    "Scenario 2: 2 SECs, 2 Obs", sc2_sec_df_loc, sc2_obs_loc, sc2_obs_B, sc2_epsilon
)

# Scenario 3: More Observations than SECs (Overdetermined)
sc3_sec_df_loc = np.array([[50.0, 30.0, R_EARTH + 100e3]])
sc3_obs_loc = np.array(
    [
        [48.0, 28.0, R_EARTH + 0.0],
        [50.0, 32.0, R_EARTH + 0.0],
        [52.0, 30.0, R_EARTH + 5e3],
    ]
)
sc3_obs_B = np.array([[0.5, -0.2, 8.0], [0.3, 0.1, 8.5], [0.6, -0.1, 7.5]])
sc3_epsilon = 0.1
run_test_scenario(
    "Scenario 3: 1 SEC, 3 Obs (Overdetermined)",
    sc3_sec_df_loc,
    sc3_obs_loc,
    sc3_obs_B,
    sc3_epsilon,
)

# Scenario 4: More SECs than Observation Components (Underdetermined T_obs_flat columns vs rows)
sc4_sec_df_loc = np.array(
    [
        [0.0, 0.0, R_EARTH + 110e3],
        [5.0, 5.0, R_EARTH + 110e3],
        [-5.0, -5.0, R_EARTH + 110e3],
        [0.0, 10.0, R_EARTH + 110e3],  # 4 SECs
    ]
)
sc4_obs_loc = np.array(
    [
        [1.0, 1.0, R_EARTH + 0.0]  # 1 observation point => 3 flat components
    ]
)
sc4_obs_B = np.array(
    [  # B-field for that single observation point
        [2.0, 3.0, -7.0]
    ]
)
sc4_epsilon = 0.001
run_test_scenario(
    "Scenario 4: 4 SECs, 1 Obs (Underdetermined)",
    sc4_sec_df_loc,
    sc4_obs_loc,
    sc4_obs_B,
    sc4_epsilon,
)

# Scenario 5: Zero SECs
sc5_sec_df_loc = np.empty((0, 3))  # Correct empty shape
sc5_obs_loc = np.array([[0.0, 0.0, R_EARTH + 0.0]])
sc5_obs_B = np.array([[1.0, 1.0, 1.0]])  # B data doesn't matter if no SECs
sc5_epsilon = 0.05
run_test_scenario(
    "Scenario 5: 0 SECs, 1 Obs", sc5_sec_df_loc, sc5_obs_loc, sc5_obs_B, sc5_epsilon
)

# Scenario 6: Zero Observations
sc6_sec_df_loc = np.array([[0.0, 0.0, R_EARTH + 110e3]])
sc6_obs_loc = np.empty((0, 3))  # Correct empty shape
sc6_obs_B = np.empty((0, 3))  # Correct empty shape -> becomes (1, 0, 3)
sc6_epsilon = 0.05
run_test_scenario(
    "Scenario 6: 1 SEC, 0 Obs", sc6_sec_df_loc, sc6_obs_loc, sc6_obs_B, sc6_epsilon
)

# Scenario 7: One SEC at North Pole, obs nearby
sc7_sec_df_loc = np.array([[90.0, 0.0, R_EARTH + 110e3]])
sc7_obs_loc = np.array([[85.0, 0.0, R_EARTH + 0.0]])
sc7_obs_B = np.array([[0.0, 5.0, 10.0]])  # Single obs point B
sc7_epsilon = 0.02
run_test_scenario(
    "Scenario 7: 1 SEC at Pole, 1 Obs",
    sc7_sec_df_loc,
    sc7_obs_loc,
    sc7_obs_B,
    sc7_epsilon,
)

# Scenario 8: Identical SECs (Zero Singular Value)
sc8_sec_df_loc = np.array(
    [
        [20.0, 20.0, R_EARTH + 110e3],
        [20.0, 20.0, R_EARTH + 110e3],  # Identical SEC
    ]
)
sc8_obs_loc = np.array([[10.0, 10.0, R_EARTH + 0.0]])
sc8_obs_B = np.array([[1.0, 2.0, 3.0]])  # Single obs point B
sc8_epsilon = 1e-8
run_test_scenario(
    "Scenario 8: Identical SECs (Zero Singular Value)",
    sc8_sec_df_loc,
    sc8_obs_loc,
    sc8_obs_B,
    sc8_epsilon,
)


print("Python test data generation complete.")
