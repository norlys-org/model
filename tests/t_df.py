import numpy as np
from secs import T_df


R_EARTH_M = 6371e3  # Earth radius in meters
IONO_ALT_M = 110e3  # Typical ionospheric E-region altitude in meters

obs_loc1 = np.array([[60.0, 0.0, R_EARTH_M]])  # Lat, Lon (deg), Radius (m)
sec_loc1 = np.array([[62.0, 5.0, R_EARTH_M + IONO_ALT_M]])
T1 = T_df(obs_loc1, sec_loc1)

print("--- Test Case 1: Ground Obs, Iono SEC (obs_r < sec_r) ---")
print(f"Obs Loc: {obs_loc1}")
print(f"SEC Loc: {sec_loc1}")
print(f"T_df Result (Bx, By, Bz) in Tesla for unit current (1 Ampere-meter? Check paper for exact unit of SECs):\n{T1}\n")

# Test Case 2: Satellite observation, SEC in ionosphere below
# (obs_r > sec_r)
obs_loc2 = np.array([[70.0, 10.0, R_EARTH_M + 400e3]]) # Satellite at 400km altitude
sec_loc2 = np.array([[71.0, 12.0, R_EARTH_M + IONO_ALT_M]])
T2 = T_df(obs_loc2, sec_loc2)

print("--- Test Case 2: Satellite Obs, Iono SEC (obs_r > sec_r) ---")
print(f"Obs Loc: {obs_loc2}")
print(f"SEC Loc: {sec_loc2}")
print(f"T_df Result:\n{T2}\n")

# Test Case 3: Observation and SEC at the same radius (e.g., both in ionosphere)
# (obs_r == sec_r, handled by _calc_T_df_under)
obs_loc3 = np.array([[50.0, -20.0, R_EARTH_M + IONO_ALT_M]])
sec_loc3 = np.array([[50.5, -19.0, R_EARTH_M + IONO_ALT_M]])
T3 = T_df(obs_loc3, sec_loc3)

print("--- Test Case 3: Obs and SEC at Same Radius ---")
print(f"Obs Loc: {obs_loc3}")
print(f"SEC Loc: {sec_loc3}")
print(f"T_df Result:\n{T3}\n")

# Test Case 4: Observation directly above SEC (same lat, lon)
# (theta = 0, sin_theta = 0)
obs_loc4 = np.array([[80.0, 30.0, R_EARTH_M + 200e3]]) # obs_r > sec_r
sec_loc4 = np.array([[80.0, 30.0, R_EARTH_M + IONO_ALT_M]])
T4 = T_df(obs_loc4, sec_loc4)

print("--- Test Case 4: Obs directly above SEC (same lat/lon, obs_r > sec_r) ---")
print(f"Obs Loc: {obs_loc4}")
print(f"SEC Loc: {sec_loc4}")
print(f"T_df Result (Btheta part should be zeroed out by sin_theta division):\n{T4}\n")

# Test Case 5: Observation at SEC location (same lat, lon, r)
# (theta = 0, sin_theta = 0, obs_r == sec_r) -> singularity expected for Br
# The _calc_T_df_under has factor -> inf
obs_loc5 = np.array([[85.0, 45.0, R_EARTH_M + IONO_ALT_M]])
sec_loc5 = np.array([[85.0, 45.0, R_EARTH_M + IONO_ALT_M]])
T5 = T_df(obs_loc5, sec_loc5)

print("--- Test Case 5: Obs AT SEC location (singularity) ---")
print(f"Obs Loc: {obs_loc5}")
print(f"SEC Loc: {sec_loc5}")
print(f"T_df Result (expect inf/-inf for Br component, 0 for Btheta components):\n{T5}\n")


# Test Case 6: Multiple observation points, single SEC
obs_loc6 = np.array([
    [60.0, 0.0, R_EARTH_M],
    [70.0, 10.0, R_EARTH_M + 400e3]
])
sec_loc6 = np.array([[65.0, 5.0, R_EARTH_M + IONO_ALT_M]])
T6 = T_df(obs_loc6, sec_loc6) # Expected shape: (2, 3, 1)

print("--- Test Case 6: Multiple Obs, Single SEC ---")
print(f"Obs Locs:\n{obs_loc6}")
print(f"SEC Loc: {sec_loc6}")
print(f"T_df Result (shape {T6.shape}):\n{T6}\n")

# Test Case 7: Single observation point, multiple SECs
obs_loc7 = np.array([[60.0, 0.0, R_EARTH_M]])
sec_loc7 = np.array([
    [62.0, 5.0, R_EARTH_M + IONO_ALT_M],
    [63.0, -5.0, R_EARTH_M + IONO_ALT_M + 10e3] # Slightly different altitude
])
T7 = T_df(obs_loc7, sec_loc7) # Expected shape: (1, 3, 2)

print("--- Test Case 7: Single Obs, Multiple SECs ---")
print(f"Obs Loc: {obs_loc7}")
print(f"SEC Locs:\n{sec_loc7}")
print(f"T_df Result (shape {T7.shape}):\n{T7}\n")

# Test Case 8: Antipodal case (theta = pi)
# obs_r < sec_r
obs_loc8 = np.array([[60.0, 0.0, R_EARTH_M]])
sec_loc8 = np.array([[-60.0, 180.0, R_EARTH_M + IONO_ALT_M]]) # Antipodal to obs_loc1
T8 = T_df(obs_loc8, sec_loc8)

print("--- Test Case 8: Antipodal case (obs_r < sec_r) ---")
print(f"Obs Loc: {obs_loc8}")
print(f"SEC Loc: {sec_loc8}")
print(f"T_df Result:\n{T8}\n")
