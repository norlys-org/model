use crate::{
    grid::GeographicalPoint,
    secs::R_EARTH,
    sphere::{angular_distance, bearing},
};
use std::f64::consts::PI;

/// Physical constant: permeability of free space (µ0)
const MU0: f64 = 4.0 * PI * 1e-7;

/// Calculates the "Transfer Matrix" (T) for Divergence-Free Spherical Elementary Current Systems (SECS).
///
/// **What it does:**
/// This function determines the magnetic field influence that each hypothetical "elementary current"
/// located high above the Earth (at `secs_locs`) would have on each ground-based observation
/// point (`obs_locs`). It essentially builds a map of potential influences.
/// This function deals with "divergence-free" SECS, which represent currents that flow in closed loops.
///
/// **The Physics:**
/// The calculation is based on the fundamental principle that electric currents create magnetic fields.
/// This relationship is formally described by the Biot-Savart Law.
/// (See: https://en.wikipedia.org/wiki/Biot%E2%80%93Savart_law)
///
/// **Specific Formulas:**
/// The exact mathematical formulas implemented here are analytical solutions derived from the
/// Biot-Savart law specifically for the geometry of these spherical elementary currents. They come
/// from the work of Amm & Viljanen in their paper on using SECS for ionospheric field continuation.
/// The comments below reference specific equations from that paper (e.g., Eq. 9, 10, A.7, A.8).
/// (See: https://link.springer.com/content/pdf/10.1007/978-3-030-26732-2.pdf)
///
/// **The Transfer Matrix (Output):**
/// The function returns a 3D matrix `T`. Each element `T[i][k][j]` represents the magnetic field's
/// k-th component (where 0=Bx/North, 1=By/East, 2=Bz/Down) measured at the i-th observation point (`obs_locs[i]`),
/// *caused by* the j-th elementary current (`secs_locs[j]`) *if that current had a standard strength of 1 Ampere*.
///
/// # Arguments
/// * `obs_locs` - A slice of `GeographicalPoint` structures representing the observation locations (e.g., ground magnetometers).
/// * `secs_locs` - A slice of `GeographicalPoint` structures representing the locations (poles) of the hypothetical Spherical Elementary Currents.
///
/// # Returns
/// A 3D vector `Vec<Vec<Vec<f64>>>` representing the transfer matrix T, with dimensions [nobs][3][nsec].
pub fn t_df(obs_locs: &[GeographicalPoint], secs_locs: &[GeographicalPoint]) -> Vec<Vec<Vec<f64>>> {
    let nobs = obs_locs.len();
    let nsec = secs_locs.len();

    // Convert location data for calculations
    let obs_lat_lon: Vec<(f64, f64)> = obs_locs.iter().map(|p| (p.latitude, p.longitude)).collect();
    let secs_lat_lon: Vec<(f64, f64)> = secs_locs
        .iter()
        .map(|p| (p.latitude, p.longitude))
        .collect();

    let theta = angular_distance(&obs_lat_lon, &secs_lat_lon);
    let alpha = bearing(&obs_lat_lon, &secs_lat_lon);

    // Initialize the transfer matrix (3D vector)
    // This creates a vector of size nobs x 3 x nsec filled with zeros
    let mut t: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; nsec]; 3]; nobs];

    // Calculate transfer function for each observation-SEC pair
    for i in 0..nobs {
        let obs_r = R_EARTH + obs_locs[i].altitude;

        for j in 0..nsec {
            let sec_r = R_EARTH + secs_locs[j].altitude;
            let x = obs_r / sec_r;
            let sin_theta = theta[i][j].sin();
            let cos_theta = theta[i][j].cos();

            // Factor used in calculations
            let factor = 1.0 / (1.0 - 2.0 * x * cos_theta + x * x).sqrt();

            // Radial component - Amm & Viljanen: Equation 9
            let mut br = MU0 / (4.0 * PI * obs_r) * (factor - 1.0);

            // Theta component - Amm & Viljanen: Equation 10
            let mut b_theta = -MU0 / (4.0 * PI * obs_r) * (factor * (x - cos_theta) + cos_theta);

            if sin_theta != 0.0 {
                b_theta /= sin_theta;
            } else {
                b_theta = 0.0;
            }

            // Check if SEC is below observation
            if sec_r < obs_r {
                // Flipped calculation for SECs below observations
                let x = sec_r / obs_r;

                // Amm & Viljanen: Equation A.7
                br = MU0 * x / (4.0 * PI * obs_r)
                    * (1.0 / (1.0 - 2.0 * x * cos_theta + x * x).sqrt() - 1.0);

                // Amm & Viljanen: Equation A.8
                b_theta = -MU0 / (4.0 * PI * obs_r)
                    * ((obs_r - sec_r * cos_theta)
                        / (obs_r * obs_r - 2.0 * obs_r * sec_r * cos_theta + sec_r * sec_r).sqrt()
                        - 1.0);

                if sin_theta != 0.0 {
                    b_theta /= sin_theta;
                } else {
                    b_theta = 0.0;
                }
            }

            // Transform to Cartesian coordinates
            t[i][0][j] = -b_theta * alpha[i][j].sin(); // Bx
            t[i][1][j] = -b_theta * alpha[i][j].cos(); // By
            t[i][2][j] = -br; // Bz
        }
    }

    t
}

#[cfg(test)]
mod secs_t_df_tests {
    use super::*;
    use crate::grid::GeographicalPoint;

    // If not using assert_approx_eq crate, define a helper:
    fn assert_f64_near(a: f64, b: f64, epsilon: f64, msg: &str) {
        if (a.is_nan() && b.is_nan())
            || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
        {
            return;
        }
        assert!(
            (a - b).abs() < epsilon,
            "{} | {:.10e} != {:.10e}",
            msg,
            a,
            b
        );
    }

    // Define a default epsilon for comparisons, can be adjusted per test if needed
    const DEFAULT_EPSILON: f64 = 1e-15; // A bit looser than Python's typical float64 precision for safety
    const IONO_ALT_M: f64 = 110_000.0;
    const SAT_ALT_M: f64 = 400_000.0;
    const SAT_ALT2_M: f64 = 200_000.0;

    #[test]
    fn test_case_1_ground_obs_iono_sec() {
        let obs_locs = vec![
            GeographicalPoint {
                latitude: 60.0,
                longitude: 0.0,
                altitude: 0.0,
            }, // R_EARTH
        ];
        let secs_locs = vec![
            GeographicalPoint {
                latitude: 62.0,
                longitude: 5.0,
                altitude: IONO_ALT_M,
            }, // R_EARTH + 110e3
        ];

        let t = t_df(&obs_locs, &secs_locs);

        // Expected from Python:
        // Bx:  1.38473593e-13
        // By:  1.55458626e-13
        // Bz: -2.59975868e-13
        assert_f64_near(t[0][0][0], 1.38473593e-13, DEFAULT_EPSILON, "TC1 Bx");
        assert_f64_near(t[0][1][0], 1.55458626e-13, DEFAULT_EPSILON, "TC1 By");
        assert_f64_near(t[0][2][0], -2.59975868e-13, DEFAULT_EPSILON, "TC1 Bz");
    }

    #[test]
    fn test_case_2_satellite_obs_iono_sec() {
        let obs_locs = vec![
            GeographicalPoint {
                latitude: 70.0,
                longitude: 10.0,
                altitude: SAT_ALT_M,
            }, // R_EARTH + 400e3
        ];
        let secs_locs = vec![
            GeographicalPoint {
                latitude: 71.0,
                longitude: 12.0,
                altitude: IONO_ALT_M,
            }, // R_EARTH + 110e3
        ];

        let t = t_df(&obs_locs, &secs_locs);

        // Expected from Python:
        // Bx: -5.55039253e-14
        // By: -3.57533223e-14
        // Bz: -2.83500034e-13
        assert_f64_near(t[0][0][0], -5.55039253e-14, DEFAULT_EPSILON, "TC2 Bx");
        assert_f64_near(t[0][1][0], -3.57533223e-14, DEFAULT_EPSILON, "TC2 By");
        assert_f64_near(t[0][2][0], -2.83500034e-13, DEFAULT_EPSILON, "TC2 Bz");
    }

    #[test]
    fn test_case_3_obs_and_sec_at_same_radius() {
        let obs_locs = vec![GeographicalPoint {
            latitude: 50.0,
            longitude: -20.0,
            altitude: IONO_ALT_M,
        }];
        let secs_locs = vec![GeographicalPoint {
            latitude: 50.5,
            longitude: -19.0,
            altitude: IONO_ALT_M,
        }];

        let t = t_df(&obs_locs, &secs_locs);

        // Expected from Python:
        // Bx:  6.81364374e-13
        // By:  8.59460003e-13
        // Bz: -1.07371836e-12
        assert_f64_near(t[0][0][0], 6.81364374e-13, DEFAULT_EPSILON, "TC3 Bx");
        assert_f64_near(t[0][1][0], 8.59460003e-13, DEFAULT_EPSILON, "TC3 By");
        assert_f64_near(t[0][2][0], -1.07371836e-12, DEFAULT_EPSILON, "TC3 Bz");
    }

    #[test]
    fn test_case_4_obs_directly_above_sec() {
        let obs_locs = vec![
            GeographicalPoint {
                latitude: 80.0,
                longitude: 30.0,
                altitude: SAT_ALT2_M,
            }, // R_EARTH + 200e3
        ];
        let secs_locs = vec![
            GeographicalPoint {
                latitude: 80.0,
                longitude: 30.0,
                altitude: IONO_ALT_M,
            }, // R_EARTH + 110e3
        ];

        let t = t_df(&obs_locs, &secs_locs);

        // Expected from Python (Bx and By are effectively zero):
        // Bx: -9.74437523e-19
        // By: -5.96670897e-35 (extremely small, treat as 0)
        // Bz: -1.08088278e-12
        let high_precision_epsilon = 1e-20; // For very small numbers near zero.
                                            // Or an epsilon relative to the magnitude if not comparing to zero.
        assert_f64_near(
            t[0][0][0],
            -9.74437523e-19,
            high_precision_epsilon,
            "TC4 Bx",
        ); // Python has sin(alpha) non-zero
        assert_f64_near(t[0][1][0], 0.0, high_precision_epsilon, "TC4 By"); // Python has cos(alpha) effectively zero
        assert_f64_near(t[0][2][0], -1.08088278e-12, DEFAULT_EPSILON, "TC4 Bz");
    }

    #[test]
    fn test_case_5_obs_at_sec_location_singularity() {
        let obs_locs = vec![GeographicalPoint {
            latitude: 85.0,
            longitude: 45.0,
            altitude: IONO_ALT_M,
        }];
        let secs_locs = vec![GeographicalPoint {
            latitude: 85.0,
            longitude: 45.0,
            altitude: IONO_ALT_M,
        }];

        let t = t_df(&obs_locs, &secs_locs);

        // Expected from Python:
        // Bx: -0.
        // By: -0.
        // Bz: -inf
        assert_f64_near(t[0][0][0], 0.0, DEFAULT_EPSILON, "TC5 Bx"); // Python had -0.0
        assert_f64_near(t[0][1][0], 0.0, DEFAULT_EPSILON, "TC5 By"); // Python had -0.0
        assert!(
            t[0][2][0].is_infinite() && t[0][2][0].is_sign_negative(),
            "TC5 Bz should be -inf, got {}",
            t[0][2][0]
        );
    }

    #[test]
    fn test_case_6_multiple_obs_single_sec() {
        let obs_locs = vec![
            GeographicalPoint {
                latitude: 60.0,
                longitude: 0.0,
                altitude: 0.0,
            },
            GeographicalPoint {
                latitude: 70.0,
                longitude: 10.0,
                altitude: SAT_ALT_M,
            },
        ];
        let secs_locs = vec![GeographicalPoint {
            latitude: 65.0,
            longitude: 5.0,
            altitude: IONO_ALT_M,
        }];

        let t = t_df(&obs_locs, &secs_locs);

        // Obs 1:
        // Bx:  1.31095734e-13
        // By:  5.45320388e-14
        // Bz: -1.46625859e-13
        assert_f64_near(t[0][0][0], 1.31095734e-13, DEFAULT_EPSILON, "TC6 Obs1 Bx");
        assert_f64_near(t[0][1][0], 5.45320388e-14, DEFAULT_EPSILON, "TC6 Obs1 By");
        assert_f64_near(t[0][2][0], -1.46625859e-13, DEFAULT_EPSILON, "TC6 Obs1 Bz");

        // Obs 2:
        // Bx:  7.77118539e-14
        // By:  3.34219561e-14
        // Bz: -1.26026766e-13
        assert_f64_near(t[1][0][0], 7.77118539e-14, DEFAULT_EPSILON, "TC6 Obs2 Bx");
        assert_f64_near(t[1][1][0], 3.34219561e-14, DEFAULT_EPSILON, "TC6 Obs2 By");
        assert_f64_near(t[1][2][0], -1.26026766e-13, DEFAULT_EPSILON, "TC6 Obs2 Bz");
    }

    #[test]
    fn test_case_7_single_obs_multiple_secs() {
        let obs_locs = vec![GeographicalPoint {
            latitude: 60.0,
            longitude: 0.0,
            altitude: 0.0,
        }];
        let secs_locs = vec![
            GeographicalPoint {
                latitude: 62.0,
                longitude: 5.0,
                altitude: IONO_ALT_M,
            },
            GeographicalPoint {
                latitude: 63.0,
                longitude: -5.0,
                altitude: IONO_ALT_M + 10_000.0,
            },
        ];

        let t = t_df(&obs_locs, &secs_locs);

        // SEC 1:
        // Bx:  1.38473593e-13
        // By:  1.55458626e-13
        // Bz: -2.59975868e-13
        assert_f64_near(t[0][0][0], 1.38473593e-13, DEFAULT_EPSILON, "TC7 SEC1 Bx");
        assert_f64_near(t[0][1][0], 1.55458626e-13, DEFAULT_EPSILON, "TC7 SEC1 By");
        assert_f64_near(t[0][2][0], -2.59975868e-13, DEFAULT_EPSILON, "TC7 SEC1 Bz");

        // SEC 2:
        // Bx:  1.44132837e-13
        // By: -1.05941124e-13
        // Bz: -2.12584479e-13
        assert_f64_near(t[0][0][1], 1.44132837e-13, DEFAULT_EPSILON, "TC7 SEC2 Bx");
        assert_f64_near(t[0][1][1], -1.05941124e-13, DEFAULT_EPSILON, "TC7 SEC2 By");
        assert_f64_near(t[0][2][1], -2.12584479e-13, DEFAULT_EPSILON, "TC7 SEC2 Bz");
    }

    #[test]
    fn test_case_8_antipodal_case() {
        let obs_locs = vec![GeographicalPoint {
            latitude: 60.0,
            longitude: 0.0,
            altitude: 0.0,
        }];
        let secs_locs = vec![GeographicalPoint {
            latitude: -60.0,
            longitude: 180.0,
            altitude: IONO_ALT_M,
        }];

        let t = t_df(&obs_locs, &secs_locs);

        // Expected from Python:
        // Bx: 0.0
        // By: 0.0
        // Bz: 7.78089013e-15
        assert_f64_near(t[0][0][0], 0.0, DEFAULT_EPSILON, "TC8 Bx");
        assert_f64_near(t[0][1][0], 0.0, DEFAULT_EPSILON, "TC8 By");
        assert_f64_near(t[0][2][0], 7.78089013e-15, DEFAULT_EPSILON, "TC8 Bz");
    }

    // It's also good to ensure R_EARTH is what we expect for these tests.
    // This isn't a test of t_df itself, but a sanity check for the test setup.
    #[test]
    fn check_r_earth_constant_for_tests() {
        // This value MUST match the one used in the Python script (6371e3 meters)
        // for the altitude calculations to be correct.
        assert_eq!(
            R_EARTH, 6371_000.0,
            "R_EARTH constant mismatch with Python script assumptions."
        );
    }
}
