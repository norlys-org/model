use crate::{
    grid::GeographicalPoint,
    sphere::{angular_distance, bearing},
};
use std::f32::consts::PI;

/// Physical constant: permeability of free space (µ0)
const MU0: f32 = 4.0 * PI * 1e-7;

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
/// A 3D vector `Vec<Vec<Vec<f32>>>` representing the transfer matrix T, with dimensions [nobs][3][nsec].
pub fn t_df(obs_locs: &[GeographicalPoint], secs_locs: &[GeographicalPoint]) -> Vec<Vec<Vec<f32>>> {
    let nobs = obs_locs.len();
    let nsec = secs_locs.len();

    // Convert location data for calculations
    let obs_lat_lon: Vec<(f32, f32)> = obs_locs.iter().map(|p| (p.latitude, p.longitude)).collect();
    let secs_lat_lon: Vec<(f32, f32)> = secs_locs
        .iter()
        .map(|p| (p.latitude, p.longitude))
        .collect();

    let theta = angular_distance(&obs_lat_lon, &secs_lat_lon);
    let alpha = bearing(&obs_lat_lon, &secs_lat_lon);

    // Initialize the transfer matrix (3D vector)
    // This creates a vector of size nobs x 3 x nsec filled with zeros
    let mut t: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; nsec]; 3]; nobs];

    // Calculate transfer function for each observation-SEC pair
    for i in 0..nobs {
        let obs_r = obs_locs[i].radius;

        for j in 0..nsec {
            let sec_r = secs_locs[j].radius;
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
mod tests {
    use crate::secs::R_EARTH;

    use super::*;

    const R_SECS: f32 = R_EARTH + 110e3;

    #[test]
    fn test_t_df_dimensions() {
        let obs_locs = vec![
            GeographicalPoint {
                latitude: 0.0,
                longitude: 0.0,
                radius: R_EARTH,
            },
            GeographicalPoint {
                latitude: 10.0,
                longitude: 10.0,
                radius: R_EARTH,
            },
        ];
        let secs_locs = vec![
            GeographicalPoint {
                latitude: 90.0,
                longitude: 0.0,
                radius: R_SECS,
            },
            GeographicalPoint {
                latitude: 80.0,
                longitude: 0.0,
                radius: R_SECS,
            },
            GeographicalPoint {
                latitude: 70.0,
                longitude: 0.0,
                radius: R_SECS,
            },
        ];

        let result = t_df(&obs_locs, &secs_locs);

        assert_eq!(result.len(), 2, "Number of observation points mismatch");
        assert_eq!(
            result[0].len(),
            3,
            "Number of components mismatch (should be 3: Bx, By, Bz)"
        );
        assert_eq!(result[0][0].len(), 3, "Number of SEC points mismatch");
        assert_eq!(
            result[1].len(),
            3,
            "Number of components mismatch for obs 2"
        );
        assert_eq!(
            result[1][0].len(),
            3,
            "Number of SEC points mismatch for obs 2"
        );
    }
}
