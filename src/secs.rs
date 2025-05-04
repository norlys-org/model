use nalgebra::{DMatrix, DVector, SVD};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::ops::Range;
use wasm_bindgen::prelude::*;

use crate::grid::{geographical_grid, GeographicalPoint};

const MU0: f32 = 4.0 * PI * 1e-7; // Magnetic permeability in T*m/A (or N/A^2)

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct ObservationVector {
    /// The longitude in degrees.
    pub lon: f32,
    /// The latitude in degrees.
    pub lat: f32,
    // i vector (usually North component) in nano teslas
    pub i: f32,
    // j vector (usually East component) in nano teslas
    pub j: f32,
    // k vector (usually Down component) in nano teslas
    pub k: f32,
    // Altitude from the surface of the earth where the measurement has been conducted (usually 0)
    // in meters
    pub alt: f32,
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct PredictionVector {
    /// The longitude in degrees.
    pub lon: f32,
    /// The latitude in degrees.
    pub lat: f32,
    // i vector (usually North component) in nano teslas
    pub i: f32,
    // j vector (usually East component) in nano teslas
    pub j: f32,
    // k vector (usually Down component) in nano teslas
    pub k: f32,
}

pub type ObservationMatrix = Vec<ObservationVector>;
pub type PredictionMatrix = Vec<PredictionVector>;

struct Secs {
    sec_df_loc: Vec<GeographicalPoint>,
    sec_amps: Option<DVector<f32>>,
}

impl Secs {
    fn new(sec_df_loc: Vec<GeographicalPoint>) -> Self {
        if sec_df_loc.is_empty() {
            panic!("Must specify at least one divergence free SEC location");
        }
        Secs {
            sec_df_loc,
            sec_amps: None,
        }
    }

    fn nsec(&self) -> usize {
        self.sec_df_loc.len()
    }

    /// Calculate the angular distance in radians between sets of points.
    /// Output matrix dimensions: (n_obs, n_sec)
    fn calc_angular_distance(
        obs_loc: &[GeographicalPoint],
        sec_loc: &[GeographicalPoint],
    ) -> DMatrix<f32> {
        let n_obs = obs_loc.len();
        let n_sec = sec_loc.len();
        let mut theta_matrix = DMatrix::<f32>::zeros(n_obs, n_sec);

        for r in 0..n_obs {
            let lat1 = obs_loc[r].lat_rad();
            let lon1 = obs_loc[r].lon_rad();
            let sin_lat1 = lat1.sin();
            let cos_lat1 = lat1.cos();

            for c in 0..n_sec {
                let lat2 = sec_loc[c].lat_rad();
                let lon2 = sec_loc[c].lon_rad();
                let sin_lat2 = lat2.sin();
                let cos_lat2 = lat2.cos();

                let dlon = lon2 - lon1;
                let cos_dlon = dlon.cos();

                // Clamp the argument to arccos to avoid domain errors due to floating point inaccuracies
                let cos_theta =
                    (sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_dlon).clamp(-1.0, 1.0);
                theta_matrix[(r, c)] = cos_theta.acos();
            }
        }
        theta_matrix
    }

    /// Calculate the bearing in radians between sets of points.
    /// Bearing is angle from North (+lat) towards East (+lon).
    /// Python code calculates angle from East (By) towards North (Bx).
    /// Let's adapt the formula to match the Python comment and variable name `alpha`.
    /// Output matrix dimensions: (n_obs, n_sec)
    fn calc_bearing(obs_loc: &[GeographicalPoint], sec_loc: &[GeographicalPoint]) -> DMatrix<f32> {
        let n_obs = obs_loc.len();
        let n_sec = sec_loc.len();
        let mut alpha_matrix = DMatrix::<f32>::zeros(n_obs, n_sec);

        for r in 0..n_obs {
            let lat1 = obs_loc[r].lat_rad();
            let lon1 = obs_loc[r].lon_rad();
            let sin_lat1 = lat1.sin();
            let cos_lat1 = lat1.cos();

            for c in 0..n_sec {
                let lat2 = sec_loc[c].lat_rad();
                let lon2 = sec_loc[c].lon_rad();
                let sin_lat2 = lat2.sin();
                let cos_lat2 = lat2.cos();

                let dlon = lon2 - lon1;
                let sin_dlon = dlon.sin();
                let cos_dlon = dlon.cos();

                let y = sin_dlon * cos_lat2;
                let x = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_dlon;

                // Original Python: alpha = np.pi / 2 - np.arctan2(y, x)
                // atan2(y, x) gives angle from +X axis towards +Y axis.
                // In geo coords: X=North, Y=East. atan2 gives angle from North towards East.
                // Python alpha seems to be angle from East towards North? Let's double check T_df usage.
                // T[:, 0, :] = -Btheta * np.sin(alpha) # Bx (North) component
                // T[:, 1, :] = -Btheta * np.cos(alpha) # By (East) component
                // This implies alpha is angle relative to East, positive towards North.
                // Let angle_from_north = atan2(y, x).
                // If alpha = PI/2 - angle_from_north, then:
                // sin(alpha) = sin(PI/2 - angle_from_north) = cos(angle_from_north)
                // cos(alpha) = cos(PI/2 - angle_from_north) = sin(angle_from_north)
                // So: Bx = -Btheta * cos(angle_from_north), By = -Btheta * sin(angle_from_north)
                // This seems correct: Btheta is positive along the direction *from* sec *to* obs.
                // We want components along North and East.
                // If bearing `az` (angle_from_north) = 0 (obs is North of sec), then B is southward (-Bx), By=0. Bx=-Btheta*cos(0)=-Btheta. By=-Btheta*sin(0)=0. OK.
                // If bearing `az` = PI/2 (obs is East of sec), then B is westward (-By), Bx=0. Bx=-Btheta*cos(PI/2)=0. By=-Btheta*sin(PI/2)=-Btheta. OK.
                // So, the Python alpha calculation seems correct for its usage.
                let angle_from_east_towards_north = PI / 2.0 - y.atan2(x);
                alpha_matrix[(r, c)] = angle_from_east_towards_north;
            }
        }
        alpha_matrix
    }

    /// Calculate the divergence free magnetic field transfer function matrix T.
    /// T maps SEC amplitudes to observed magnetic field components (Bn, Be, Bd).
    ///
    /// Output matrix dimensions: (n_obs * 3, n_sec)
    /// The matrix is structured such that rows correspond to:
    /// obs[0].Bn, obs[0].Be, obs[0].Bd, obs[1].Bn, obs[1].Be, obs[1].Bd, ...
    fn t_df(obs_loc: &[GeographicalPoint], sec_loc: &[GeographicalPoint]) -> DMatrix<f32> {
        let n_obs = obs_loc.len();
        let n_sec = sec_loc.len();

        // Precompute radii
        let obs_r: Vec<f32> = obs_loc.iter().map(|p| p.radius()).collect();
        let sec_r: Vec<f32> = sec_loc.iter().map(|p| p.radius()).collect();

        // Calculate angular distances and bearings
        // theta, alpha dimensions: (n_obs, n_sec)
        let theta = Self::calc_angular_distance(obs_loc, sec_loc);
        let alpha = Self::calc_bearing(obs_loc, sec_loc);

        // Initialize component matrices: (n_obs, n_sec)
        let mut br_matrix = DMatrix::<f32>::zeros(n_obs, n_sec);
        let mut btheta_matrix = DMatrix::<f32>::zeros(n_obs, n_sec);

        let factor_mu = MU0 / (4.0 * PI); // Constant factor

        for r_idx in 0..n_obs {
            for c_idx in 0..n_sec {
                let obs_r_i = obs_r[r_idx];
                let sec_r_j = sec_r[c_idx];
                let theta_ij = theta[(r_idx, c_idx)];
                let cos_theta = theta_ij.cos();
                let sin_theta = theta_ij.sin();

                // --- Case 1: Observation point is AT or ABOVE SEC shell (r_obs >= r_sec) ---
                if obs_r_i >= sec_r_j {
                    let x = obs_r_i / sec_r_j;
                    // Handle potential division by zero if r_sec is zero (shouldn't happen with R_EARTH offset)
                    if sec_r_j == 0.0 {
                        continue;
                    };

                    // Denominator term: sqrt(1 - 2*x*cos(theta) + x^2)
                    let denom_arg = 1.0 - 2.0 * x * cos_theta + x * x;
                    let factor = if denom_arg > 1e-12 {
                        // Avoid sqrt(0) or small negative numbers
                        1.0 / denom_arg.sqrt()
                    } else {
                        // This happens when obs_loc is very close to sec_loc (theta=0, x=1)
                        // The limit is complex, but B should be infinite *at* the current filament.
                        // For practical purposes near the singularity, use a large number or handle separately if needed.
                        // Let's return 0 for B components if exactly at the pole, as the formula breaks.
                        // T_df returns B=0 if theta=0 in Python (via divide by sin_theta=0 -> out=zeros)
                        0.0 // Or consider f32::INFINITY or a very large number? Let's follow Python's effective 0.
                    };

                    // Amm & Viljanen Eq 9 (Br) - Radial component (+up)
                    // Note: Paper uses r, theta, phi coords. Br is radial outward.
                    // We use North, East, Down. Our Bz = -Br.
                    let br = factor_mu / obs_r_i * (factor - 1.0);

                    // Amm & Viljanen Eq 10 (Btheta) - Colatitude component (+southward from SEC pole)
                    // Note: Paper Btheta is positive in direction of increasing theta (away from SEC pole).
                    // Our Bn, Be components depend on bearing `alpha`.
                    let btheta_num = -factor_mu / obs_r_i * (factor * (x - cos_theta) + cos_theta);

                    let btheta = if sin_theta.abs() > 1e-9 {
                        btheta_num / sin_theta
                    } else {
                        // Limit as theta -> 0 : Should be 0
                        0.0
                    };
                    br_matrix[(r_idx, c_idx)] = br;
                    btheta_matrix[(r_idx, c_idx)] = btheta;
                }
                // --- Case 2: Observation point is BELOW SEC shell (r_obs < r_sec) ---
                else {
                    // obs_r_i < sec_r_j
                    let x = sec_r_j / obs_r_i; // Flipped ratio
                                               // Handle potential division by zero if r_obs is zero (shouldn't happen)
                    if obs_r_i == 0.0 {
                        continue;
                    };

                    // Amm & Viljanen Eq A.7 (Br)
                    let denom_arg_a7 = 1.0 - 2.0 * x * cos_theta + x * x;
                    let factor_a7 = if denom_arg_a7 > 1e-12 {
                        1.0 / denom_arg_a7.sqrt()
                    } else {
                        0.0 // As above
                    };
                    let br = factor_mu * x / obs_r_i * (factor_a7 - 1.0);

                    // Amm & Viljanen Eq A.8 (Btheta)
                    // Original: -mu0/(4*pi*obs_r) * ( (obs_r - sec_r*cos_theta)/sqrt(obs_r^2 - 2*obs_r*sec_r*cos_theta + sec_r^2) - 1 ) / sin_theta
                    // Simplify denom: sqrt(obs_r^2*(1 - 2*(sec_r/obs_r)*cos_theta + (sec_r/obs_r)^2)) = obs_r * sqrt(1 - 2*x*cos_theta + x^2)
                    let denom_sqrt_a8 = if denom_arg_a7 > 1e-12 {
                        obs_r_i * denom_arg_a7.sqrt()
                    } else {
                        0.0
                    };

                    let btheta_num = if denom_sqrt_a8 > 1e-12 {
                        -factor_mu / obs_r_i
                            * ((obs_r_i - sec_r_j * cos_theta) / denom_sqrt_a8 - 1.0)
                    } else {
                        // If denom is zero (at pole), numerator likely is too? Limit is complex. Let's set to 0 like Python.
                        0.0
                    };

                    let btheta = if sin_theta.abs() > 1e-9 {
                        btheta_num / sin_theta
                    } else {
                        // Limit as theta -> 0 : Should be 0
                        0.0
                    };

                    br_matrix[(r_idx, c_idx)] = br;
                    btheta_matrix[(r_idx, c_idx)] = btheta;
                }
            }
        }

        // --- Transform Btheta, Br to Bn, Be, Bd (Bx, By, Bz in Python code) ---
        // Output matrix T has shape (n_obs * 3, n_sec)
        // Bx = Bn (North), By = Be (East), Bz = Bd (Down)
        // Python:
        // T[:, 0, :] = -Btheta * np.sin(alpha)  # Bx = North
        // T[:, 1, :] = -Btheta * np.cos(alpha)  # By = East
        // T[:, 2, :] = -Br                     # Bz = Down = - Radial outward

        let mut t_matrix = DMatrix::<f32>::zeros(n_obs * 3, n_sec);

        for j in 0..n_sec {
            // Iterate over SECs (columns)
            for i in 0..n_obs {
                // Iterate over Obs (rows)
                let alpha_ij = alpha[(i, j)];
                let sin_alpha = alpha_ij.sin();
                let cos_alpha = alpha_ij.cos();
                let btheta_ij = btheta_matrix[(i, j)];
                let br_ij = br_matrix[(i, j)];

                let bn = -btheta_ij * sin_alpha; // North component
                let be = -btheta_ij * cos_alpha; // East component
                let bd = -br_ij; // Down component

                t_matrix[(i * 3, j)] = bn;
                t_matrix[(i * 3 + 1, j)] = be;
                t_matrix[(i * 3 + 2, j)] = bd;
            }
        }

        // Convert from Tesla to nanoTesla (as input B is in nT)
        t_matrix *= 1e9;

        t_matrix
    }

    /// Fits the SECS model to observations using SVD.
    fn fit(
        &mut self,
        obs_loc: &[GeographicalPoint],
        obs_b: &DVector<f32>, // Flattened observation vector (Bn1, Be1, Bd1, Bn2, ...)
        epsilon: f32,
    ) {
        let n_obs = obs_loc.len();
        if obs_b.len() != n_obs * 3 {
            panic!(
                "Observation B vector length must be 3 times the number of observation locations."
            );
        }
        if self.nsec() == 0 {
            panic!("No SEC locations defined for fitting.");
        }

        // Calculate the transfer matrix T: (n_obs * 3, n_sec)
        let t_matrix = Self::t_df(obs_loc, &self.sec_df_loc);

        // --- Solve T * m = B for m (sec_amps) using SVD ---
        // B has shape (n_obs * 3, 1)
        // T has shape (n_obs * 3, n_sec)
        // m has shape (n_sec, 1)

        let svd = SVD::new(t_matrix.clone(), true, true); // Compute U and V^T

        let s_max = svd.singular_values.max();
        let threshold = epsilon * s_max;

        // Calculate the inverse of singular values, applying threshold
        let mut s_inv = svd
            .singular_values
            .map(|s| if s < threshold { 0.0 } else { 1.0 / s });

        // Pad s_inv if T is not square (nalgebra SVD might return fewer singular values)
        let rank = s_inv.len();
        let num_rows_s = t_matrix.nrows(); // n_obs * 3
        let num_cols_s = t_matrix.ncols(); // n_sec

        // Create diagonal matrix Sigma_inv of size (n_sec, n_obs*3) - transpose of Sigma in U*Sigma*V^T
        let mut sigma_inv_diag = DMatrix::<f32>::zeros(num_cols_s, num_rows_s);
        for i in 0..rank {
            sigma_inv_diag[(i, i)] = s_inv[i];
        }

        // Calculate m = V * Sigma_inv * U^T * B
        // Need U matrix and V matrix (not V^T)
        let u = svd
            .u
            .expect("SVD U matrix computation failed or was disabled");
        let v_t = svd
            .v_t
            .expect("SVD V^T matrix computation failed or was disabled");
        let v = v_t.transpose(); // V = (V^T)^T

        // m = V * sigma_inv_diag * U^T * obs_b
        let sec_amps_vec = v * sigma_inv_diag * u.transpose() * obs_b; // (nsec, 1)

        self.sec_amps = Some(sec_amps_vec);
    }

    /// Predicts the magnetic field (Bn, Be, Bd) at prediction locations.
    fn predict(&self, pred_loc: &[GeographicalPoint]) -> Result<DMatrix<f32>, String> {
        if pred_loc.is_empty() {
            return Ok(DMatrix::zeros(0, 3)); // Return empty matrix if no prediction points
        }
        let n_pred = pred_loc.len();

        let sec_amps = self
            .sec_amps
            .as_ref()
            .ok_or_else(|| "SECS model has not been fitted yet. Call .fit() first.".to_string())?; // sec_amps shape: (nsec, 1)

        // Calculate the transfer matrix T_pred: (n_pred * 3, n_sec)
        let t_pred = Self::t_df(pred_loc, &self.sec_df_loc);

        // Predict B_pred = T_pred * sec_amps
        // Result dimensions: (n_pred * 3, 1)
        let b_pred_flat = t_pred * sec_amps;

        // Reshape B_pred_flat from (n_pred * 3, 1) to (n_pred, 3)
        // Each column corresponds to Bn, Be, Bd
        let b_pred_reshaped = DMatrix::from_vec(n_pred, 3, b_pred_flat.as_slice().to_vec());

        Ok(b_pred_reshaped) // Shape (n_pred, 3)
    }
}

/// Interpolates magnetic field observations using Spherical Elementary Current Systems (SECS).
///
/// # Arguments
///
/// * `observations` - A vector of observed magnetic field vectors at different locations.
/// * `lat_range` - The latitude range for the SEC grid.
/// * `lat_steps` - The number of latitude steps for the SEC grid.
/// * `lon_range` - The longitude range for the SEC grid.
/// * `lon_steps` - The number of longitude steps for the SEC grid.
/// * `sec_altitude` - The altitude (in meters) above the Earth's surface for the SEC grid.
/// * `prediction_altitude` - The altitude (in meters) for the prediction grid points (assumed same lat/lon range and steps as SEC grid for simplicity, matching Python example).
/// * `epsilon` - Regularization parameter for SVD (default in Python was 0.05 or 0.1).
/// * `pred_lat_range` - The latitude range for the prediction grid.
/// * `pred_lat_steps` - The number of latitude steps for the prediction grid.
/// * `pred_lon_range` - The longitude range for the prediction grid.
/// * `pred_lon_steps` - The number of longitude steps for the prediction grid.
///
/// # Returns
///
/// A `PredictionMatrix` containing the interpolated magnetic field vectors at the prediction grid points.
/// Returns `Err` if fitting or prediction fails.
// We modify the signature slightly to allow separate prediction grid definition and epsilon parameter.
pub fn secs_interpolate(
    observations: ObservationMatrix,
    sec_lat_range: Range<f32>,
    sec_lat_steps: usize,
    sec_lon_range: Range<f32>,
    sec_lon_steps: usize,
    sec_altitude: f32,
    pred_lat_range: Range<f32>,
    pred_lat_steps: usize,
    pred_lon_range: Range<f32>,
    pred_lon_steps: usize,
    prediction_altitude: f32,
    epsilon: f32,
) -> Result<PredictionMatrix, String> {
    if observations.is_empty() {
        return Ok(Vec::new()); // Return empty if no observations
    }

    // 1. Create the SEC grid locations
    let sec_grid = geographical_grid(
        sec_lat_range,
        sec_lat_steps,
        sec_lon_range,
        sec_lon_steps,
        sec_altitude,
    );

    // 2. Prepare observation data
    let n_obs = observations.len();
    let mut obs_loc = Vec::with_capacity(n_obs);
    let mut obs_b_flat_vec = Vec::with_capacity(n_obs * 3);
    for obs in &observations {
        obs_loc.push(GeographicalPoint::new(obs.lat, obs.lon, obs.alt));
        obs_b_flat_vec.push(obs.i); // North Component (Bn)
        obs_b_flat_vec.push(obs.j); // East Component (Be)
        obs_b_flat_vec.push(obs.k); // Down Component (Bd) - Use k if available, else 0? Python uses 0.
                                    // Assuming input i=Bx(N), j=By(E), k=Bz(D)
                                    // Need B in nanoTesla
    }
    // Convert flat vec to DVector (nalgebra column vector)
    // Data is already in nT, T_df converts model output to nT.
    let obs_b_dvector = DVector::from_vec(obs_b_flat_vec);

    // 3. Initialize and fit the SECS model
    let mut secs_model = Secs::new(sec_grid);
    secs_model.fit(&obs_loc, &obs_b_dvector, epsilon); // Use the provided epsilon

    // 4. Create the prediction grid locations
    let pred_grid = geographical_grid(
        pred_lat_range,
        pred_lat_steps,
        pred_lon_range,
        pred_lon_steps,
        prediction_altitude,
    );

    // 5. Predict magnetic fields at prediction locations
    // predict returns Result<DMatrix<f32>, String> with shape (n_pred, 3)
    let b_pred_matrix = secs_model.predict(&pred_grid)?; // Propagate potential error

    // 6. Format the results into PredictionMatrix
    let n_pred = pred_grid.len();
    let mut predictions = Vec::with_capacity(n_pred);
    for i in 0..n_pred {
        let point = &pred_grid[i];
        predictions.push(PredictionVector {
            lon: point.longitude,
            lat: point.latitude,
            i: b_pred_matrix[(i, 0)], // Predicted North component (Bn) in nT
            j: b_pred_matrix[(i, 1)], // Predicted East component (Be) in nT
            k: b_pred_matrix[(i, 2)], // Predicted Down component (Bd) in nT
        });
    }

    Ok(predictions)
}

// --- Tests (Keep the existing grid tests and add SECS tests if desired) ---
#[cfg(test)]
mod tests {
    use super::*;

    // --- Basic Integration Test ---
    #[test]
    fn test_secs_interpolate_basic_run() {
        // Define simple observation points
        let observations = vec![
            ObservationVector {
                lat: 60.0,
                lon: 0.0,
                i: 10.0,
                j: 5.0,
                k: -2.0,
                alt: 0.0,
            }, // B = (10, 5, -2) nT North, East, Down
            ObservationVector {
                lat: 65.0,
                lon: 10.0,
                i: 12.0,
                j: 3.0,
                k: -1.0,
                alt: 0.0,
            },
            ObservationVector {
                lat: 62.0,
                lon: -5.0,
                i: 9.0,
                j: 6.0,
                k: -3.0,
                alt: 0.0,
            },
        ];

        // Define SEC grid parameters
        let sec_lat_range = 50.0..70.0;
        let sec_lat_steps = 5;
        let sec_lon_range = -20.0..20.0;
        let sec_lon_steps = 5;
        let sec_altitude = 110_000.0; // 110 km

        // Define Prediction grid parameters (can be same or different)
        let pred_lat_range = 55.0..75.0;
        let pred_lat_steps = 3; // Fewer points for faster test
        let pred_lon_steps = 3;
        let pred_lon_range = -10.0..10.0;
        let prediction_altitude = 0.0; // Predict at ground level

        let epsilon = 0.1;

        // Run the interpolation
        let result = secs_interpolate(
            observations,
            sec_lat_range,
            sec_lat_steps,
            sec_lon_range,
            sec_lon_steps,
            sec_altitude,
            pred_lat_range,
            pred_lat_steps,
            pred_lon_range,
            pred_lon_steps,
            prediction_altitude,
            epsilon,
        );

        // Check if it ran successfully and produced the expected number of points
        assert!(result.is_ok());
        let predictions = result.unwrap();
        assert_eq!(predictions.len(), pred_lat_steps * pred_lon_steps); // 3 * 3 = 9 prediction points

        // Optionally, add checks for reasonable values, non-NaN, etc.
        for p in predictions {
            assert!(!p.i.is_nan() && p.i.is_finite());
            assert!(!p.j.is_nan() && p.j.is_finite());
            assert!(!p.k.is_nan() && p.k.is_finite());
            // Check if values are roughly within expected geophysical ranges (e.g., not millions of nT)
            assert!(p.i.abs() < 10000.0);
            assert!(p.j.abs() < 10000.0);
            assert!(p.k.abs() < 10000.0);
        }
    }
}
