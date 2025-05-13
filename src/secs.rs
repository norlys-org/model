use crate::{grid::GeographicalPoint, matrix::t_df};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

pub const R_EARTH: f64 = 6371e3;

// #[wasm_bindgen]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct ObservationVector {
    /// The longitude in degrees.
    pub lon: f64,
    /// The latitude in degrees.
    pub lat: f64,
    // i vector (usually x magnetometer component) in nano teslas
    pub i: f64,
    // j vector (usually y magnetometer component) in nano teslas
    pub j: f64,
    // k vector (usually k magnetometer component) in nano teslas
    pub k: f64,
    // Altitude from the surface of the earth where the measurement has been conducted (usually 0)
    // in meters
    pub alt: f64,
}

// #[wasm_bindgen]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct PredictionVector {
    /// The longitude in degrees.
    pub lon: f64,
    /// The latitude in degrees.
    pub lat: f64,
    // i vector (usually x magnetometer component) in nano teslas
    pub i: f64,
    // j vector (usually y magnetometer component) in nano teslas
    pub j: f64,
    // k vector (usually k magnetometer component) in nano teslas
    pub k: f64,
}

pub type ObservationMatrix = Vec<ObservationVector>;
pub type PredictionMatrix = Vec<PredictionVector>;

pub struct SECS {
    /// The latitude, longiutde, and radius of the divergence free (df) SEC locations.
    secs_locs: Vec<GeographicalPoint>,
    /// Storage of the scaling factors (amplitudes) for SECs for the last fit.
    pub sec_amps: Option<DVector<f64>>,
    /// Storage of the variance of the scaling factors for SECs for the last fit.
    pub sec_amps_var: Option<DVector<f64>>,

    // Cache fields for transfer function calculation
    _obs_loc_cache: Option<Vec<GeographicalPoint>>,
    _t_obs_flat_cache: Option<DMatrix<f64>>,
}

impl SECS {
    pub fn new(secs_locs: Vec<GeographicalPoint>) -> Self {
        SECS {
            secs_locs,
            sec_amps: None,
            sec_amps_var: None,
            _obs_loc_cache: None,
            _t_obs_flat_cache: None,
        }
    }

    /// Calculate the T transfer matrix (magnetic field from unit currents).
    /// This is analogous to _calc_T in Python, but simplified for df-only
    /// and reshaped.
    fn _calc_t_obs_flat(&self, obs_locs: &[GeographicalPoint]) -> DMatrix<f64> {
        let nobs = obs_locs.len();
        let current_nsec = self.secs_locs.len();

        if nobs == 0 || current_nsec == 0 {
            return DMatrix::zeros(nobs * 3, current_nsec);
        }

        // t_matrix_3d has shape (nobs, 3, nsec)
        let t_matrix_3d = t_df(obs_locs, &self.secs_locs);
        let n_flat_obs = nobs * 3;

        // Reshape t_matrix_3d into DMatrix<f64> of shape (n_flat_obs, nsec)
        // DMatrix is column-major. from_fn fills (0,0), (1,0)...(R-1,0), (0,1)...
        // t_matrix_3d[obs_idx][comp_idx][sec_idx]
        // r_flat = obs_idx * 3 + comp_idx  => obs_idx = r_flat / 3
        //                                 => comp_idx = r_flat % 3
        // c_sec = sec_idx
        DMatrix::from_fn(n_flat_obs, current_nsec, |r_flat, c_sec| {
            t_matrix_3d[r_flat / 3][r_flat % 3][c_sec]
        })
    }

    /// Fits the SECS to the given observations.
    ///
    /// Given a number of observation locations and measurements,
    /// this function fits the SEC system to them. It uses singular
    /// value decomposition (SVD) to fit the SEC amplitudes with the
    /// `epsilon_reg` parameter used to regularize the solution.
    ///
    /// Parameters
    /// ----------
    /// obs : Vec<ObservationVector>
    ///     Contains latitude, longitude, radius, and B-field components (i, j, k)
    ///     of the observation locations. This represents a single time snapshot.
    ///
    /// epsilon_reg : f64
    ///     Value used to regularize/smooth the SECS amplitudes. Singular values `s`
    ///     are filtered such that `s < epsilon_reg * s_max` are effectively treated
    ///     as zero in the pseudo-inverse. Corresponds to 'relative' mode in Python.
    ///     Typically between 0 and 1. A higher number means more regularization.
    ///
    /// Returns
    /// -------
    /// &mut Self
    ///     The SECS instance with updated `sec_amps` and `sec_amps_var`.
    ///
    /// Panics
    /// ------
    /// Panics if SVD computation or pseudo-inverse fails (e.g., due to ill-conditioned matrix
    /// or an extreme `epsilon_reg` value).
    pub fn fit(
        &mut self,
        obs: &[ObservationVector],
        epsilon_reg: f64,
    ) -> Result<&mut Self, String> {
        let nobs = obs.len();
        let current_nsec = self.secs_locs.len();

        let mut current_obs_loc_vec: Vec<GeographicalPoint> = Vec::with_capacity(nobs);
        let n_flat_obs = nobs * 3;
        let mut obs_b_values: Vec<f64> = Vec::with_capacity(n_flat_obs);
        let mut obs_std_values: Vec<f64> = Vec::with_capacity(n_flat_obs);

        for ob_item in obs {
            current_obs_loc_vec.push(GeographicalPoint {
                latitude: ob_item.lat,
                longitude: ob_item.lon,
                altitude: ob_item.alt,
            });
            obs_b_values.push(ob_item.i);
            obs_b_values.push(ob_item.j);
            obs_b_values.push(ob_item.k);

            obs_std_values.push(1.0);
            obs_std_values.push(1.0);
            obs_std_values.push(1.0);
        }

        let obs_b_dvector = DVector::from_vec(obs_b_values);
        let obs_std_dvector = DVector::from_vec(obs_std_values);

        let t_obs_flat: DMatrix<f64>;
        match &self._obs_loc_cache {
            Some(cached_locs) if *cached_locs == current_obs_loc_vec => {
                t_obs_flat = self
                    ._t_obs_flat_cache
                    .as_ref()
                    .expect("Cache inconsistency: obs_loc cached but not t_obs_flat")
                    .clone();
            }
            _ => {
                let new_t_obs_flat = self._calc_t_obs_flat(&current_obs_loc_vec);
                self._obs_loc_cache = Some(current_obs_loc_vec);
                self._t_obs_flat_cache = Some(new_t_obs_flat.clone());
                t_obs_flat = new_t_obs_flat;
            }
        }

        let mut weighted_t_data = Vec::with_capacity(n_flat_obs * current_nsec);
        for c_idx in 0..current_nsec {
            for r_idx in 0..n_flat_obs {
                weighted_t_data.push(t_obs_flat[(r_idx, c_idx)] / obs_std_dvector[r_idx]);
            }
        }
        let weighted_t = DMatrix::from_vec(n_flat_obs, current_nsec, weighted_t_data);

        // --- SVD and Pseudo-inverse construction to match Python's "relative" mode ---
        let svd_weighted_t = weighted_t.clone().svd(true, true);

        let u_matrix = svd_weighted_t
            .u
            .as_ref()
            .ok_or_else(|| "SVD U factor missing".to_string())?;
        let v_t_matrix = svd_weighted_t
            .v_t
            .as_ref()
            .ok_or_else(|| "SVD V_t factor missing".to_string())?;
        let singular_values_vec = &svd_weighted_t.singular_values;

        let pinv_weighted_t: DMatrix<f64>;

        if singular_values_vec.is_empty() {
            // This case implies weighted_t has a zero dimension (e.g., 0 observations or 0 SECs after flattening)
            // The pseudo-inverse of an Mx0 or 0xN matrix is an 0xM or Nx0 zero matrix.
            pinv_weighted_t = DMatrix::zeros(weighted_t.ncols(), weighted_t.nrows());
        } else {
            let s_max = singular_values_vec.amax();

            // Python: valid = S >= epsilon * S.max()
            // W = 1.0 / S (where S is S[valid])
            // Effectively, if s_i < epsilon * S.max(), its inverse is treated as 0.
            let actual_threshold = epsilon_reg * s_max;

            let mut s_inv_filtered_values = DVector::zeros(singular_values_vec.len());
            for i in 0..singular_values_vec.len() {
                if singular_values_vec[i] >= actual_threshold {
                    // If singular_values_vec[i] is zero here, it means actual_threshold is also zero.
                    // 1.0 / 0.0 results in f64::INFINITY, matching Python's behavior.
                    s_inv_filtered_values[i] = 1.0 / singular_values_vec[i];
                }
                // Else, it remains 0.0 from initialization (singular value's inverse is cut)
            }

            let s_inv_filtered_diag_matrix = DMatrix::from_diagonal(&s_inv_filtered_values);

            // Pseudo-inverse = V * S_inv_filtered_diag * U^T
            // svd_weighted_t.v_t is V^T, so svd_weighted_t.v_t.transpose() is V
            pinv_weighted_t =
                v_t_matrix.transpose() * s_inv_filtered_diag_matrix * u_matrix.transpose();
        }

        // --- End of SVD and Pseudo-inverse ---

        let b_weighted = obs_b_dvector.component_div(&obs_std_dvector);
        let fitted_sec_amps = &pinv_weighted_t * b_weighted;
        self.sec_amps = Some(fitted_sec_amps);

        let mut sec_amps_var_values: Vec<f64> = Vec::with_capacity(current_nsec);
        if n_flat_obs > 0 {
            // only calculate variance if there are observations
            for i_sec in 0..current_nsec {
                let mut sum_sq_val = 0.0;
                for j_obs_comp in 0..n_flat_obs {
                    let val = pinv_weighted_t[(i_sec, j_obs_comp)] * obs_std_dvector[j_obs_comp];
                    sum_sq_val += val * val;
                }
                sec_amps_var_values.push(sum_sq_val);
            }
        } else {
            // No observations, variance is undefined or zero
            for _ in 0..current_nsec {
                sec_amps_var_values.push(0.0); // Or perhaps f64::NAN or Option<f64>
            }
        }
        self.sec_amps_var = Some(DVector::from_vec(sec_amps_var_values));

        Ok(self)
    }

    /// Calculate the predicted magnetic field (B).
    ///
    /// After a set of observations has been fit to this system using `.fit()`,
    /// this function predicts the magnetic fields at any other location based
    /// on the fitted SEC amplitudes. This implementation only considers
    /// divergence-free SECs.
    ///
    /// Parameters
    /// ----------
    /// pred_locs: &[GeographicalPoint]
    ///     A slice of geographical points where the predictions are desired.
    ///
    /// Returns
    /// -------
    /// Result<Vec<PredictionVector>, String>
    ///     A vector of predicted B-fields (i=Bx, j=By, k=Bz components) at each `pred_loc`,
    ///     or an error string if the model hasn't been fitted or dimensions mismatch.
    pub fn predict(&self, pred_locs: &[GeographicalPoint]) -> Result<PredictionMatrix, String> {
        let npred = pred_locs.len();
        let current_nsec = self.secs_locs.len();

        if npred == 0 {
            return Ok(Vec::new()); // No locations to predict at.
        }

        // Check if fit() has been called and amps are available and valid
        let sec_amps_vec = match &self.sec_amps {
            Some(amps) => {
                if amps.len() != current_nsec {
                    return Err(format!(
                         "Fitted amplitude count ({}) does not match current SEC count ({}). Refit the model.",
                         amps.len(), current_nsec
                     ));
                }
                if current_nsec == 0 && amps.len() == 0 {
                    // Special case: 0 SECs fitted, prediction is zero.
                    let zero_predictions = pred_locs
                        .iter()
                        .map(|loc| PredictionVector {
                            lon: loc.longitude,
                            lat: loc.latitude,
                            i: 0.0,
                            j: 0.0,
                            k: 0.0,
                        })
                        .collect();
                    return Ok(zero_predictions);
                }
                amps // Return the valid amps vector
            }
            None => {
                return Err(String::from(
                    "SECS model has not been fitted yet. Call .fit() before .predict_b().",
                ));
            }
        };

        // At this point, current_nsec > 0 and sec_amps_vec.len() == current_nsec

        // Calculate the prediction transfer matrix T_pred = (npred, 3, nsec)
        // For DF only, this comes directly from t_df
        // t_pred_3d[p][k][s] is influence of sec 's' on component 'k' at point 'p'
        // TODO: Implement caching here if desired
        let t_pred_3d = t_df(pred_locs, &self.secs_locs);

        // --- Core Calculation ---
        // predicted_B[p][k] = sum_s (sec_amps_vec[s] * t_pred_3d[p][k][s])

        let mut predictions: Vec<PredictionVector> = Vec::with_capacity(npred);

        for p in 0..npred {
            // Basic dimension check on t_pred_3d before indexing
            if p >= t_pred_3d.len()
                || t_pred_3d[p].len() != 3
                || (current_nsec > 0 && t_pred_3d[p][0].len() != current_nsec)
            {
                return Err(format!("Internal error: Unexpected dimension of T_pred matrix at prediction index p={}. NPred={}, Nsec={}. T_pred[p].len={}, T_pred[p][0].len={}",
                     p, npred, current_nsec, t_pred_3d.get(p).map_or(0, |v| v.len()), t_pred_3d.get(p).and_then(|v| v.get(0)).map_or(0, |v| v.len())
                 ));
            }

            // Use dot product for the sum: B_k = T_k . amps
            // where T_k is the k-th row vector [T(p,k,0), T(p,k,1), ..., T(p,k,nsec-1)]
            let bx = DVector::from_row_slice(&t_pred_3d[p][0]).dot(sec_amps_vec);
            let by = DVector::from_row_slice(&t_pred_3d[p][1]).dot(sec_amps_vec);
            let bz = DVector::from_row_slice(&t_pred_3d[p][2]).dot(sec_amps_vec);

            predictions.push(PredictionVector {
                lon: pred_locs[p].longitude,
                lat: pred_locs[p].latitude,
                i: bx,
                j: by,
                k: bz,
            });
        }

        Ok(predictions)
    }
}
