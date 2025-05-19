use nalgebra::{DMatrix, DVector};
use ndarray::Array;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_sys::console::assert;

use crate::geo::GeographicalPoint;

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

    pub fn fit(&mut self, obs: &[ObservationVector], epsilon_reg: f64) {
        let n_times = obs.len();
        let obs_b = Array::from_shape_vec(
            (1, obs.len() * 3),
            obs.iter()
                .flat_map(|obs| vec![obs.i, obs.j, obs.k])
                .collect(),
        )
        .unwrap();
    }

    // pub fn predict(&self, pred_locs: &[GeographicalPoint]) -> Result<PredictionMatrix, String> {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit() {
        let mut secs = SECS::new(vec![GeographicalPoint::new(10.0, 20.0, 0.0)]);
        secs.fit(
            &[
                ObservationVector {
                    lon: 50.0,
                    lat: 40.0,
                    i: 1.0,
                    j: 3.0,
                    k: 5.0,
                    alt: 0.0,
                },
                ObservationVector {
                    lon: 60.0,
                    lat: 50.0,
                    i: 2.0,
                    j: 4.0,
                    k: 6.0,
                    alt: 0.0,
                },
            ],
            0.05,
        );
    }
}
