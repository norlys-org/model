use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{geo::GeographicalPoint, svd::svd, t_df::t_df};

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
    sec_locs: Vec<GeographicalPoint>,
    /// Storage of the scaling factors (amplitudes) for SECs for the last fit.
    sec_amps: Option<Array2<f64>>,

    // Cache fields for transfer function calculation
    obs_locs_cache: Vec<GeographicalPoint>,
    t_obs_flat_cache: Option<Array2<f64>>,
}

impl SECS {
    pub fn new(sec_locs: Vec<GeographicalPoint>) -> Self {
        SECS {
            sec_locs,
            sec_amps: None,
            obs_locs_cache: vec![],
            t_obs_flat_cache: None,
        }
    }

    pub fn fit(&mut self, obs: &[ObservationVector], epsilon: f64) {
        let obs_b: Array2<f64> = Array2::from_shape_vec(
            (1, obs.len() * 3),
            obs.iter()
                .flat_map(|obs| vec![obs.i, obs.j, obs.k])
                .collect(),
        )
        .unwrap();

        let obs_locs: Vec<GeographicalPoint> = obs
            .iter()
            .map(|obs| GeographicalPoint::new(obs.lat, obs.lon, obs.alt))
            .collect();

        // Check if transfer matrix has already been computed in this instance
        if obs_locs != self.obs_locs_cache {
            let t = t_df(&obs_locs, &self.sec_locs);
            self.t_obs_flat_cache = Some(
                t.clone()
                    .into_shape((t.len() / self.sec_locs.len(), self.sec_locs.len()))
                    .unwrap(),
            );
            self.obs_locs_cache = obs_locs;
        }

        // SVD
        let vwu: Array2<f64> = svd(self.t_obs_flat_cache.as_ref().unwrap(), epsilon);
        self.sec_amps = Some(obs_b.dot(&vwu.t()));
    }

    // pub fn predict(&self, pred_locs: &[GeographicalPoint]) -> Result<PredictionMatrix, String> {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fit_few_points() {
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

        let sec_amps_expected: f64 = -1.803280385158305e+14;
        assert_relative_eq!(
            secs.sec_amps.as_ref().unwrap()[[0, 0]],
            sec_amps_expected,
            max_relative = 1e-15
        );
    }
}
