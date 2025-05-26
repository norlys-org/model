use ndarray::{Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

use candid::CandidType;

use crate::{geo::GeographicalPoint, svd::svd, t_df::t_df};

// #[wasm_bindgen]
#[derive(CandidType, Serialize, Deserialize, Debug, Clone, Copy)]
pub struct ObservationVector {
    /// The longitude in degrees.
    pub lon: f32,
    /// The latitude in degrees.
    pub lat: f32,
    // i vector (usually x magnetometer component) in nano teslas
    pub i: f32,
    // j vector (usually y magnetometer component) in nano teslas
    pub j: f32,
    // k vector (usually k magnetometer component) in nano teslas
    pub k: f32,
}

// #[wasm_bindgen]
#[derive(CandidType, Serialize, Deserialize, Debug, Clone, Copy)]
pub struct PredictionVector {
    /// The longitude in degrees.
    pub lon: f32,
    /// The latitude in degrees.
    pub lat: f32,
    // i vector (usually x magnetometer component) in nano teslas
    pub i: f32,
    // j vector (usually y magnetometer component) in nano teslas
    pub j: f32,
    // k vector (usually k magnetometer component) in nano teslas
    pub k: f32,
}

#[derive(Debug)]
pub struct SECS {
    /// The latitude and longiutde of the divergence free (df) SEC locations.
    sec_locs: Vec<GeographicalPoint>,
    /// The altitude in meters above the surface of the earth at which poles are located
    sec_locs_altitude: f32,
    /// Storage of the scaling factors (amplitudes) for SECs for the last fit.
    pub sec_amps: Option<Array2<f32>>,

    // Cache fields for transfer function calculation
    pub obs_locs_cache: Vec<GeographicalPoint>,
    pub t_obs_flat_cache: Option<Array2<f32>>,
    /// The latitude, longiutde, and radius of the prediction locations.
    pred_locs_cache: Vec<GeographicalPoint>,
    t_pred_cache: Option<Array3<f32>>,
}

impl SECS {
    pub fn new(sec_locs: Vec<GeographicalPoint>, sec_locs_altitude: f32) -> Self {
        SECS {
            sec_locs,
            sec_locs_altitude,
            sec_amps: None,
            obs_locs_cache: vec![],
            t_obs_flat_cache: None,
            pred_locs_cache: vec![],
            t_pred_cache: None,
        }
    }

    pub fn fit(&mut self, obs: &[ObservationVector], obs_altitude: f32, epsilon: f32) {
        let obs_b: Array2<f32> = Array2::from_shape_vec(
            (1, obs.len() * 3),
            obs.iter()
                .flat_map(|obs| vec![obs.i, obs.j, obs.k])
                .collect(),
        )
        .unwrap();

        let obs_locs: Vec<GeographicalPoint> = obs
            .iter()
            .map(|obs| GeographicalPoint::new(obs.lat, obs.lon))
            .collect();

        // Check if transfer matrix has already been computed in this instance
        if obs_locs != self.obs_locs_cache {
            let t = t_df(
                &obs_locs,
                obs_altitude,
                &self.sec_locs,
                self.sec_locs_altitude,
            );
            self.t_obs_flat_cache = Some(
                t.clone()
                    .into_shape((t.len() / self.sec_locs.len(), self.sec_locs.len()))
                    .unwrap(),
            );
            self.obs_locs_cache = obs_locs;
        }

        // SVD
        let vwu: Array2<f32> = svd(self.t_obs_flat_cache.as_ref().unwrap(), epsilon);
        self.sec_amps = Some(obs_b.dot(&vwu.t()));
    }

    pub fn predict(
        &mut self,
        pred_locs: &[GeographicalPoint],
        pred_altitude: f32,
    ) -> Vec<PredictionVector> {
        // check if transfer matrix was already computed for these locations
        if pred_locs != self.pred_locs_cache {
            self.t_pred_cache = Some(t_df(
                pred_locs,
                pred_altitude,
                &self.sec_locs,
                self.sec_locs_altitude,
            ));
            self.pred_locs_cache = pred_locs.to_vec();
        }

        let amps = &self.sec_amps.as_ref().unwrap().index_axis(Axis(1), 0);
        let t_pred = &self.t_pred_cache.as_ref().unwrap().index_axis(Axis(2), 0);
        let pred = amps * t_pred;

        pred_locs
            .iter()
            .enumerate()
            .map(|(i, loc)| PredictionVector {
                lon: loc.lon,
                lat: loc.lat,
                i: pred[[i, 0]],
                j: pred[[i, 1]],
                k: pred[[i, 2]],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fit_few_points() {
        let mut secs = SECS::new(vec![GeographicalPoint::new(10.0, 20.0)], 0.0);

        secs.fit(
            &[
                ObservationVector {
                    lon: 50.0,
                    lat: 40.0,
                    i: 1.0,
                    j: 3.0,
                    k: 5.0,
                },
                ObservationVector {
                    lon: 60.0,
                    lat: 50.0,
                    i: 2.0,
                    j: 4.0,
                    k: 6.0,
                },
            ],
            0.0,
            0.05,
        );

        let sec_amps_expected: f32 = -1.803280385158305e+14;
        assert_relative_eq!(
            secs.sec_amps.as_ref().unwrap()[[0, 0]],
            sec_amps_expected,
            max_relative = 1e-15
        );
    }

    #[test]
    fn test_predict_few_points() {
        let mut secs = SECS::new(vec![GeographicalPoint::new(10.0, 20.0)], 0.0);

        secs.fit(
            &[
                ObservationVector {
                    lon: 50.0,
                    lat: 40.0,
                    i: 1.0,
                    j: 3.0,
                    k: 5.0,
                },
                ObservationVector {
                    lon: 60.0,
                    lat: 50.0,
                    i: 2.0,
                    j: 4.0,
                    k: 6.0,
                },
                ObservationVector {
                    lon: 70.0,
                    lat: 60.0,
                    i: 6.0,
                    j: 8.0,
                    k: 10.0,
                },
                ObservationVector {
                    lon: 80.0,
                    lat: 70.0,
                    i: 7.0,
                    j: 9.0,
                    k: 11.0,
                },
            ],
            0.0,
            0.05,
        );

        let expected: Vec<PredictionVector> = vec![
            PredictionVector {
                lon: 50.0,
                lat: 40.0,
                i: -2.9721985560476623,
                j: -3.524936055971201,
                k: 2.1625944014276293,
            },
            PredictionVector {
                lon: 60.0,
                lat: 50.0,
                i: -1.9295344150381564,
                j: -2.619481700089664,
                k: 0.7058854169668979,
            },
            PredictionVector {
                lon: 70.0,
                lat: 60.0,
                i: -1.3007860482981708,
                j: -2.126884161397398,
                k: -0.1073352052343382,
            },
            PredictionVector {
                lon: 80.0,
                lat: 70.0,
                i: -0.8729398495707357,
                j: -1.8459492569556268,
                k: -0.5871575186673494,
            },
        ];
        let pred = secs.predict(
            &[
                GeographicalPoint::new(40.0, 50.0),
                GeographicalPoint::new(50.0, 60.0),
                GeographicalPoint::new(60.0, 70.0),
                GeographicalPoint::new(70.0, 80.0),
            ],
            110e3,
        );

        for (actual, expected) in pred.iter().zip(expected.iter()) {
            assert_relative_eq!(actual.lon, expected.lon, max_relative = 1e-15);
            assert_relative_eq!(actual.lat, expected.lat, max_relative = 1e-15);
            assert_relative_eq!(actual.i, expected.i, max_relative = 1e-15);
            assert_relative_eq!(actual.j, expected.j, max_relative = 1e-15);
            assert_relative_eq!(actual.k, expected.k, max_relative = 1e-15);
        }
    }
}
