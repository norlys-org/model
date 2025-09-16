use geo::geographical_grid;
use ic_cdk::caller;
use model::{ObservationVector, PredictionVector, SECS};
use overlays::{IntoScores, Overlays, ScoreVector};

use std::cell::RefCell;
use std::collections::HashSet;

use candid::Principal;

mod geo;
mod model;
mod overlays;
mod sphere;
mod svd;
mod t_df;

#[derive(Clone)]
struct PredictionStorage {
    /// Predictions from the absolute observations
    abs: Option<Vec<ScoreVector>>,
    /// Predictions from the derivative
    drv: Option<Vec<ScoreVector>>,
}

// MARK: Storage
// storage in heap memory since it is not sensitive and can suffer from a refresh on install
thread_local! {
    static STORED_SECS: RefCell<Option<SECS>> = RefCell::new(None);
    static PREDICTIONS: RefCell<PredictionStorage> = RefCell::new(PredictionStorage { abs: None, drv: None });
    static AUTHORIZED_USERS: RefCell<HashSet<Principal>> = RefCell::new(HashSet::new());
}

impl PredictionStorage {
    pub fn load() -> Self {
        PREDICTIONS.with(|p| p.borrow().clone())
    }

    /// Store the given data either in `.abs` or `.drv` depending on `is_derivative`
    pub fn store(data: Vec<ScoreVector>, is_derivative: bool) {
        let buf = PREDICTIONS.with(|p| p.borrow().clone());

        PREDICTIONS.with(|p| {
            if is_derivative {
                *p.borrow_mut() = PredictionStorage {
                    abs: buf.abs,
                    drv: Some(data),
                };
            } else {
                *p.borrow_mut() = PredictionStorage {
                    drv: buf.drv,
                    abs: Some(data),
                };
            }
        });
    }
}

impl SECS {
    pub fn load() -> Self {
        STORED_SECS.with(|p| p.borrow().clone()).unwrap_or_default()
    }

    pub fn store(self) {
        STORED_SECS.with(|p| {
            *p.borrow_mut() = Some(self);
        });
    }

    pub fn clear() {
        STORED_SECS.with(|p| {
            *p.borrow_mut() = None;
        });
    }
}

// MARK: Authorization calls

fn is_authorized() -> bool {
    let caller = caller();

    if caller == Principal::anonymous() {
        return false;
    }

    AUTHORIZED_USERS.with(|users| users.borrow().contains(&caller))
}

fn require_authorization() {
    if !is_authorized() {
        ic_cdk::trap("Access denied: caller not authorized");
    }
}

// prefix a_ for authorization

#[ic_cdk::update]
pub fn a_add_authorized_user(user_principal: Principal) {
    require_authorization();
    AUTHORIZED_USERS.with(|users| {
        users.borrow_mut().insert(user_principal);
    });
}

#[ic_cdk::update]
pub fn a_remove_authorized_user(user_principal: Principal) {
    require_authorization();
    AUTHORIZED_USERS.with(|users| {
        users.borrow_mut().remove(&user_principal);
    });
}

#[ic_cdk::query]
pub fn a_list_authorized_users() -> Vec<Principal> {
    require_authorization();
    AUTHORIZED_USERS.with(|users| users.borrow().iter().cloned().collect())
}

// Initialize function to add the first authorized user (call this after deployment)
#[ic_cdk::update]
pub fn a_initialize_authorized_user(user_principal: Principal) {
    // only allow if no users are authorized yet
    AUTHORIZED_USERS.with(|users| {
        if users.borrow().is_empty() {
            users.borrow_mut().insert(user_principal);
        } else {
            ic_cdk::trap("Authorized users already exist");
        }
    });
}

// MARK: Model calls
// Requiring authorization on all update calls since we rely on the memory set after each
// prefix m_ for model

// Returns whether fiting predictions is neccesary
#[ic_cdk::update]
pub fn m_fit_obs(obs: Vec<ObservationVector>) -> bool {
    require_authorization();

    let mut secs: SECS = if STORED_SECS.with(|storage| storage.borrow().is_some()) {
        SECS::load()
    } else {
        SECS::new(geographical_grid(45.0..85.0, 50, -170.0..35.0, 50), 110e3)
    };

    let obs_zero_k: Vec<ObservationVector> = obs
        .into_iter()
        .map(|mut o| {
            o.k = 0.0;
            o
        })
        .collect();
    secs.fit(&obs_zero_k, 0.0, 0.1);
    let needs_pred_fit = secs.t_pred_cache.is_none();
    secs.store();
    needs_pred_fit
}

#[ic_cdk::update]
pub fn m_fit_pred() {
    require_authorization();

    let pred_grid = geographical_grid(45.0..85.0, 37, -180.0..179.0, 130);
    let mut secs = SECS::load();
    secs.calc_t_pred(&pred_grid, 110e3);
    secs.store();
}

#[ic_cdk::update]
pub fn m_predict(is_derivative: bool) -> Vec<PredictionVector> {
    require_authorization();

    let secs = SECS::load();
    let raw_prediction: Vec<PredictionVector> = secs.predict();
    let prediction: Vec<ScoreVector> = if is_derivative {
        raw_prediction.clone().into_derivative_scores()
    } else {
        raw_prediction.clone().into_scores()
    };

    PredictionStorage::store(prediction.ponderate_auroral_zone(), is_derivative);

    raw_prediction
}

#[ic_cdk::update]
pub fn m_scores() -> Vec<u16> {
    require_authorization();

    let predictions = PredictionStorage::load();
    match predictions.drv {
        None => predictions.abs.unwrap_or_default().encode(),
        Some(drv) => predictions
            .abs
            .unwrap_or_default()
            .max_score_vectors(drv)
            .encode(),
    }
}

ic_cdk::export_candid!();

#[cfg(feature = "canbench-rs")]
mod benches {
    use super::*;
    use canbench_rs::bench;

    #[bench]
    fn calc_t() {
        fit_obs(vec![ObservationVector {
            lon: 1.0,
            lat: 1.0,
            i: 1.0,
            j: 1.0,
            k: 1.0,
        }]);
        fit_pred();
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::Array2;

    use super::*;
    use crate::{geo::GeographicalPoint, model::PredictionVector};
    use serde_json;
    use std::fs;

    // #[test]
    // fn test_large_infer() {
    //     let mut secs = SECS::new(geographical_grid(45.0..85.0, 37, -170.0..35.0, 74), 0.0);
    //     secs.fit(
    //         &[ObservationVector {
    //             lon: 10.0,
    //             lat: 69.0,
    //             i: 500.0,
    //             j: 100.0,
    //             k: 100.0,
    //         }],
    //         0.0,
    //         0.05,
    //     );
    //
    //     let pred_grid = geographical_grid(45.0..85.0, 37, -180.0..179.0, 130);
    //     secs.calc_t_pred(&pred_grid, 110e3);
    //     let pred = secs.predict();
    //
    //     println!("{:?}", pred);
    // }
    //

    #[test]
    fn test_ic() {
        let obs = vec![
            ObservationVector {
                lon: 11.95,
                lat: 78.92,
                i: 4.0,
                j: 2.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 15.82,
                lat: 78.2,
                i: -34.0,
                j: -32.0,
                k: 2.0,
            },
            ObservationVector {
                lon: 25.01,
                lat: 76.51,
                i: -61.0,
                j: 4.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 19.2,
                lat: 74.5,
                i: -123.0,
                j: -30.0,
                k: 1.0,
            },
            ObservationVector {
                lon: 351.3,
                lat: 70.9,
                i: 5.0,
                j: -25.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 25.79,
                lat: 71.09,
                i: 17.0,
                j: -24.0,
                k: 1.0,
            },
            ObservationVector {
                lon: 12.1,
                lat: 67.53,
                i: 74.0,
                j: -21.0,
                k: 4.0,
            },
            ObservationVector {
                lon: 22.22,
                lat: 70.54,
                i: 17.0,
                j: -26.0,
                k: 1.0,
            },
            ObservationVector {
                lon: 27.01,
                lat: 69.76,
                i: 10.66666666666606,
                j: -25.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 18.94,
                lat: 69.66,
                i: 38.0,
                j: -31.0,
                k: 3.0,
            },
            ObservationVector {
                lon: 26.63,
                lat: 67.37,
                i: 56.089285714286234,
                j: -11.428571428571558,
                k: 0.0,
            },
            ObservationVector {
                lon: 23.7,
                lat: 69.46,
                i: 19.33333333333394,
                j: -24.888888888888914,
                k: 6.0,
            },
            ObservationVector {
                lon: 16.03,
                lat: 69.3,
                i: 53.0,
                j: -35.0,
                k: 2.0,
            },
            ObservationVector {
                lon: 16.98,
                lat: 66.4,
                i: 47.0,
                j: -18.0,
                k: 1.0,
            },
            ObservationVector {
                lon: 27.29,
                lat: 68.56,
                i: 24.0,
                j: -23.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 24.08,
                lat: 66.9,
                i: 46.33333333333394,
                j: -12.888888888888687,
                k: 0.0,
            },
            ObservationVector {
                lon: 12.5,
                lat: 66.11,
                i: 82.0,
                j: 1697.0,
                k: 2.0,
            },
            ObservationVector {
                lon: 26.25,
                lat: 65.54,
                i: 31.0,
                j: -12.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 10.98,
                lat: 64.94,
                i: 27.0,
                j: -16.0,
                k: 1.0,
            },
            ObservationVector {
                lon: 27.23,
                lat: 64.52,
                i: 17.83333333333212,
                j: -10.41666666666697,
                k: 0.0,
            },
            ObservationVector {
                lon: 30.97,
                lat: 62.77,
                i: 12.66666666666606,
                j: -7.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 26.6,
                lat: 62.25,
                i: 11.33333333333394,
                j: -12.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 9.11,
                lat: 62.07,
                i: 15.0,
                j: -17.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 4.84,
                lat: 61.08,
                i: 13.0,
                j: -17.0,
                k: 1.0,
            },
            ObservationVector {
                lon: 10.75,
                lat: 60.21,
                i: 51.0,
                j: -10.0,
                k: 1.0,
            },
            ObservationVector {
                lon: 24.65,
                lat: 60.5,
                i: 13.33333333333394,
                j: -12.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 5.24,
                lat: 59.21,
                i: 12.0,
                j: -18.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 26.46,
                lat: 58.26,
                i: 17.70833333333212,
                j: -14.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 290.77,
                lat: 77.47,
                i: -25.0,
                j: 8.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 321.7,
                lat: 72.3,
                i: -34.0,
                j: 29.0,
                k: -1.0,
            },
            ObservationVector {
                lon: 306.47,
                lat: 69.25,
                i: -1.0,
                j: 26.0,
                k: -2.0,
            },
            ObservationVector {
                lon: 314.56,
                lat: 61.16,
                i: -6.0,
                j: 20.66666666666697,
                k: 0.0,
            },
            ObservationVector {
                lon: 254.763,
                lat: 40.137,
                i: 13.0,
                j: -15.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 203.378,
                lat: 71.322,
                i: 5.0,
                j: 18.0,
                k: 5.0,
            },
            ObservationVector {
                lon: 242.878,
                lat: 48.265,
                i: -2.0,
                j: -9.0,
                k: -1.0,
            },
            ObservationVector {
                lon: 224.675,
                lat: 57.058,
                i: -11.0,
                j: 14.0,
                k: -2.0,
            },
            ObservationVector {
                lon: 297.647,
                lat: 82.497,
                i: -27.4949951171875,
                j: -24.8551025390625,
                k: -6.679931640625,
            },
            ObservationVector {
                lon: 264.0,
                lat: 64.3,
                i: -4.070068359375,
                j: -17.900009155273438,
                k: 0.22998046875,
            },
            ObservationVector {
                lon: 262.9,
                lat: 49.6,
                i: -12.27490234375,
                j: -20.7900390625,
                k: 2.1796875,
            },
            ObservationVector {
                lon: 255.0,
                lat: 69.2,
                i: -36.534912109375,
                j: -20.865020751953125,
                k: 3.02978515625,
            },
            ObservationVector {
                lon: 274.1,
                lat: 80.0,
                i: 66.070068359375,
                j: -163.27001953125,
                k: -1.730224609375,
            },
            ObservationVector {
                lon: 265.9,
                lat: 58.8,
                i: -4.642578125,
                j: -28.399993896484375,
                k: -0.5,
            },
            ObservationVector {
                lon: 291.5,
                lat: 63.8,
                i: 51.789794921875,
                j: -14.5950927734375,
                k: -1.8896484375,
            },
            ObservationVector {
                lon: 246.7,
                lat: 54.6,
                i: -9.5390625,
                j: -1.630126953125,
                k: 0.7392578125,
            },
            ObservationVector {
                lon: 284.5,
                lat: 45.4,
                i: -5.079345703125,
                j: -29.310302734375,
                k: -1.23046875,
            },
            ObservationVector {
                lon: 265.1,
                lat: 74.7,
                i: -41.2532324598551,
                j: -140.4658432006836,
                k: 17.650146484375,
            },
            ObservationVector {
                lon: 245.5,
                lat: 62.4,
                i: -4.810546875,
                j: -32.288499124005284,
                k: 0.080078125,
            },
            ObservationVector {
                lon: 307.3,
                lat: 47.6,
                i: 0.5225824004955939,
                j: -31.922744140625582,
                k: 0.470703125,
            },
            ObservationVector {
                lon: 236.6,
                lat: 48.6,
                i: 19.917784737219336,
                j: 22.48283650507892,
                k: 0.41015625,
            },
            ObservationVector {
                lon: -147.447,
                lat: 65.136,
                i: 46.95000000000073,
                j: 23.0,
                k: -1.090909090909091,
            },
            ObservationVector {
                lon: -141.205,
                lat: 64.786,
                i: 37.0,
                j: 5.0,
                k: 3.5999999999999996,
            },
            ObservationVector {
                lon: -149.592,
                lat: 68.627,
                i: -23.183333333333394,
                j: 139.14166666666665,
                k: -3.272727272727273,
            },
            ObservationVector {
                lon: 18.823,
                lat: 68.358,
                i: 83.90833333333285,
                j: -32.11666666666679,
                k: -2.0,
            },
            ObservationVector {
                lon: 58.567,
                lat: 56.433,
                i: 37.65476190476147,
                j: -1.5714285714284415,
                k: -2.0,
            },
            ObservationVector {
                lon: 20.789,
                lat: 51.836,
                i: 28.458408679929562,
                j: -2.423146473779525,
                k: -2.0,
            },
            ObservationVector {
                lon: 2.26,
                lat: 48.025,
                i: 26.0,
                j: -25.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 4.6,
                lat: 50.1,
                i: 17.0,
                j: -17.0,
                k: 1.0,
            },
            ObservationVector {
                lon: 343.559,
                lat: 28.321,
                i: 16.16666666666788,
                j: -25.0,
                k: 1.0,
            },
            ObservationVector {
                lon: 18.811,
                lat: 54.603,
                i: 38.0,
                j: 19.0,
                k: -3.0,
            },
            ObservationVector {
                lon: 15.55,
                lat: 77.0,
                i: 56.0,
                j: 17.0,
                k: -2.0,
            },
            ObservationVector {
                lon: 18.748,
                lat: 64.612,
                i: 35.0,
                j: -31.0,
                k: -2.0,
            },
            ObservationVector {
                lon: 5.682,
                lat: 50.298,
                i: 21.0,
                j: -20.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 26.25,
                lat: 44.68,
                i: 30.0,
                j: -16.0,
                k: 1.0,
            },
            ObservationVector {
                lon: 249.27,
                lat: 32.17,
                i: 23.0,
                j: -6.0,
                k: 0.0,
            },
            ObservationVector {
                lon: 17.353,
                lat: 59.903,
                i: 10.0,
                j: -17.0,
                k: -3.0,
            },
        ];

        let mut secs = SECS::new(geographical_grid(45.0..85.0, 50, -170.0..35.0, 50), 110e3);
        let obs_zero_k: Vec<ObservationVector> = obs
            .into_iter()
            .map(|mut o| {
                o.k = 0.0;
                o
            })
            .collect();
        secs.fit(&obs_zero_k, 0.0, 0.1);

        // MARK: Amplitudes
        let json_content =
            fs::read_to_string("resources/sec_amps.json").expect("Failed to read JSON file");
        let expected_vec: Vec<Vec<f64>> =
            serde_json::from_str(&json_content).expect("Failed to parse JSON");

        let rows = expected_vec.len();
        let cols = expected_vec[0].len();
        let flat_expected: Vec<f64> = expected_vec.into_iter().flatten().collect();
        let expected = Array2::from_shape_vec((rows, cols), flat_expected)
            .expect("Failed to create Array2 from JSON data");

        let calculated = secs.sec_amps.as_ref().expect("sec_amps should be Some");

        let calc = calculated.as_slice().unwrap();
        let exp = expected.as_slice().unwrap();
        assert_eq!(calculated.shape(), expected.shape());
        let mut failed = false;
        for i in 0..calc.len().min(10) {
            let rel_err = ((calc[i] - exp[i]) / exp[i]).abs();
            if rel_err > 1e-3 {
                println!(
                    "idx {}: calc={:.6e}, exp={:.6e}, rel_err={:.2e}",
                    i, calc[i], exp[i], rel_err
                );
                failed = true;
            }
        }
        assert!(!failed, "First 10 values already show large error");
        assert_eq!(
            calculated.shape(),
            expected.shape(),
            "Sec ampls dimensions don't match: calculated {:?} vs expected {:?}",
            calculated.shape(),
            expected.shape()
        );
        assert_relative_eq!(calc, exp, max_relative = 1e-2);

        std::fs::write(
            "rust_sec_amps.json",
            serde_json::to_string(&secs.sec_amps).unwrap(),
        )
        .unwrap();

        let pred_grid = geographical_grid(45.0..85.0, 37, -180.0..179.0, 130);
        secs.calc_t_pred(&pred_grid, 110e3);

        let pred: Vec<PredictionVector> = secs.predict();

        std::fs::write("pred.json", serde_json::to_string(&pred).unwrap()).unwrap();
    }

    #[test]
    fn test_infer() {
        let obs_grid = vec![
            GeographicalPoint::new(50.0, 40.0),
            GeographicalPoint::new(50.0, 50.0),
            GeographicalPoint::new(60.0, 40.0),
            GeographicalPoint::new(60.0, 50.0),
        ];

        let pred_grid = vec![
            GeographicalPoint::new(40.0, 50.0),
            GeographicalPoint::new(50.0, 60.0),
            GeographicalPoint::new(60.0, 70.0),
            GeographicalPoint::new(70.0, 80.0),
        ];

        let mut secs = SECS::new(obs_grid, 0.0);

        let obs = vec![
            ObservationVector {
                lon: 50.0,
                lat: 40.0,
                i: 100.0,
                j: 300.0,
                k: 5.0,
            },
            ObservationVector {
                lon: 60.0,
                lat: 50.0,
                i: 200.0,
                j: 400.0,
                k: 6.0,
            },
            ObservationVector {
                lon: 70.0,
                lat: 60.0,
                i: 600.0,
                j: 800.0,
                k: 10.0,
            },
            ObservationVector {
                lon: 80.0,
                lat: 70.0,
                i: 700.0,
                j: 900.0,
                k: 11.0,
            },
        ];

        secs.fit(&obs, 0.0, 0.05);
        secs.calc_t_pred(&pred_grid, 110e3);
        let pred = secs.predict();

        let expected: Vec<PredictionVector> = vec![
            PredictionVector {
                lon: 50.0,
                lat: 40.0,
                i: 257.823724031822,
                j: -257.9989840916952,
                k: 293.0630608681675,
            },
            PredictionVector {
                lon: 60.0,
                lat: 50.0,
                i: 104.39995806060591,
                j: -203.84660351258992,
                k: 168.9489278540623,
            },
            PredictionVector {
                lon: 70.0,
                lat: 60.0,
                i: 10.54058323539585,
                j: -241.81166590941018,
                k: 190.60748617783958,
            },
            PredictionVector {
                lon: 80.0,
                lat: 70.0,
                i: -44.89013986250664,
                j: -183.73983360583236,
                k: 135.6276277064731,
            },
        ];

        for (actual, expected) in pred.iter().zip(expected.iter()) {
            assert_relative_eq!(actual.lon, expected.lon, max_relative = 1e-10);
            assert_relative_eq!(actual.lat, expected.lat, max_relative = 1e-10);
            assert_relative_eq!(actual.i, expected.i, max_relative = 1e-10);
            assert_relative_eq!(actual.j, expected.j, max_relative = 1e-10);
            assert_relative_eq!(actual.k, expected.k, max_relative = 1e-10);
        }
    }
}
