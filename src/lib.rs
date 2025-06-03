use geo::geographical_grid;
use model::{ObservationVector, SECS};

use std::cell::RefCell;

mod geo;
mod model;
mod overlays;
mod sphere;
mod svd;
mod t_df;

// storage in heap memory since it is not sensitive and can suffer from a refresh on install
thread_local! {
    static STORED_SECS: RefCell<Option<SECS>> = RefCell::new(None);
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

#[ic_cdk::update]
pub fn fit_obs(obs: Vec<ObservationVector>) {
    let mut secs: SECS = if STORED_SECS.with(|storage| storage.borrow().is_some()) {
        SECS::load()
    } else {
        SECS::new(geographical_grid(45.0..85.0, 37, -170.0..35.0, 74), 0.0)
    };

    secs.fit(&obs, 0.0, 0.05);
    secs.store();
}

#[ic_cdk::update]
pub fn fit_pred() {
    let pred_grid = geographical_grid(45.0..85.0, 37, -180.0..179.0, 130);
    let mut secs = SECS::load();
    secs.calc_t_pred(&pred_grid, 110e3);
    secs.store();
}

#[ic_cdk::update]
pub fn predict() -> Vec<f64> {
    let mut secs = SECS::load();
    secs.predict().into_iter().map(|v| v.i).collect()
    // .into_scores()
    // .ponderate_auroral_zone()
    // .encode()
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

    use super::*;
    use crate::{geo::GeographicalPoint, model::PredictionVector};

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
