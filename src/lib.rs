use geo::{geographical_grid, GeographicalPoint};
use model::{ObservationVector, PredictionVector, SECS};
use ndarray::{Array2, Array3};

use candid::{CandidType, Decode, Deserialize, Encode};
use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
use ic_stable_structures::{storable::Bound, DefaultMemoryImpl, StableBTreeMap, Storable};
use std::{borrow::Cow, cell::RefCell};

mod geo;
mod model;
mod sphere;
mod svd;
mod t_df;

type Memory = VirtualMemory<DefaultMemoryImpl>;
const MAX_VALUE_SIZE: u32 = 500_000_000;
const STORED_SECS_ID: u64 = 1;

#[derive(CandidType, Deserialize, Clone, Debug, Default)]
pub struct StoredSECS {
    sec_locs: Vec<GeographicalPoint>,
    sec_locs_altitude: f32,

    sec_amps: Vec<f32>,
    sec_amps_shape: Vec<usize>,

    pub obs_locs_cache: Vec<GeographicalPoint>,
    pub pred_locs_cache: Vec<GeographicalPoint>,
    // Using vectors instead of Array2/3 as ndarray does not implement CandidType
    // will need to reconvert them afterwards
    pub t_obs_flat_cache: Vec<f32>,
    pub t_obs_flat_shape: Vec<usize>,

    pub t_pred_cache: Vec<f32>,
    pub t_pred_shape: Vec<usize>,
}

impl SECS {
    /// Loads SECS from stable memory
    pub fn load() -> Self {
        let stored_secs: StoredSECS = STORED_SECS.with(|p| p.borrow().clone()).unwrap_or_default();

        SECS {
            sec_locs: stored_secs.sec_locs,
            sec_locs_altitude: stored_secs.sec_locs_altitude,

            sec_amps: Array2::from_shape_vec(
                [stored_secs.sec_amps_shape[0], stored_secs.sec_amps_shape[1]],
                stored_secs.sec_amps,
            )
            .ok(),

            obs_locs_cache: stored_secs.obs_locs_cache,
            t_obs_flat_cache: Array2::from_shape_vec(
                [
                    stored_secs.t_obs_flat_shape[0],
                    stored_secs.t_obs_flat_shape[1],
                ],
                stored_secs.t_obs_flat_cache,
            )
            .ok(),

            pred_locs_cache: stored_secs.pred_locs_cache,
            t_pred_cache: Array3::from_shape_vec(
                [
                    stored_secs.t_pred_shape[0],
                    stored_secs.t_pred_shape[1],
                    stored_secs.t_pred_shape[2],
                ],
                stored_secs.t_pred_cache,
            )
            .ok(),
        }
    }

    /// Converts self to StoredSECS and store it in stable memory
    pub fn store(self) {
        let t_obs: Array2<f32> = self.t_obs_flat_cache.unwrap_or_default();
        let t_pred: Array3<f32> = self.t_pred_cache.unwrap_or_default();
        let amps: Array2<f32> = self.sec_amps.unwrap_or_default();

        STORED_SECS.with(|p| {
            *p.borrow_mut() = Some(StoredSECS {
                sec_locs: self.sec_locs,
                sec_locs_altitude: self.sec_locs_altitude,

                sec_amps_shape: amps.shape().to_vec(),
                sec_amps: amps.into_raw_vec(),

                obs_locs_cache: self.obs_locs_cache,
                pred_locs_cache: self.pred_locs_cache,

                t_obs_flat_shape: t_obs.shape().to_vec(),
                t_obs_flat_cache: t_obs.into_raw_vec(),

                t_pred_shape: t_pred.shape().to_vec(),
                t_pred_cache: t_pred.into_raw_vec(),
            });
        });
    }
}

thread_local! {
    static STORED_SECS: RefCell<Option<StoredSECS>> = RefCell::new(None);
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
pub fn predict() -> Vec<PredictionVector> {
    let mut secs = SECS::load();
    secs.predict()
}

// #[wasm_bindgen]
// pub fn wasm_infer(js_obs: JsValue) -> Result<JsValue, JsValue> {
//     console_error_panic_hook::set_once();
//     let observations: Vec<ObservationVector> =
//         serde_wasm_bindgen::from_value(js_obs).expect("Failed to deserialize observations");
//
//     let obs_grid = geographical_grid(45.0..85.0, 37, -170.0..35.0, 130);
//
//     let mut secs = SECS::new(obs_grid, 0.0);
//     secs.fit(&observations, 0.0, 0.05);
//
//     let pred_grid = geographical_grid(45.0..85.0, 37, -180.0..179.0, 130);
//     secs.calc_t_pred(&pred_grid, 110e3);
//     let pred = secs.predict(&pred_grid, 110e3);
//
//     Ok(serde_wasm_bindgen::to_value(&pred)?)
// }

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
    use super::*;
    use approx::assert_relative_eq;
    use std::fs;

    #[test]
    fn test_infer() {
        // let secs: SECS = secs_cache::secs;

        let obs_grid = geographical_grid(45.0..85.0, 37, -170.0..35.0, 130);
        let pred_grid = geographical_grid(45.0..85.0, 37, -180.0..179.0, 130);

        let mut secs = SECS::new(obs_grid, 0.0);

        let obs = vec![ObservationVector {
            lon: 1.0,
            lat: 1.0,
            i: 1.0,
            j: 1.0,
            k: 1.0,
        }];

        secs.fit(&obs, 0.0, 0.05);
        secs.calc_t_pred(&pred_grid, 110e3);
        let pred = secs.predict();
        println!("{:?}", pred.iter().take(10));

        secs.obs_locs_cache = vec![];
        secs.t_obs_flat_cache = None;
        secs.sec_amps = None;

        let debug_output = format!("{:#?}", secs);
        fs::write("secs_cache.txt", debug_output).unwrap();
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::secs::{ObservationVector, SECS};
//
//     use super::*;
//
//     #[test]
//     fn test_secs() {
//         let grid = geographical_grid(45.0..85.0, 37, -180.0..179.0, 130, 110.0);
//         let obs: Vec<ObservationVector> = vec![
//             ObservationVector {
//                 lon: 11.95,
//                 lat: 78.92,
//                 i: -71.5,
//                 j: -135.0,
//                 k: -191.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 15.82,
//                 lat: 78.2,
//                 i: -78.0,
//                 j: -150.0,
//                 k: -164.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 19.2,
//                 lat: 74.5,
//                 i: 4.8333335,
//                 j: -112.0,
//                 k: -151.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 351.3,
//                 lat: 70.9,
//                 i: 124.5,
//                 j: -24.0,
//                 k: -209.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 25.79,
//                 lat: 71.09,
//                 i: 72.0,
//                 j: -104.0,
//                 k: -83.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 12.1,
//                 lat: 67.53,
//                 i: 230.0,
//                 j: -96.0,
//                 k: -143.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 22.22,
//                 lat: 70.54,
//                 i: 73.0,
//                 j: -125.0,
//                 k: -108.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 27.01,
//                 lat: 69.76,
//                 i: 68.0,
//                 j: -111.0,
//                 k: -72.333336,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 18.94,
//                 lat: 69.66,
//                 i: 116.0,
//                 j: -121.0,
//                 k: -161.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 26.63,
//                 lat: 67.37,
//                 i: 160.96428,
//                 j: -92.0,
//                 k: -39.25,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 23.7,
//                 lat: 69.46,
//                 i: 76.0,
//                 j: -125.0,
//                 k: -96.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 16.03,
//                 lat: 69.3,
//                 i: 140.0,
//                 j: -117.0,
//                 k: -175.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 16.98,
//                 lat: 66.4,
//                 i: 216.0,
//                 j: -69.0,
//                 k: -12.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 20.77,
//                 lat: 69.06,
//                 i: 136.0,
//                 j: -146.0,
//                 k: -128.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 27.29,
//                 lat: 68.56,
//                 i: 81.0,
//                 j: -107.0,
//                 k: -58.88889,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 23.53,
//                 lat: 68.02,
//                 i: 146.5,
//                 j: -77.0,
//                 k: -78.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 24.08,
//                 lat: 66.9,
//                 i: 165.0,
//                 j: -61.666668,
//                 k: -33.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 12.5,
//                 lat: 66.11,
//                 i: 233.0,
//                 j: -71.0,
//                 k: -6.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 26.25,
//                 lat: 65.54,
//                 i: 145.0,
//                 j: -49.0,
//                 k: 42.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 10.98,
//                 lat: 64.94,
//                 i: 196.0,
//                 j: -50.0,
//                 k: 84.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 27.23,
//                 lat: 64.52,
//                 i: 70.0,
//                 j: -21.0,
//                 k: 97.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 30.97,
//                 lat: 62.77,
//                 i: 9.0,
//                 j: -0.16666667,
//                 k: 77.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 26.6,
//                 lat: 62.25,
//                 i: 6.6666665,
//                 j: -4.0,
//                 k: 92.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 9.11,
//                 lat: 62.07,
//                 i: 41.0,
//                 j: -30.0,
//                 k: 123.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 4.84,
//                 lat: 61.08,
//                 i: 22.0,
//                 j: -31.5,
//                 k: 95.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 10.75,
//                 lat: 60.21,
//                 i: -3.0,
//                 j: -11.0,
//                 k: 94.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 24.65,
//                 lat: 60.5,
//                 i: -11.875,
//                 j: 5.0,
//                 k: 79.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 5.24,
//                 lat: 59.21,
//                 i: 4.0,
//                 j: -21.0,
//                 k: 67.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 26.46,
//                 lat: 58.26,
//                 i: -27.0,
//                 j: 6.0,
//                 k: 60.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 11.67,
//                 lat: 55.62,
//                 i: 11.0,
//                 j: -33.0,
//                 k: 70.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 290.77,
//                 lat: 77.47,
//                 i: -15.0,
//                 j: -153.60715,
//                 k: 51.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 302.82,
//                 lat: 74.57,
//                 i: 124.55838,
//                 j: -367.02618,
//                 k: -214.66805,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 303.85,
//                 lat: 72.78,
//                 i: 265.62704,
//                 j: -290.49402,
//                 k: -139.12407,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 321.7,
//                 lat: 72.3,
//                 i: 3.0,
//                 j: -30.0,
//                 k: -142.5,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 307.87,
//                 lat: 70.68,
//                 i: 105.0,
//                 j: 94.0,
//                 k: 0.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 306.47,
//                 lat: 69.25,
//                 i: -5.0,
//                 j: -85.0,
//                 k: -109.5,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 309.28,
//                 lat: 67.02,
//                 i: 256.11157,
//                 j: -121.24982,
//                 k: -339.35138,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 314.56,
//                 lat: 61.16,
//                 i: 84.0,
//                 j: 58.0,
//                 k: -49.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 254.763,
//                 lat: 40.137,
//                 i: -44.0,
//                 j: -20.0,
//                 k: -4.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 203.378,
//                 lat: 71.322,
//                 i: -237.0,
//                 j: -53.0,
//                 k: 274.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 212.14,
//                 lat: 64.874,
//                 i: -115.0,
//                 j: -60.0,
//                 k: -155.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 242.878,
//                 lat: 48.265,
//                 i: -39.0,
//                 j: -20.0,
//                 k: -10.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 199.538,
//                 lat: 55.348,
//                 i: -25.0,
//                 j: -5.0,
//                 k: -35.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 224.675,
//                 lat: 57.058,
//                 i: -43.0,
//                 j: -32.0,
//                 k: -9.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 297.647,
//                 lat: 82.497,
//                 i: -110.400024,
//                 j: 127.60986,
//                 k: 17.615234,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 264.0,
//                 lat: 64.3,
//                 i: -7.915283,
//                 j: 81.909996,
//                 k: -24.003906,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 262.9,
//                 lat: 49.6,
//                 i: -13.779785,
//                 j: -11.180054,
//                 k: 4.609375,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 255.0,
//                 lat: 69.2,
//                 i: -111.38501,
//                 j: 59.245026,
//                 k: 51.39453,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 274.1,
//                 lat: 80.0,
//                 i: -17.535034,
//                 j: 145.89514,
//                 k: 67.53516,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 265.9,
//                 lat: 58.8,
//                 i: 72.06055,
//                 j: 7.439995,
//                 k: 31.19336,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 291.5,
//                 lat: 63.8,
//                 i: 72.10449,
//                 j: -59.029907,
//                 k: -256.11328,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 246.7,
//                 lat: 54.6,
//                 i: -33.798973,
//                 j: -27.870117,
//                 k: 4.4648438,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 284.5,
//                 lat: 45.4,
//                 i: -59.874023,
//                 j: -23.930258,
//                 k: 15.128662,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 265.1,
//                 lat: 74.7,
//                 i: -131.75987,
//                 j: 31.866024,
//                 k: 196.35051,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 307.3,
//                 lat: 47.6,
//                 i: -62.539063,
//                 j: -22.45491,
//                 k: 11.0,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: 236.6,
//                 lat: 48.6,
//                 i: -46.173145,
//                 j: -26.061047,
//                 k: -11.208085,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: -147.447,
//                 lat: 65.136,
//                 i: -317.0,
//                 j: -78.0,
//                 k: -134.05577,
//                 alt: 0.0,
//             },
//             ObservationVector {
//                 lon: -141.205,
//                 lat: 64.786,
//                 i: -166.88467,
//                 j: 51.166668,
//                 k: -48.0,
//                 alt: 0.0,
//             },
//         ];
//
//         let mut secs = SECS::new(grid.clone());
//         secs.fit(&obs, 0.05);
//         match secs.predict(&grid) {
//             Ok(r) => {
//                 let mut v: Vec<_> = r.iter().filter(|x| x.i.is_finite()).collect();
//                 v.sort_by(|a, b| b.i.partial_cmp(&a.i).unwrap());
//                 for item in v.into_iter().take(10) {
//                     println!("{:?}", item);
//                 }
//             }
//             Err(_) => {}
//         }
//     }
// }
