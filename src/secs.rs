use crate::{
    grid::{geographical_grid, GeographicalPoint},
    matrix::t_df,
    svd::solve_svd,
};
use serde::{Deserialize, Serialize};
use std::mem;
use std::ops::Range;
use wasm_bindgen::prelude::*;

pub const R_EARTH: f32 = 6371e3;

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Observation {
    /// The longitude in degrees.
    pub longitude: f32,
    /// The latitude in degrees.
    pub latitude: f32,
    // i vector (usually x magnetometer component) in nano teslas
    pub i: f32,
    // j vector (usually y magnetometer component) in nano teslas
    pub j: f32,
    // Altitude from the surface of the earth where the measurement has been conducted (usually 0)
    // in meters
    pub altitude: f32,
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Debug)]
pub struct PredictedPoint {
    /// The longitude in degrees.
    longitude: f32,
    /// The latitude in degrees.
    latitude: f32,
    // i vector (usually x magnetometer component) in nano teslas
    i: f32,
    // j vector (usually y magnetometer component) in nano teslas
    j: f32,
}

pub fn secs_interpolate(
    observations: Vec<Observation>,
    lat_range: Range<f32>,
    lat_steps: usize,
    lon_range: Range<f32>,
    lon_steps: usize,
    sec_altitude: f32,
    prediction_altitude: f32,
) -> Vec<PredictedPoint> {
    let secs_locs = geographical_grid(
        lat_range.clone(),
        50,
        lon_range.clone(),
        50,
        R_EARTH + sec_altitude,
    );
    let obs_locs: Vec<GeographicalPoint> = observations
        .iter()
        .map(|obs| GeographicalPoint {
            latitude: obs.latitude,
            longitude: obs.longitude,
            altitude: obs.altitude,
        })
        .collect();

    let t = t_df(&obs_locs, &secs_locs);

    let flat_t: Vec<Vec<f32>> = t
        // let flat_t: Vec<Vec<f32>> = t_df(&obs_locs, &secs_locs)
        .into_iter()
        .flat_map(|mut row| {
            vec![
                mem::take(&mut row[0]), // Bx row
                mem::take(&mut row[1]), // By row
            ]
        })
        .collect();

    let flat_b: Vec<f32> = observations
        .into_iter()
        .flat_map(|obs| {
            vec![
                obs.i, // Bx component
                obs.j, // By component
            ]
        })
        .collect();

    let sec_amps = solve_svd(flat_t, flat_b, 0f32);
    let pred = geographical_grid(
        lat_range,
        lat_steps,
        lon_range,
        lon_steps,
        prediction_altitude,
    );
    let t_pred = t_df(&pred, &secs_locs);

    let mut result: Vec<PredictedPoint> = Vec::with_capacity(pred.len());
    for i in 0..pred.len() {
        let mut bx = 0f32;
        let mut by = 0f32;

        for j in 0..secs_locs.len() {
            bx += t_pred[i][0][j] * sec_amps[j];
            by += t_pred[i][1][j] * sec_amps[j];
        }

        result.push(PredictedPoint {
            longitude: pred[i].longitude,
            latitude: pred[i].latitude,
            i: bx,
            j: by,
        })
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secs_interpolate() {
        let obs: Vec<Observation> = vec![Observation {
            latitude: 69f32,
            longitude: 19f32,
            i: -100f32,
            j: -100f32,
            altitude: 0f32,
        }];

        let pred = secs_interpolate(obs, 45f32..85f32, 37, -180f32..179f32, 130, 110e3, 0f32);
        // println!("{:?}", pred);
    }
}
