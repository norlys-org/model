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
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
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
    // i vector (usually x magnetometer component) in nano teslas
    pub i: f32,
    // j vector (usually y magnetometer component) in nano teslas
    pub j: f32,
    // k vector (usually k magnetometer component) in nano teslas
    pub k: f32,
}

pub type ObservationMatrix = Vec<ObservationVector>;
pub type PredictionMatrix = Vec<PredictionVector>;

pub fn secs_interpolate(
    observations: ObservationMatrix,
    lat_range: Range<f32>,
    lat_steps: usize,
    lon_range: Range<f32>,
    lon_steps: usize,
    sec_altitude: f32,
    prediction_altitude: f32,
) -> PredictionMatrix {
    let secs_locs = geographical_grid(45f32..85f32, 50, -170f32..32f32, 50, R_EARTH + sec_altitude);
    let obs_locs: Vec<GeographicalPoint> = observations
        .iter()
        .map(|obs| GeographicalPoint {
            latitude: obs.lat,
            longitude: obs.lon,
            altitude: obs.alt,
        })
        .collect();

    let t = t_df(&obs_locs, &secs_locs);

    let flat_t: Vec<Vec<f32>> = t
        .into_iter()
        .flat_map(|mut row| {
            vec![
                mem::take(&mut row[0]), // Bx row
                mem::take(&mut row[1]), // By row
                mem::take(&mut row[2]), // Bz row
            ]
        })
        .collect();

    let flat_b: Vec<f32> = observations
        .into_iter()
        .flat_map(|obs| {
            vec![
                obs.i, // Bx component
                obs.j, // By component
                obs.k, // Bz component
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

    let mut result: PredictionMatrix = Vec::with_capacity(pred.len());
    for i in 0..pred.len() {
        let mut bx = 0f32;
        let mut by = 0f32;
        let mut bz = 0f32;

        for j in 0..secs_locs.len() {
            bx += t_pred[i][0][j] * sec_amps[j];
            by += t_pred[i][1][j] * sec_amps[j];
            bz += t_pred[i][2][j] * sec_amps[j];
        }

        result.push(PredictionVector {
            lon: pred[i].longitude,
            lat: pred[i].latitude,
            i: bx,
            j: by,
            k: bz
        })
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secs_interpolate() {
        let obs: ObservationMatrix = vec![ObservationVector {
            lat: 69f32,
            lon: 19f32,
            i: -100f32,
            j: -100f32,
            k: -100f32,
            alt: 0f32,
        }];

        let pred = secs_interpolate(obs, 45f32..85f32, 37, -180f32..179f32, 130, 110e3, 0f32);
        println!("{:?}", &pred[..10]);
    }
}
