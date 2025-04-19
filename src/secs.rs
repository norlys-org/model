use crate::{
    grid::{geographical_grid, GeographicalPoint},
    matrix::t_df,
    svd::solve_svd,
};
use serde::{Deserialize, Serialize};
use std::mem;
use std::ops::Range;
use std::time::{Duration, Instant};
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
    println!("--- Starting secs_interpolate ---");
    let total_start = Instant::now();

    let mut start = Instant::now();
    let secs_locs = geographical_grid(45f32..85f32, 30, -170f32..35f32, 30, R_EARTH + sec_altitude);
    println!(" -> Generating secs_locs took: {:?}", start.elapsed());

    start = Instant::now();
    let obs_locs: Vec<GeographicalPoint> = observations
        .iter()
        .map(|obs| GeographicalPoint {
            latitude: obs.latitude,
            longitude: obs.longitude,
            altitude: obs.altitude,
        })
        .collect();
    println!(" -> Generating obs_locs took: {:?}", start.elapsed());

    start = Instant::now();
    let t_matrix_intermediate = t_df(&obs_locs, &secs_locs);
    println!(
        " -> Calculating intermediate T matrix (t_df) took: {:?}",
        start.elapsed()
    );

    start = Instant::now();
    let flat_t: Vec<Vec<f32>> = t_matrix_intermediate
        .into_iter()
        .flat_map(|mut row| {
            vec![
                mem::take(&mut row[0]), // Bx row
                mem::take(&mut row[1]), // By row
            ]
        })
        .collect();
    println!(" -> Flattening T matrix took: {:?}", start.elapsed());

    start = Instant::now();
    let flat_b: Vec<f32> = observations // Use into_iter if you don't need observations later
        .into_iter() // Consumes observations
        .flat_map(|obs| {
            vec![
                obs.i, // Bx component
                obs.j, // By component
            ]
        })
        .collect();
    println!(
        " -> Flattening observations (B vector) took: {:?}",
        start.elapsed()
    );

    start = Instant::now();
    let sec_amps = solve_svd(flat_t, flat_b, 0.1f32); // Assuming flat_t and flat_b are consumed or cloned by solve_svd
    println!(" -> Solving SVD took: {:?}", start.elapsed());

    start = Instant::now();
    let pred = geographical_grid(
        lat_range,
        lat_steps,
        lon_range,
        lon_steps,
        prediction_altitude,
    );
    println!(" -> Generating prediction grid took: {:?}", start.elapsed());

    start = Instant::now();
    let t_pred = t_df(&pred, &secs_locs);
    println!(
        " -> Calculating prediction T matrix (t_df) took: {:?}",
        start.elapsed()
    );

    start = Instant::now();
    let mut result: Vec<PredictedPoint> = Vec::with_capacity(pred.len());
    let num_pred_points = pred.len();
    let num_sec_locs = secs_locs.len(); // Cache length for efficiency

    for i in 0..num_pred_points {
        let mut bx = 0f32;
        let mut by = 0f32;

        // Ensure indices are valid if t_pred or sec_amps structure might vary
        if i < t_pred.len() && t_pred[i].len() >= 2 {
            // Pre-fetch rows for potential minor optimization
            let t_pred_bx_row = &t_pred[i][0];
            let t_pred_by_row = &t_pred[i][1];

            for j in 0..num_sec_locs {
                // Check bounds for robustness, though loop condition should guarantee j < num_sec_locs
                if j < t_pred_bx_row.len() && j < t_pred_by_row.len() && j < sec_amps.len() {
                    bx += t_pred_bx_row[j] * sec_amps[j];
                    by += t_pred_by_row[j] * sec_amps[j];
                } else {
                    // Log an error or warning if indices go out of bounds unexpectedly
                    eprintln!("Warning: Index out of bounds during final calculation. i={}, j={}, t_pred[i][0].len={}, t_pred[i][1].len={}, sec_amps.len={}",
                              i, j, t_pred_bx_row.len(), t_pred_by_row.len(), sec_amps.len());
                }
            }
        } else {
            eprintln!(
                "Warning: Index 'i' out of bounds for t_pred. i={}, t_pred.len={}",
                i,
                t_pred.len()
            );
        }

        result.push(PredictedPoint {
            longitude: pred[i].longitude,
            latitude: pred[i].latitude,
            i: bx,
            j: by,
        })
    }
    println!(
        " -> Final result calculation loop took: {:?}",
        start.elapsed()
    );

    println!(
        "--- Finished secs_interpolate. Total time: {:?} ---",
        total_start.elapsed()
    );
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

        let pred = secs_interpolate(obs, 45f32..85f32, 10, -180f32..179f32, 10, 110e3, 0f32);
        println!("{:?}", pred);
    }
}
