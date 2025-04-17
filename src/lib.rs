mod grid;
mod matrix;
mod secs;
mod sphere;
mod svd;

use secs::{secs_interpolate, Observation, PredictedPoint};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn infer(observations: Vec<Observation>) -> Vec<PredictedPoint> {
    secs_interpolate(
        observations,
        45f32..85f32,
        37,
        -180f32..179f32,
        130,
        110e3,
        0f32,
    )
}
