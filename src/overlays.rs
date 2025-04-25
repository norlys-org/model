use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct ScoreVector {
    pub lat: f32,
    pub lon: f32,
    pub score: f32,
}

/// Given the value of the i component of a predicted vector, ponderate to give
/// a score between 0 and 10
/// Generated from a curve with the following points.
/// 0 0
/// 50 2
/// 100 3
/// 200 4
/// 800 10
pub fn ponderate_i(value: f32) -> f32 {
    let numerator = -0.05732817 - 22.81964;
    let denominator = 1.0 + (value / 1055.17).powf(0.8849212);
    let result = 22.81964 + numerator / denominator;

    f32::min(10.0, result)
}
