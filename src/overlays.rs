use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use wasm_bindgen::prelude::*;

use crate::secs::R_EARTH;

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
pub fn ponderate_i(i: f32) -> f32 {
    let numerator = -0.05732817 - 22.81964;
    let denominator = 1.0 + (i / 1055.17).powf(0.8849212);
    let result = 22.81964 + numerator / denominator;

    f32::min(10.0, result)
}

/// Ponderate the derivative of the `i` component of the vector
pub fn ponderate_didt(didt: f32) -> f32 {
    if didt < 10f32 {
        0f32
    } else {
        didt / 10f32
    }
}

/// Approximate the distance between two points on a sphere given in degrees
fn approx_distance(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    let to_rad = |angle: f32| -> f32 { angle * PI };
    let lat1_rad = to_rad(lat1);
    let lat2_rad = to_rad(lat2);

    let delta_lat = lat1_rad - lat2_rad;
    let delta_lon = to_rad(lon2 - lon1);

    let x = delta_lon * f32::cos((lat1_rad + lat2_rad) / 2f32);
    let y = delta_lat;

    R_EARTH * f32::sqrt(x * x + y * y)
}

/// Given a distance in km (usually distance from geomagnetic pole) return a weight
/// from 0 to 1 computed according to the follow points
/// 0 0.3
/// 2200 1
/// 2700 1
/// 3300 0.5
/// 4000 0.2
fn auroral_zone_weight(d: f32) -> f32 {
    if (0.0..=2200.0).contains(&d) {
        -2.0790e-11 * d.powi(3) - 7.6000e-67 * d.powi(2) + 5.5517e-4 * d + 0.0
    } else if (2200.0..=2700.0).contains(&d) {
        -7.3879e-10 * d.powi(3) + 4.7388e-6 * d.powi(2) - 9.8701e-3 * d + 7.6452
    } else if (2700.0..=3300.0).contains(&d) {
        9.7750e-10 * d.powi(3) - 9.1631e-6 * d.powi(2) + 2.7665e-2 * d - 26.136
    } else if (3300.0..=5000.0).contains(&d) {
        -1.0080e-10 * d.powi(3) + 1.5121e-6 * d.powi(2) - 7.5632e-3 * d + 12.615
    } else {
        0.3
    }
}

/// Given a ScoreVector ponderate the score depending on its vicinity to the auroral oval
pub fn apply_auroral_zone_overlay(vec: ScoreVector) -> ScoreVector {
    let geomag_n_pole: (f32, f32) = (-72.6_f32, 80.9_f32);

    let d = approx_distance(vec.lat, vec.lon, geomag_n_pole.0, geomag_n_pole.1);
    let w = auroral_zone_weight(d);

    ScoreVector {
        lon: vec.lon,
        lat: vec.lat,
        // score: f32::max(vec.score * w, ponderate_didt())
        score: vec.score * w,
    }
}
