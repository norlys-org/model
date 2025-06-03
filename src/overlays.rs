use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::{geo::R_EARTH, model::PredictionVector};

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct ScoreVector {
    pub lat: f64,
    pub lon: f64,
    pub score: f64,
}

/// Given the value of the i component of a predicted vector, ponderate to give
/// a score between 0 and 10
/// Generated from a curve with the following points.
/// 0 0
/// 50 2
/// 100 3
/// 200 4
/// 800 10
fn ponderate_i(i: f64) -> f64 {
    let numerator = -0.05732817 - 22.81964;
    let denominator = 1.0 + (i / 1055.17).powf(0.8849212);
    let result = 22.81964 + numerator / denominator;

    f64::min(10.0, result)
}

pub trait IntoScores {
    fn into_scores(self) -> Vec<ScoreVector>;
}

impl IntoScores for Vec<PredictionVector> {
    fn into_scores(self) -> Vec<ScoreVector> {
        self.into_iter()
            .map(|pv| ScoreVector {
                lat: pv.lat,
                lon: pv.lon,
                score: pv.i,
                // score: ponderate_i(pv.i),
            })
            .collect()
    }
}

/// Approximate the distance between two points on a sphere given in degrees
fn approx_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let to_rad = |angle: f64| -> f64 { angle * PI };
    let lat1_rad = to_rad(lat1);
    let lat2_rad = to_rad(lat2);

    let delta_lat = lat1_rad - lat2_rad;
    let delta_lon = to_rad(lon2 - lon1);

    let x = delta_lon * f64::cos((lat1_rad + lat2_rad) / 2f64);
    let y = delta_lat;

    R_EARTH * f64::sqrt(x * x + y * y)
}

/// Given a distance in km (usually distance from geomagnetic pole) return a weight
/// from 0 to 1 computed according to the follow points
/// 0 0.3
/// 2200 1
/// 2700 1
/// 3300 0.5
/// 4000 0.2
fn auroral_zone_weight(d: f64) -> f64 {
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
pub fn ponderate_auroral_zone(lon: f64, lat: f64, score: f64) -> f64 {
    let geomag_n_pole: (f64, f64) = (-72.6_f64, 80.9_f64);

    let d = approx_distance(lat, lon, geomag_n_pole.0, geomag_n_pole.1);
    let w = auroral_zone_weight(d);

    // score: f64::max(vec.score * w, ponderate_didt())
    score * w
}

/// Encodes the score and the derivative flag into a single byte, see specification document
pub fn encode_score(val: f64, flag: bool) -> u8 {
    let clamped = val.clamp(0.0, 10.0);
    let rounded = (clamped * 10.0).round() as u8;
    assert!(rounded <= 100);
    let flag_bit = if flag { 1 << 7 } else { 0 };
    flag_bit | (rounded & 0b0111_1111)
}

pub trait Overlays {
    fn ponderate_auroral_zone(self) -> Self;
    fn encode(self) -> Vec<u8>;
}

impl Overlays for Vec<ScoreVector> {
    fn ponderate_auroral_zone(self) -> Self {
        self.into_iter()
            .map(|v| ScoreVector {
                lat: v.lat,
                lon: v.lon,
                score: ponderate_auroral_zone(v.lon, v.lat, v.score),
            })
            .collect()
    }

    fn encode(self) -> Vec<u8> {
        self.into_iter()
            .map(|v| v.score as u8)
            // .map(|v| encode_score(v.score, false))
            .collect()
    }
}

///// Ponderate the derivative of the `i` component of the vector
//pub fn ponderate_didt(didt: f64) -> f64 {
//    if didt < 10f64 {
//        0f64
//    } else {
//        didt / 10f64
//    }
//}
