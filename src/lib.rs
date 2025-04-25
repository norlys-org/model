mod grid;
mod matrix;
mod overlays;
pub mod secs;
mod sphere;
mod svd;

use overlays::{ponderate_i, ScoreVector};
use secs::{secs_interpolate, ObservationMatrix, PredictionVector};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn infer(js_obs: JsValue) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let observations: ObservationMatrix =
        serde_wasm_bindgen::from_value(js_obs).expect("Failed to deserialize observations");

    let pred = secs_interpolate(
        observations,
        45f32..85f32,
        37,
        -180f32..179f32,
        130,
        110e3,
        0f32,
    );

    let pred_score: Vec<ScoreVector> = pred
        .into_iter()
        .map(|v| ScoreVector {
            lon: v.lon,
            lat: v.lat,
            score: ponderate_i(v.i),
        })
        .collect();

    Ok(serde_wasm_bindgen::to_value(&pred_score)?)
}
