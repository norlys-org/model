mod grid;
mod matrix;
mod overlays;
pub mod secs;
mod sphere;
mod svd;

use overlays::{apply_auroral_zone_overlay, encode_score, ponderate_i, ScoreVector};
use secs::{secs_interpolate, ObservationMatrix};
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
        0.1,
    );

    let pred_score: Vec<f32> = pred
        .into_iter()
        .map(|v| {
            // encode_score(
                // apply_auroral_zone_overlay(v.lon, v.lat, ponderate_i(v.i))
            v.i
            //     false, // TODO: Leave to false as of now derivative is not computed
            // )
        })
        .collect();

    Ok(serde_wasm_bindgen::to_value(&pred_score)?)
}
