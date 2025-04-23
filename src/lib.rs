mod grid;
mod matrix;
pub mod secs;
mod sphere;
mod svd;

use secs::{secs_interpolate, ObservationVector, ObservationMatrix};
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

    Ok(serde_wasm_bindgen::to_value(&pred)?)
}
