mod grid;
mod matrix;
pub mod secs;
mod sphere;
mod svd;

use secs::{secs_interpolate, Observation};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn infer(js_obs: JsValue) -> Result<JsValue, JsValue> {
    println!("{:?}", js_obs);
    let observations: Vec<Observation> =
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
