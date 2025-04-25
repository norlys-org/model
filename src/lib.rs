mod grid;
mod matrix;
pub mod secs;
mod sphere;
mod svd;

use prost::Message;
use protobufs::{ObservationMatrix, PredictionMatrix};
use secs::secs_interpolate;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn infer(bytes: &[u8]) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();

    web_sys::console::log_1(&format!("{:?}", bytes).into());
    let observations = ObservationMatrix::decode(bytes)
        .map_err(|e| JsValue::from_str(&format!("Failed to deserialize observations: {}", e)))?;

    web_sys::console::log_1(&format!("{:?}", observations).into());
    let pred = secs_interpolate(
        observations.matrix,
        45f32..85f32,
        37,
        -180f32..179f32,
        130,
        110e3,
        0f32,
    );

    let mut buf: Vec<u8> = vec![];
    let matrix = PredictionMatrix { matrix: pred };
    matrix.encode(&mut buf).unwrap();

    Ok(serde_wasm_bindgen::to_value(&buf)?)
}
