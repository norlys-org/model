mod grid;
pub mod helpers;
mod matrix;
mod overlays;
pub mod secs;
mod sphere;
mod svd;

use secs::{ObservationMatrix, ObservationVector};

// #[wasm_bindgen]
// pub fn infer(js_obs: JsValue) -> Result<JsValue, JsValue> {
//     console_error_panic_hook::set_once();
//     let observations: ObservationMatrix =
//         serde_wasm_bindgen::from_value(js_obs).expect("Failed to deserialize observations");
//
//     let pred = secs_interpolate(
//         observations,
//         45f32..85f32,
//         37,
//         -180f32..179f32,
//         130,
//         110e3,
//         0f32,
//     );
//
//     let pred_score: Vec<f32> = pred
//         .into_iter()
//         .map(|v| {
//             // encode_score(
//             // apply_auroral_zone_overlay(v.lon, v.lat, ponderate_i(v.i))
//             v.i
//             //     false, // TODO: Leave to false as of now derivative is not computed
//             // )
//         })
//         .collect();
//
//     Ok(serde_wasm_bindgen::to_value(&pred_score)?)
// }

#[cfg(test)]
mod tests {
    use crate::{grid::geographical_grid, secs::SECS};

    use super::*;

    #[test]
    fn test_secs() {
        let grid = geographical_grid(45.0..85.0, 37, -180.0..179.0, 130, 110.0);
        let obs: ObservationMatrix = vec![ObservationVector {
            lat: 69f32,
            lon: 19f32,
            i: -100f32,
            j: -100f32,
            k: -100f32,
            alt: 0f32,
        }];

        let mut secs = SECS::new(grid.clone());
        secs.fit(&obs, 0.1);
        secs.predict_b(&grid);
    }
}
