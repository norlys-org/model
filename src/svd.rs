use ndarray::{Array2, Array3};
use ndarray_linalg::SVD;

fn svd(t_obs_flat: &Array2<f64>, epsilon: f64) {
    let (u, s, vh) = t_obs_flat.svd(true, true).unwrap();
}
