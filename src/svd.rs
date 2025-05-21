use nalgebra::{DMatrix, SVD};
use ndarray::{s, Array2, Array3};

pub fn svd(t_obs_flat: &Array2<f64>, epsilon: f64) -> Array2<f64> {
    // Convert ndarray Array2 to nalgebra DMatrix
    let (rows, cols) = t_obs_flat.dim();
    let flat_data: Vec<f64> = t_obs_flat.iter().cloned().collect();
    let matrix = DMatrix::from_vec(rows, cols, flat_data);

    let svd = SVD::new(matrix, true, true);
    let (u, s, vh) = (svd.u.unwrap(), svd.singular_values, svd.v_t.unwrap());

    let valid_indices: Vec<usize> = s
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| {
            if val >= epsilon * s.max() {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let u = Array2::from_shape_fn((u.nrows(), u.ncols()), |(i, j)| u[(i, j)]);
    let vh = Array2::from_shape_fn((vh.nrows(), vh.ncols()), |(i, j)| vh[(i, j)]);

    let u = valid_indices
        .iter()
        .map(|&idx| u.slice(s![.., idx]).to_owned())
        .collect::<Vec<_>>();

    let vh = valid_indices
        .iter()
        .map(|&idx| vh.slice(s![idx, ..]).to_owned())
        .collect::<Vec<_>>();

    // let w_values: Vec<f64> = valid_indices.iter().map(|&idx| 1.0 / s[idx]).collect();
    //
    // let w_diag = Array2::from_diag(&ndarray::Array1::from(w_values));
    //
    // // Vh.T @ (W @ U.T)
    // let u_t = u.t().to_owned();
    // let w_u_t = w_diag.dot(&u_t);
    // let vh_t = vh.t().to_owned();
    //
    // // Return the final result
    // vh_t.dot(&w_u_t)
}
