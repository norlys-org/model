use nalgebra::{DMatrix, SVD};
use ndarray::{Array1, Array2};

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

    let u_t = Array2::from_shape_fn((u.nrows(), valid_indices.len()), |(i, j)| {
        u[(i, valid_indices[j])]
    })
    .t()
    .to_owned();

    let vh_t = Array2::from_shape_fn((valid_indices.len(), vh.ncols()), |(i, j)| {
        vh[(valid_indices[i], j)]
    })
    .t()
    .to_owned();

    let w_diag = Array2::from_diag(&Array1::from_iter(
        valid_indices.iter().map(|&idx| 1.0 / s[idx]),
    ));

    // Vh.T @ (W @ U.T)
    vh_t.dot(&w_diag.dot(&u_t))
}
