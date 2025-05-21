use nalgebra::{DMatrix, SVD};
use ndarray::{Array2, Array3};

pub fn svd(t_obs_flat: &Array2<f64>, epsilon: f64) {
    // Convert ndarray Array2 to nalgebra DMatrix
    let (rows, cols) = t_obs_flat.dim();
    let flat_data: Vec<f64> = t_obs_flat.iter().cloned().collect();
    let matrix = DMatrix::from_vec(rows, cols, flat_data);

    let svd = SVD::new(matrix, true, true);
    let (u, s, vh) = (svd.u.unwrap(), svd.singular_values, svd.v_t.unwrap());

    let valid = s
        .iter()
        .map(|&val| val >= epsilon * s.max())
        .collect::<Vec<bool>>();

    println!("{:?}", valid);

    // let s_max = s.iter().fold(0.0, |max, &val| f64::max(max, val));
    // let valid = s.mapv(|val| val >= epsilon * s_max);
    // println!("{:?}", valid);
}
