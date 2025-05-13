use nalgebra::{DMatrix, DVector};

pub fn solve_svd(a: Vec<Vec<f64>>, b: Vec<f64>, eps: f64) -> Vec<f64> {
    let m = a.len();
    let n = a[0].len();
    let flat_a: Vec<f64> = a.into_iter().flatten().collect();
    let dm = DMatrix::from_vec(m, n, flat_a);
    let dv = DVector::from_vec(b);

    dm.svd(true, true)
        .solve(&dv, eps)
        .expect("singular or ill‑conditioned")
        .data
        .as_vec()
        .clone()
}
