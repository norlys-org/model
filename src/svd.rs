use crate::matrix::{multiply_matrices, transpose_matrix};

fn solve_svd(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, epsilon: f32) {
    let at = transpose_matrix(a.clone());
    let ata = multiply_matrices(at, a);

    let max_diag = ata
        .iter()
        .flat_map(|vec| vec.iter().copied())
        .fold(f32::NEG_INFINITY, f32::max);
}
