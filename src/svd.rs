use crate::matrix::{invert_matrix, multiply_matrices, multiply_matrix_vector, transpose_matrix};

/// Solves a linear system using SVD (Singular Value Decomposition)
/// Simplified implementation for demonstration purposes
///
/// # Arguments
/// * `a` - Matrix a
/// * `b` - Vector b
/// * `epsilon` - Regularization parameter
///
/// # Returns
/// Solution vector x
fn solve_svd(a: Vec<Vec<f32>>, b: Vec<f32>, epsilon: f32) -> Vec<f32> {
    let at = transpose_matrix(a.clone());
    let mut ata = multiply_matrices(at.clone(), a);

    // Add regularization to diagonal
    let max_diag = ata
        .iter()
        .flat_map(|vec| vec.iter().copied())
        .fold(f32::NEG_INFINITY, f32::max);
    let reg = epsilon * max_diag;

    for i in 0..ata.len() {
        ata[i][i] += reg;
    }

    let ata_inv = invert_matrix(ata);

    let at_b = multiply_matrix_vector(at, b);
    multiply_matrix_vector(ata_inv, at_b)
}
