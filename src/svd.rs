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
pub fn solve_svd(a: Vec<Vec<f32>>, b: Vec<f32>, epsilon: f32) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compare two vectors element‑wise within `tol`.
    fn assert_vec_approx_eq(v1: &[f32], v2: &[f32], tol: f32) {
        assert_eq!(
            v1.len(),
            v2.len(),
            "Vectors have different lengths ({} vs {})",
            v1.len(),
            v2.len()
        );
        for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert!(
                (a - b).abs() < tol,
                "Mismatch at index {i}: {a} vs {b} (tol={tol})"
            );
        }
    }

    #[test]
    fn solves_2x2_system_correctly() {
        // A · x = b  with known analytic solution x = [0.0, 2.5]
        let a = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let b = vec![5.0, 5.0];
        let epsilon = 1e-6;

        let x = solve_svd(a.clone(), b.clone(), epsilon);

        // Expected analytic answer
        let expected_x = vec![0.0, 2.5];
        assert_vec_approx_eq(&x, &expected_x, 1e-3);

        // Round‑trip check: A · x ≈ b
        let b_hat = multiply_matrix_vector(a, x);
        assert_vec_approx_eq(&b_hat, &b, 1e-3);
    }
}
