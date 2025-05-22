use nalgebra::{DMatrix, SVD};
use ndarray::{Array1, Array2};

/// Performs SVD-based matrix decomposition to generate the VWU transformation matrix
///
/// This function applies singular value decomposition to the input transfer function
/// matrix and constructs a filtered VWU matrix by discarding small singular values
/// based on a relative cutoff value.
///
/// # Parameters
///
/// * `t_obs_flat` - A 2D array representing the flattened transfer function data
/// * `epsilon` - Relative cutoff value for singular value filtering (range: 0.01-0.1)
///
/// # Output
///
/// Returns a 2D array representing the computed VWU matrix, which is the product:
/// V_filtered^T * W_inverse * U_filtered^T, where:
/// - V_filtered^T: transposed right singular vectors (after filtering)
/// - W_inverse: diagonal matrix containing reciprocals of retained singular values
/// - U_filtered^T: transposed left singular vectors (after filtering)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_few_points() {
        let t_obs_flat = Array2::from_shape_vec(
            (12, 1),
            vec![
                -1.740219584057594e-14,
                -2.063846893630430e-14,
                -7.196789740341245e-15,
                -1.249155868086343e-14,
                -1.695818904037060e-14,
                -2.260086589981562e-15,
                -9.105042385762502e-15,
                -1.488743707273469e-14,
                4.951651778613771e-16,
                -6.470234386398055e-15,
                -1.368218481923297e-14,
                2.120640891125677e-15,
            ],
        )
        .unwrap();

        let expected = Array2::from_shape_vec(
            (1, 12),
            vec![
                -9.844820937276402e+12,
                -1.167565478281233e+13,
                -4.071388861840475e+12,
                -7.066760974717956e+12,
                -9.593636116521627e+12,
                -1.278582771102278e+12,
                -5.150931108655837e+12,
                -8.422164279654078e+12,
                2.801262872271265e+11,
                -3.660359849977943e+12,
                -7.740325462951015e+12,
                1.199695144030031e+12,
            ],
        )
        .unwrap();

        let vwu = svd(&t_obs_flat, 0.05);

        assert_eq!(vwu.shape(), expected.shape(), "VWU have different shapes");

        // relative epsilon
        let epsilon = 1e-15;
        vwu.iter()
            .zip(expected.iter())
            .enumerate()
            .for_each(|(i, (&a, &b))| {
                let diff = (a - b).abs();
                assert!(
                    diff <= a.abs().max(b.abs()) * epsilon,
                    "T differ at index {}: {} vs {} (diff: {}, epsilon: {})",
                    i,
                    a,
                    b,
                    diff,
                    epsilon
                );
            });
    }
}
