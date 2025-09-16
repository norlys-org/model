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
    let matrix = DMatrix::from_row_slice(rows, cols, &flat_data);

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
    use approx::assert_relative_eq;

    use super::*;
    use std::fs;

    #[test]
    fn test_svd_real() {
        let json_content =
            fs::read_to_string("resources/t_obs_flat.json").expect("Failed to read JSON file");
        let input_vec: Vec<Vec<f64>> =
            serde_json::from_str(&json_content).expect("Failed to parse JSON");

        let rows = input_vec.len();
        let cols = input_vec[0].len();
        let flat_input: Vec<f64> = input_vec.into_iter().flatten().collect();
        let input = Array2::from_shape_vec((rows, cols), flat_input)
            .expect("Failed to create Array2 from JSON data");

        let calculated = svd(&input, 0.1);

        let json_content =
            fs::read_to_string("resources/vwu.json").expect("Failed to read JSON file");
        let expected_vec: Vec<Vec<f64>> =
            serde_json::from_str(&json_content).expect("Failed to parse JSON");

        let rows = expected_vec.len();
        let cols = expected_vec[0].len();
        let flat_expected: Vec<f64> = expected_vec.into_iter().flatten().collect();
        let expected = Array2::from_shape_vec((rows, cols), flat_expected)
            .expect("Failed to create Array2 from JSON data");

        let calc = calculated.as_slice().unwrap();
        let exp = expected.as_slice().unwrap();
        assert_eq!(calculated.shape(), expected.shape());
        let mut failed = false;
        for i in 0..calc.len().min(10) {
            let rel_err = ((calc[i] - exp[i]) / exp[i]).abs();
            if rel_err > 1e-10 {
                println!(
                    "idx {}: calc={:.6e}, exp={:.6e}, rel_err={:.2e}",
                    i, calc[i], exp[i], rel_err
                );
                failed = true;
            }
        }
        assert!(!failed, "First 10 values already show large error");
        assert_eq!(
            calculated.shape(),
            expected.shape(),
            "VWU dimensions don't match: calculated {:?} vs expected {:?}",
            calculated.shape(),
            expected.shape()
        );
        assert_relative_eq!(calc, exp, max_relative = 1e-5);
    }

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

        let expected: Array2<f64> = Array2::from_shape_vec(
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
        assert_relative_eq!(
            vwu.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            max_relative = 1e-15
        );
    }

    #[test]
    fn test_svd_realistic_points() {
        let t_obs_flat = Array2::from_shape_vec(
            (12, 4),
            vec![
                6.827264853717463e-14,
                9.689517718394877e-14,
                4.750590192220720e-14,
                5.109387420181284e-14,
                -4.235379862644485e-14,
                5.933118429556925e-30,
                -1.188994785874325e-14,
                3.128597474864381e-30,
                -5.801333306544970e-14,
                -7.435028445824706e-14,
                -2.746474094370104e-14,
                -2.949906202435753e-14,
                1.029209624918833e-14,
                9.849135400444682e-15,
                4.984896951419508e-14,
                7.654964943601531e-14,
                -7.619581196376068e-14,
                -1.469577043150429e-13,
                -4.332805722881490e-14,
                -3.703381769963078e-14,
                -5.461510323377026e-14,
                -1.243912186379883e-13,
                -4.399344090250921e-14,
                -6.263786866121732e-14,
                -1.529317787688337e-14,
                -3.549064384959199e-14,
                1.515077631500203e-14,
                1.467917674496734e-14,
                -4.961348294564456e-14,
                -5.570147667694302e-14,
                -6.529077176559688e-14,
                -9.612852906727201e-14,
                -3.028994262796804e-14,
                -4.399344090250921e-14,
                -4.494903947820231e-14,
                -7.469424710631983e-14,
                -1.679131089590554e-14,
                -2.681911342757982e-14,
                -1.035067228718271e-14,
                -2.556627325533606e-14,
                -3.456686380796599e-14,
                -3.301269459917760e-14,
                -5.220307220143968e-14,
                -5.773737523700458e-14,
                -1.750630669003064e-14,
                -2.134917321240096e-14,
                -3.154322664941039e-14,
                -4.116250468023389e-14,
            ],
        )
        .unwrap();

        let expected: Array2<f64> = Array2::from_shape_vec(
            (4, 12),
            vec![
                4.181245072508486e+12,
                -1.464431263701310e+13,
                -5.496216950246574e+12,
                -3.054444689387419e+12,
                2.362529690873081e+11,
                6.233009520175254e+12,
                2.364369877695222e+11,
                -1.259231987541912e+12,
                2.213263730553430e+12,
                8.977996221789597e+11,
                -3.212427148770240e+12,
                5.033174201230632e+10,
                -5.374623046072026e+11,
                7.517903614410475e+12,
                1.195831669210517e+12,
                -6.268184447810822e+11,
                -3.838208120474751e+12,
                -5.453743999201780e+12,
                -1.955793240615901e+12,
                1.833089195211492e+12,
                -2.912834112346965e+11,
                -6.710643671299893e+11,
                2.555785231740496e+12,
                6.374114187769015e+11,
                7.465719598816196e+11,
                -2.961583055312898e+12,
                -7.260074173813654e+11,
                1.257848045451077e+12,
                1.221292560767908e+12,
                1.534126225000792e+12,
                1.073222580723768e+12,
                -1.890277010633517e+12,
                -7.727492037016722e+11,
                1.456241923723196e+10,
                -1.773666969082516e+12,
                -7.804318263285973e+11,
                -1.207583736725345e+12,
                4.424831179324823e+12,
                2.081879672921097e+12,
                3.758405073367649e+12,
                1.380212852339987e+12,
                -1.735823847470778e+12,
                1.391468467085767e+12,
                -2.192372081832731e+12,
                -2.593936453888952e+12,
                -5.795702120351111e+11,
                -7.848844774712429e+11,
                -1.246512560767249e+12,
            ],
        )
        .unwrap();

        let vwu = svd(&t_obs_flat, 0.05);

        assert_eq!(vwu.shape(), expected.shape(), "VWU have different shapes");
        assert_relative_eq!(
            vwu.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            max_relative = 1e-10
        );
    }
}
