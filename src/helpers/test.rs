use std::f32::consts::PI;

// Helper function for approximate comparison of float vectors
pub fn assert_vec_approx_eq(result: &Vec<Vec<f32>>, expected: &Vec<Vec<f32>>, tolerance: f32) {
    assert_eq!(result.len(), expected.len(), "Number of rows mismatch");
    for i in 0..result.len() {
        assert_eq!(
            result[i].len(),
            expected[i].len(),
            "Number of columns mismatch in row {}",
            i
        );
        for j in 0..result[i].len() {
            // Handle potential edge case where expected is exactly PI or -PI and result is the other
            // Or handle cases around 0 vs 2*PI etc. if normalization was applied (not needed here as atan2 returns [-pi, pi])
            let diff = (result[i][j] - expected[i][j]).abs();
            let diff_angle = (diff + PI) % (2.0 * PI) - PI; // Angle difference on the circle

            assert!(
                diff_angle.abs() < tolerance,
                "Mismatch at [{}][{}]: result = {:.6}, expected = {:.6}, diff_angle = {:.6}",
                i,
                j,
                result[i][j],
                expected[i][j],
                diff_angle
            );
        }
    }
}
