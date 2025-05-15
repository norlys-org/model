/// Calculates both the angular distance and bearing between two sets of lat/lon points
///
/// # Arguments
/// * `latlon1` - First set of points as a vector of (lat, lon) tuples in degrees
/// * `latlon2` - Second set of points as a vector of (lat, lon) tuples in degrees
///
/// # Returns
/// A tuple containing:
/// - Matrix of angular distances in radians (Vec<Vec<f64>>)
/// - Matrix of bearings in radians (Vec<Vec<f64>>)
pub fn calc_angular_distance_and_bearing(
    latlon1: &[(f64, f64)],
    latlon2: &[(f64, f64)],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = latlon1.len();
    let m = latlon2.len();

    let mut theta = vec![vec![0.0; m]; n];
    let mut alpha = vec![vec![0.0; m]; n];

    for i in 0..n {
        let (lat1_deg, lon1_deg) = latlon1[i];
        let lat1_rad = lat1_deg.to_radians();
        let lon1_rad = lon1_deg.to_radians();
        let cos_lat1 = lat1_rad.cos();
        let sin_lat1 = lat1_rad.sin();

        for j in 0..m {
            let (lat2_deg, lon2_deg) = latlon2[j];
            let lat2_rad = lat2_deg.to_radians();
            let lon2_rad = lon2_deg.to_radians();
            let cos_lat2 = lat2_rad.cos();
            let sin_lat2 = lat2_rad.sin();

            let dlon = lon2_rad - lon1_rad;
            let cos_dlon = dlon.cos();
            let sin_dlon = dlon.sin();

            // Calculate angular distance (theta)
            theta[i][j] = (sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_dlon).acos();

            // Calculate bearing (alpha)
            // Following the Python implementation:
            let x = cos_lat2 * sin_dlon;
            let y = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_dlon;

            // The original Python formula: alpha = np.pi / 2 - np.arctan2(x, y)
            alpha[i][j] = std::f64::consts::FRAC_PI_2 - x.atan2(y);
        }
    }

    (theta, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Helper function to check if two 2D vectors are approximately equal
    fn assert_matrices_approx_eq(actual: &Vec<Vec<f64>>, expected: &Vec<Vec<f64>>, epsilon: f64) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "Matrices have different row counts"
        );

        for i in 0..actual.len() {
            assert_eq!(
                actual[i].len(),
                expected[i].len(),
                "Row {} has different column counts",
                i
            );

            for j in 0..actual[i].len() {
                assert_relative_eq!(
                    actual[i][j],
                    expected[i][j],
                    max_relative = epsilon,
                    epsilon = epsilon,
                );
            }
        }
    }

    // Convert degrees to radians
    fn deg_to_rad(degrees: &[(f64, f64)]) -> Vec<(f64, f64)> {
        degrees
            .iter()
            .map(|(lat, lon)| {
                (
                    *lat * std::f64::consts::PI / 180.0,
                    *lon * std::f64::consts::PI / 180.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_case_1_basic_cardinal_points() {
        let latlon1 = vec![(0.0, 0.0)];
        let latlon2 = vec![(0.0, 90.0), (90.0, 0.0)];

        let expected_theta = vec![vec![1.5708, 1.5708]];
        let expected_alpha = vec![vec![0.0, 1.5708]];

        let (theta, alpha) = calc_angular_distance_and_bearing(&latlon1, &latlon2);

        let epsilon = 1e-4;
        assert_matrices_approx_eq(&theta, &expected_theta, epsilon);
        assert_matrices_approx_eq(&alpha, &expected_alpha, epsilon);
    }

    #[test]
    fn test_case_2_realistic_coordinates() {
        let latlon1 = vec![(40.71, -74.0), (34.05, -118.2)];
        let latlon2 = vec![(48.85, 2.35)];

        let expected_theta = vec![vec![0.9162], vec![1.4258]];
        let expected_alpha = vec![vec![0.6333], vec![0.961]];

        let (theta, alpha) = calc_angular_distance_and_bearing(&latlon1, &latlon2);

        let epsilon = 1e-4;
        assert_matrices_approx_eq(&theta, &expected_theta, epsilon);
        assert_matrices_approx_eq(&alpha, &expected_alpha, epsilon);
    }

    #[test]
    fn test_case_3_antipodal_points() {
        let latlon1 = vec![(90.0, 0.0)];
        let latlon2 = vec![(-90.0, 0.0)];

        let expected_theta = vec![vec![3.1416]]; // π radians = 180 degrees
        let expected_alpha = vec![vec![-1.5708]]; // -π/2 radians = -90 degrees

        let (theta, alpha) = calc_angular_distance_and_bearing(&latlon1, &latlon2);

        let epsilon = 1e-4;
        assert_matrices_approx_eq(&theta, &expected_theta, epsilon);
        assert_matrices_approx_eq(&alpha, &expected_alpha, epsilon);
    }

    #[test]
    fn test_case_4_multiple_points_grid() {
        let latlon1 = vec![(10.0, 10.0), (20.0, 20.0)];
        let latlon2 = vec![(10.0, 11.0), (21.0, 20.0), (15.0, 15.0)];

        let expected_theta = vec![vec![0.0172, 0.255, 0.1219], vec![0.2311, 0.0175, 0.1206]];
        let expected_alpha = vec![vec![0.0015, 0.8728, 0.8064], vec![3.9747, 1.5708, 3.9371]];

        let (theta, alpha) = calc_angular_distance_and_bearing(&latlon1, &latlon2);

        let epsilon = 1e-4;
        assert_matrices_approx_eq(&theta, &expected_theta, epsilon);
        assert_matrices_approx_eq(&alpha, &expected_alpha, epsilon);
    }

    #[test]
    fn test_case_5_edge_cases() {
        let latlon1 = vec![(45.0, 45.0), (89.5, 10.0), (0.0, 179.5)];
        let latlon2 = vec![(45.0, 45.0), (-89.5, -170.0), (0.0, -179.5)];

        let expected_theta = vec![
            vec![0.0, 2.3633, 2.0994],
            vec![0.7783, 3.1416, 1.5794],
            vec![2.0893, 1.5622, 0.0175],
        ];
        let expected_alpha = vec![
            vec![1.5708, -1.5637, 0.6237],
            vec![-0.9549, 3.1416, 1.405],
            vec![2.1904, -1.5692, 0.0],
        ];

        let (theta, alpha) = calc_angular_distance_and_bearing(&latlon1, &latlon2);

        let epsilon = 1e-4;
        assert_matrices_approx_eq(&theta, &expected_theta, epsilon);
        assert_matrices_approx_eq(&alpha, &expected_alpha, epsilon);
    }
}
