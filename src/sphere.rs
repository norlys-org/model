use ndarray::{Array, Array2, Axis};

use crate::geo::GeographicalPoint;

/// Calculates both the angular distance and bearing between two sets of lat/lon points
///
/// # Arguments
/// * `coords1` - First set of points as a vector of (lat, lon) tuples in degrees
/// * `coords2` - Second set of points as a vector of (lat, lon) tuples in degrees
///
/// # Returns
/// A tuple containing:
/// - 2D array of angular distances in radians (Array2<f32>)
/// - 2D array of bearings in radians (Array2<f32>)
pub fn angular_distance_and_bearing(
    coords1: &[GeographicalPoint],
    coords2: &[GeographicalPoint],
) -> (Array2<f32>, Array2<f32>) {
    let rows = coords1.len();
    let cols = coords2.len();

    let mut theta = Array::zeros((rows, cols));
    let mut alpha = Array::zeros((rows, cols));

    // pre-compute trig
    let coords1_trig: Vec<(f32, f32, f32, f32)> = coords1
        .iter()
        .map(|point| {
            let lat_rad = point.lat_rad();
            let lon_rad = point.lon_rad();
            (lat_rad.cos(), lat_rad.sin(), lon_rad.cos(), lon_rad.sin())
        })
        .collect();

    let coords2_trig: Vec<(f32, f32, f32, f32)> = coords2
        .iter()
        .map(|point| {
            let lat_rad = point.lat_rad();
            let lon_rad = point.lon_rad();
            (lat_rad.cos(), lat_rad.sin(), lon_rad.cos(), lon_rad.sin())
        })
        .collect();

    for i in 0..rows {
        let (cos_lat1, sin_lat1, cos_lon1, sin_lon1) = coords1_trig[i];

        for j in 0..cols {
            let (cos_lat2, sin_lat2, cos_lon2, sin_lon2) = coords2_trig[j];

            // longitude difference
            // cos(lon2 - lon1) = cos_lon2 * cos_lon1 + sin_lon2 * sin_lon1
            // sin(lon2 - lon1) = sin_lon2 * cos_lon1 - cos_lon2 * sin_lon1
            let cos_dlon = cos_lon2 * cos_lon1 + sin_lon2 * sin_lon1;
            let sin_dlon = sin_lon2 * cos_lon1 - cos_lon2 * sin_lon1;

            // angular distance (theta)
            let cos_theta = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_dlon;
            theta[[i, j]] = cos_theta.clamp(-1.0, 1.0).acos(); // Clamp to avoid NaN from floating point errors

            // bearing (alpha)
            let x = cos_lat2 * sin_dlon;
            let y = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_dlon;

            // pi/2 - arctan2(x, y)
            alpha[[i, j]] = std::f32::consts::FRAC_PI_2 - x.atan2(y);
        }
    }

    (theta, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    // Helper function to check if two 2D arrays are approximately equal
    fn assert_arrays_approx_eq(actual: &Array2<f32>, expected: &Array2<f32>, epsilon: f32) {
        assert_eq!(
            actual.shape(),
            expected.shape(),
            "Arrays have different shapes"
        );

        assert_relative_eq!(
            actual.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            max_relative = epsilon,
            epsilon = epsilon
        );
    }

    // Helper function to convert nested Vec to Array2
    fn vec_to_array2(vec_data: &Vec<Vec<f32>>) -> Array2<f32> {
        let rows = vec_data.len();
        let cols = if rows > 0 { vec_data[0].len() } else { 0 };

        let mut array = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                array[[i, j]] = vec_data[i][j];
            }
        }
        array
    }

    #[test]
    fn test_case_1_basic_cardinal_points() {
        let coords1 = vec![GeographicalPoint::new(0.0, 0.0)];
        let coords2 = vec![
            GeographicalPoint::new(0.0, 90.0),
            GeographicalPoint::new(90.0, 0.0),
        ];

        let expected_theta = vec_to_array2(&vec![vec![1.5708, 1.5708]]);
        let expected_alpha = vec_to_array2(&vec![vec![0.0, 1.5708]]);

        let (theta, alpha) = angular_distance_and_bearing(&coords1, &coords2);

        let epsilon = 1e-4;
        assert_arrays_approx_eq(&theta, &expected_theta, epsilon);
        assert_arrays_approx_eq(&alpha, &expected_alpha, epsilon);
    }

    #[test]
    fn test_case_2_realistic_coordinates() {
        let coords1 = vec![
            GeographicalPoint::new(40.71, -74.0),
            GeographicalPoint::new(34.05, -118.2),
        ];
        let coords2 = vec![GeographicalPoint::new(48.85, 2.35)];

        let expected_theta = vec_to_array2(&vec![vec![0.9162], vec![1.4258]]);
        let expected_alpha = vec_to_array2(&vec![vec![0.6333], vec![0.961]]);

        let (theta, alpha) = angular_distance_and_bearing(&coords1, &coords2);

        let epsilon = 1e-4;
        assert_arrays_approx_eq(&theta, &expected_theta, epsilon);
        assert_arrays_approx_eq(&alpha, &expected_alpha, epsilon);
    }

    #[test]
    fn test_case_3_antipodal_points() {
        let coords1 = vec![GeographicalPoint::new(90.0, 0.0)];
        let coords2 = vec![GeographicalPoint::new(-90.0, 0.0)];

        let expected_theta = vec_to_array2(&vec![vec![3.1416]]); // π radians = 180 degrees
        let expected_alpha = vec_to_array2(&vec![vec![-1.5708]]); // -π/2 radians = -90 degrees

        let (theta, alpha) = angular_distance_and_bearing(&coords1, &coords2);

        let epsilon = 1e-4;
        assert_arrays_approx_eq(&theta, &expected_theta, epsilon);
        assert_arrays_approx_eq(&alpha, &expected_alpha, epsilon);
    }

    #[test]
    fn test_case_4_multiple_points_grid() {
        let coords1 = vec![
            GeographicalPoint::new(10.0, 10.0),
            GeographicalPoint::new(20.0, 20.0),
        ];
        let coords2 = vec![
            GeographicalPoint::new(10.0, 11.0),
            GeographicalPoint::new(21.0, 20.0),
            GeographicalPoint::new(15.0, 15.0),
        ];

        let expected_theta = vec_to_array2(&vec![
            vec![0.0172, 0.255, 0.1219],
            vec![0.2311, 0.0175, 0.1206],
        ]);
        let expected_alpha = vec_to_array2(&vec![
            vec![0.0015, 0.8728, 0.8064],
            vec![3.9747, 1.5708, 3.9371],
        ]);

        let (theta, alpha) = angular_distance_and_bearing(&coords1, &coords2);

        let epsilon = 1e-4;
        assert_arrays_approx_eq(&theta, &expected_theta, epsilon);
        assert_arrays_approx_eq(&alpha, &expected_alpha, epsilon);
    }

    #[test]
    fn test_case_5_edge_cases() {
        let coords1 = vec![
            GeographicalPoint::new(45.0, 45.0),
            GeographicalPoint::new(89.5, 10.0),
            GeographicalPoint::new(0.0, 179.5),
        ];
        let coords2 = vec![
            GeographicalPoint::new(45.0, 45.0),
            GeographicalPoint::new(-89.5, -170.0),
            GeographicalPoint::new(0.0, -179.5),
        ];

        let expected_theta = vec_to_array2(&vec![
            vec![0.0, 2.3633, 2.0994],
            vec![0.7783, 3.1416, 1.5794],
            vec![2.0893, 1.5622, 0.0175],
        ]);
        let expected_alpha = vec_to_array2(&vec![
            vec![1.5708, -1.5637, 0.6237],
            vec![-0.9549, 3.1416, 1.405],
            vec![2.1904, -1.5692, 0.0],
        ]);

        let (theta, alpha) = angular_distance_and_bearing(&coords1, &coords2);

        let epsilon = 1e-4;
        assert_arrays_approx_eq(&theta, &expected_theta, epsilon);
        assert_arrays_approx_eq(&alpha, &expected_alpha, epsilon);
    }
}
