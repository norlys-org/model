/// Calculates the angular distance between two sets of lat/lon points
///
/// # Arguments
/// * `coords1` - First set of points as a vector of (lat, lon) tuples in degrees
/// * `coords2` - Second set of points as a vector of (lat, lon) tuples in degrees
///
/// # Returns
/// A matrix (Vec<Vec<f64>>) of angular distances in radians
pub fn angular_distance(coords1: &[(f64, f64)], coords2: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let mut result = Vec::with_capacity(coords1.len());

    for &(lat1_deg, lon1_deg) in coords1 {
        let lat1_rad = lat1_deg.to_radians();
        let lon1_rad = lon1_deg.to_radians();

        let mut row = Vec::with_capacity(coords2.len());

        for &(lat2_deg, lon2_deg) in coords2 {
            let lat2_rad = lat2_deg.to_radians();
            let lon2_rad = lon2_deg.to_radians();

            let dlon = lon2_rad - lon1_rad;

            let theta = (lat1_rad.sin() * lat2_rad.sin()
                + lat1_rad.cos() * lat2_rad.cos() * dlon.cos())
            .acos();

            row.push(theta);
        }

        result.push(row);
    }

    result
}

/// Calculates the bearing (direction) from each point in `coords1` to each point in `coords2`
///
/// # Arguments
/// * `coords1` - First set of points as a vector of (lat, lon) tuples in degrees
/// * `coords2` - Second set of points as a vector of (lat, lon) tuples in degrees
///
/// # Returns
/// A matrix (Vec<Vec<f64>>) of bearings in radians, measured clockwise from true north
pub fn bearing(coords1: &[(f64, f64)], coords2: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let mut result = Vec::with_capacity(coords1.len());

    for &(lat1_deg, lon1_deg) in coords1 {
        let lat1_rad = (lat1_deg as f64).to_radians();
        let lon1_rad = (lon1_deg as f64).to_radians();

        let mut row = Vec::with_capacity(coords2.len());

        for &(lat2_deg, lon2_deg) in coords2 {
            let lat2_rad = (lat2_deg as f64).to_radians();
            let lon2_rad = (lon2_deg as f64).to_radians();

            let dlon = lon2_rad - lon1_rad;

            let y = dlon.sin() * lat2_rad.cos();
            let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * dlon.cos();

            let alpha = y.atan2(x); // true bearing (in radians, from -π to π)
            row.push(alpha as f64);
        }

        result.push(row);
    }

    result
}

#[cfg(test)]
mod tests {
    use crate::helpers::test::assert_vec_approx_eq;

    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_bearing_basic_cardinal_directions() {
        let coords1 = vec![(0.0, 0.0)];
        let coords2 = vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];

        let expected = vec![vec![0.0_f64, 1.5707963_f64, PI, -1.5707963_f64]];

        let result = bearing(&coords1, &coords2);
        assert_vec_approx_eq(&result, &expected, 1e-4);
    }

    #[test]
    fn test_bearing_realistic_points() {
        let coords1 = vec![(40.71, -74.0), (34.05, -118.2), (51.50, -0.1)];
        let coords2 = vec![(48.85, 2.35)];

        let expected = vec![vec![0.9375], vec![0.6098], vec![2.5905]];

        let result = bearing(&coords1, &coords2);
        assert_vec_approx_eq(&result, &expected, 1e-4);
    }

    #[test]
    fn test_bearing_multiple_grid() {
        let coords1 = vec![(10.0, 10.0), (20.0, 20.0)];
        let coords2 = vec![(10.0, 11.0), (21.0, 20.0), (15.0, 15.0)];

        let expected = vec![vec![1.5693, 0.6980, 0.7644], vec![-2.4039, 0.0, -2.3663]];

        let result = bearing(&coords1, &coords2);
        assert_vec_approx_eq(&result, &expected, 1e-3);
    }

    #[test]
    fn test_bearing_edge_cases() {
        let coords1 = vec![
            (45.0, 45.0),
            (89.5, 10.0), // Near North Pole
            (0.0, 179.5), // Near Antimeridian
        ];
        let coords2 = vec![
            (45.0, 45.0),
            (-89.5, -170.0), // Near South Pole
            (0.0, -179.5),
        ];

        let expected = vec![
            vec![0.0, 3.1345, 0.9471],
            vec![2.5257, -1.5708, 0.1658],
            vec![-0.6196, 3.1400, 1.5708],
        ];

        let result = bearing(&coords1, &coords2);
        assert_vec_approx_eq(&result, &expected, 1e-3);
    }

    #[test]
    fn test_angular_distance_identical_and_antipodal_points() {
        let coords1 = vec![(0.0, 0.0), (90.0, 0.0), (45.0, 45.0)];
        let coords2 = vec![(0.0, 0.0), (-90.0, 0.0), (-45.0, -135.0), (45.0, 45.0)];

        let expected = vec![
            vec![0.0_f64, 1.5708_f64, 2.0944_f64, 1.0472_f64],
            vec![1.5708_f64, 3.1416_f64, 2.3562_f64, 0.7854_f64],
            vec![1.0472_f64, 2.3562_f64, 3.1416_f64, 0.0_f64],
        ];

        let result = angular_distance(&coords1, &coords2);
        assert_vec_approx_eq(&result, &expected, 5e-4);
    }

    #[test]
    fn test_angular_distance_equator_and_meridian_distances() {
        let coords1 = vec![(0.0, 0.0), (0.0, 0.0)];
        let coords2 = vec![(0.0, 90.0), (0.0, 180.0), (90.0, 0.0)];

        let expected = vec![
            vec![1.5708_f64, 3.1416_f64, 1.5708_f64],
            vec![1.5708_f64, 3.1416_f64, 1.5708_f64],
        ];

        let result = angular_distance(&coords1, &coords2);
        assert_vec_approx_eq(&result, &expected, 5e-4);
    }

    #[test]
    fn test_angular_distance_realistic_points_distance() {
        let coords1 = vec![(40.71, -74.0), (34.05, -118.2), (51.50, -0.1)];
        let coords2 = vec![(48.85, 2.35)];

        let expected = vec![vec![0.9162_f64], vec![1.4258_f64], vec![0.0537_f64]];

        let result = angular_distance(&coords1, &coords2);
        assert_vec_approx_eq(&result, &expected, 5e-4);
    }

    #[test]
    fn test_angular_distance_grid_calculation_various_distances() {
        let coords1 = vec![(10.0, 10.0), (-20.0, -30.0)];
        let coords2 = vec![(10.0, 11.0), (80.0, 10.0), (-20.001, -30.001)];

        let expected = vec![
            vec![0.0172_f64, 1.2217_f64, 0.8639_f64],
            vec![0.8776_f64, 1.7842_f64, 0.0_f64],
        ];

        let result = angular_distance(&coords1, &coords2);
        assert_vec_approx_eq(&result, &expected, 5e-4);
    }
}
