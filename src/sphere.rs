/// Calculates the angular distance between two sets of lat/lon points
///
/// # Arguments
/// * `coords1` - First set of points as a vector of (lat, lon) tuples in degrees
/// * `coords2` - Second set of points as a vector of (lat, lon) tuples in degrees
///
/// # Returns
/// A matrix (Vec<Vec<f64>>) of angular distances in radians
fn angular_distance(coords1: &[(f64, f64)], coords2: &[(f64, f64)]) -> Vec<Vec<f64>> {
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

pub fn bearing(coords1: &[(f32, f32)], coords2: &[(f32, f32)]) {}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_zero_distance() {
        let coords = vec![(0.0, 0.0)];
        let result = angular_distance(&coords, &coords);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 1);
        assert!(approx_eq(result[0][0], 0.0));
    }

    #[test]
    fn test_equator_90_degrees_apart() {
        let point1 = vec![(0.0, 0.0)];
        let point2 = vec![(0.0, 90.0)];
        let result = angular_distance(&point1, &point2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 1);
        assert!(approx_eq(result[0][0], PI / 2.0)); // 90 degrees = π/2 radians
    }

    #[test]
    fn test_multiple_points() {
        let coords1 = vec![(0.0, 0.0), (90.0, 0.0)];
        let coords2 = vec![(0.0, 90.0), (0.0, 0.0)];

        let result = angular_distance(&coords1, &coords2);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
        assert_eq!(result[1].len(), 2);

        // First row
        assert!(approx_eq(result[0][0], PI / 2.0)); // (0,0) to (0,90)
        assert!(approx_eq(result[0][1], 0.0)); // (0,0) to (0,0)

        // Second row
        assert!(approx_eq(result[1][0], PI / 2.0)); // (90,0) to (0,90)
        assert!(approx_eq(result[1][1], PI / 2.0)); // (90,0) to (0,0)
    }
}
