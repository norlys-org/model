/// Calculates the angular distance between two sets of lat/lon points
///
/// # Arguments
/// * `coords1` - First set of points as a vector of (lat, lon) tuples in degrees
/// * `coords2` - Second set of points as a vector of (lat, lon) tuples in degrees
///
/// # Returns
/// A matrix (Vec<Vec<f32>>) of angular distances in radians
pub fn angular_distance(coords1: &[(f32, f32)], coords2: &[(f32, f32)]) -> Vec<Vec<f32>> {
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
/// A matrix (Vec<Vec<f32>>) of bearings in radians, measured clockwise from true north
pub fn bearing(coords1: &[(f32, f32)], coords2: &[(f32, f32)]) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(coords1.len());

    for &(lat1_deg, lon1_deg) in coords1 {
        let lat1_rad = (lat1_deg as f32).to_radians();
        let lon1_rad = (lon1_deg as f32).to_radians();

        let mut row = Vec::with_capacity(coords2.len());

        for &(lat2_deg, lon2_deg) in coords2 {
            let lat2_rad = (lat2_deg as f32).to_radians();
            let lon2_rad = (lon2_deg as f32).to_radians();

            let dlon = lon2_rad - lon1_rad;

            let y = dlon.sin() * lat2_rad.cos();
            let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * dlon.cos();

            let alpha = y.atan2(x); // true bearing (in radians, from -π to π)
            row.push(alpha as f32);
        }

        result.push(row);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    const EPSILON: f32 = 1e-10;

    fn approx_eq(a: f32, b: f32) -> bool {
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

    #[test]
    fn test_bearing_self() {
        let coords = vec![(0.0, 0.0)];
        let result = bearing(&coords, &coords);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 1);
        // Bearing to self is undefined in theory, but this implementation returns 0
        assert!(approx_eq(result[0][0] as f32, 0.0));
    }

    #[test]
    fn test_bearing_symmetry() {
        let a = vec![(10.0, 20.0)];
        let b = vec![(15.0, 25.0)];
        let forward = bearing(&a, &b)[0][0];
        let backward = bearing(&b, &a)[0][0];

        // The backward bearing should be roughly opposite (±π)
        let diff = ((forward as f32 - backward as f32 + PI) % (2.0 * PI) - PI).abs();
        assert!(diff - PI < EPSILON);
    }
}
