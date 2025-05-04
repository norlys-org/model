use std::f32::consts::PI;

use crate::grid::GeographicalPoint;

/// Calculates the angular distance between two sets of lat/lon points
///
/// # Arguments
/// * `coords1` - First set of points as a vector of (lat, lon) tuples in degrees
/// * `coords2` - Second set of points as a vector of (lat, lon) tuples in degrees
///
/// # Returns
/// A matrix (Vec<Vec<f32>>) of angular distances in radians
pub fn calc_angular_distance(
    coords1: &[GeographicalPoint],
    coords2: &[GeographicalPoint],
) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(coords1.len());

    for point1 in coords1 {
        let lat1 = point1.latitude.to_radians();
        let lon1 = point1.longitude.to_radians();

        let mut row = Vec::with_capacity(coords2.len());

        for point2 in coords2 {
            let lat2 = point2.latitude.to_radians();
            let lon2 = point2.longitude.to_radians();

            let dlon = lon2 - lon1;

            // Use the haversine formula for better numerical stability
            let theta = (lat1.sin() * lat2.sin() + lat1.cos() * lat2.cos() * dlon.cos())
                .clamp(-1.0, 1.0)
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
pub fn calc_bearing(coords1: &[GeographicalPoint], coords2: &[GeographicalPoint]) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(coords1.len());

    for point1 in coords1 {
        let lat1 = point1.latitude.to_radians();
        let lon1 = point1.longitude.to_radians();

        let mut row = Vec::with_capacity(coords2.len());

        for point2 in coords2 {
            let lat2 = point2.latitude.to_radians();
            let lon2 = point2.longitude.to_radians();

            let dlon = lon2 - lon1;

            // Match Python implementation (PI/2 - arctan2(...))
            let alpha = PI / 2.0
                - (dlon.sin() * lat2.cos())
                    .atan2(lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos());

            row.push(alpha);
        }

        result.push(row);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f32::consts::PI;

    // Helper function to normalize angle to [0, 2π)
    fn normalize_angle(angle: f32) -> f32 {
        let mut result = angle % (2.0 * PI);
        if result < 0.0 {
            result += 2.0 * PI;
        }
        result
    }

    // Tests for calc_angular_distance

    #[test]
    fn test_angular_distance_known_points() {
        // Test case 1: Distance between identical points should be zero
        let new_york = GeographicalPoint::new(40.7128, -74.0060, 0.0);
        let result1 = calc_angular_distance(&vec![new_york], &vec![new_york]);
        assert_relative_eq!(result1[0][0], 0.0, epsilon = 1e-6);

        // Test case 2: Distance between known city pairs
        let london = GeographicalPoint::new(51.5074, -0.1278, 0.0);
        let tokyo = GeographicalPoint::new(35.6762, 139.6503, 0.0);

        let result2 = calc_angular_distance(&vec![new_york], &vec![london, tokyo]);

        // New York to London: ~5570 km, angular distance ~0.873 radians
        // New York to Tokyo: ~10,850 km, angular distance ~1.703 radians
        assert_relative_eq!(result2[0][0], 0.873, epsilon = 0.01); // NY to London
        assert_relative_eq!(result2[0][1], 1.703, epsilon = 0.01); // NY to Tokyo
    }

    #[test]
    fn test_angular_distance_edge_cases() {
        // Test case 1: Antipodal points should have distance of π radians (180 degrees)
        let point1 = GeographicalPoint::new(0.0, 0.0, 0.0);
        let point2 = GeographicalPoint::new(0.0, 180.0, 0.0);

        let result1 = calc_angular_distance(&vec![point1], &vec![point2]);
        assert_relative_eq!(result1[0][0], PI, epsilon = 1e-6);

        // Test case 2: Points at poles and equator
        let north_pole = GeographicalPoint::new(90.0, 0.0, 0.0);
        let equator_point = GeographicalPoint::new(0.0, 45.0, 0.0);

        let result2 = calc_angular_distance(&vec![north_pole], &vec![equator_point]);
        assert_relative_eq!(result2[0][0], PI / 2.0, epsilon = 1e-6); // 90 degrees
    }
}
