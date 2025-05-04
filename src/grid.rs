use std::ops::Range;

#[derive(Debug, Clone, Copy)]
pub struct GeographicalPoint {
    /// The longitude in degrees.
    pub longitude: f32,
    /// The latitude in degrees.
    pub latitude: f32,
    /// Altitude from the surface of the earth
    pub altitude: f32,
}

impl GeographicalPoint {
    pub fn new(latitude: f32, longitude: f32, altitude: f32) -> Self {
        Self { latitude, longitude, altitude }
    }
}

/// Return evenly spaced numbers over a specified interval.
///
/// # Panics
///
/// Panics if `end` is not strictly superior to `start`.
///
/// # Arguments
///
/// * `start` - The starting value of the sequence.
/// * `end` - The ending value of the sequence.
/// * `num` - The number of samples to generate.
fn linspace(start: f32, end: f32, num: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(num);

    if end <= start {
        panic!("End value needs to be strictly superior to start value.");
    }

    if num == 0 {
        return result;
    }

    if num == 1 {
        result.push(start);
        return result;
    }

    let step = (end - start) / ((num - 1) as f32);
    for i in 0..num {
        result.push(start + (i as f32) * step);
    }

    result
}

/// Generates a vector of geographical points based on the provided latitude and longitude ranges, steps, and altitude.
///
/// # Arguments
///
/// * `lat_range` - The range of latitudes.
/// * `lat_steps` - The number of latitude steps.
/// * `lon_range` - The range of longitudes.
/// * `lon_steps` - The number of longitude steps.
/// * `altitude` - The altitude above the Earth's surface in meters.
///
/// # Returns
///
/// A vector of `GeographicalPoint` instances.
pub fn geographical_grid(
    lat_range: Range<f32>,
    lat_steps: usize,
    lon_range: Range<f32>,
    lon_steps: usize,
    altitude: f32,
) -> Vec<GeographicalPoint> {
    let latitudes = linspace(lat_range.start, lat_range.end, lat_steps);
    let longitudes = linspace(lon_range.start, lon_range.end, lon_steps);
    let mut result = Vec::with_capacity(lat_steps * lon_steps);

    for lat in latitudes {
        for &lon in &longitudes {
            result.push(GeographicalPoint {
                longitude: lon,
                latitude: lat,
                altitude,
            })
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[macro_export]
    macro_rules! assert_relative_eq {
    ($a:expr, $b:expr) => {
        assert_relative_eq!($a, $b, 1e-6);
    };
    ($a:expr, $b:expr, $epsilon:expr) => {
        {
            let a_val = $a;
            let b_val = $b;
            let eps = $epsilon;
            let diff = if a_val > b_val { a_val - b_val } else { b_val - a_val };
            assert!(
                diff <= eps,
                "assertion failed: `(left ≈ right)` (left: `{:?}`, right: `{:?}`, expected diff: `{:?}`, actual diff: `{:?}`)",
                a_val, b_val, eps, diff
            );
        }
    };
}
    #[test]
    fn test_linspace_basic() {
        let result = linspace(0.0, 1.0, 5);
        assert_eq!(result.len(), 5);
        assert_relative_eq!(result[0], 0.0);
        assert_relative_eq!(result[1], 0.25);
        assert_relative_eq!(result[2], 0.5);
        assert_relative_eq!(result[3], 0.75);
        assert_relative_eq!(result[4], 1.0);
    }

    #[test]
    fn test_linspace_zero_elements() {
        let result = linspace(0.0, 1.0, 0);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_linspace_negative_range() {
        let result = linspace(-1.0, 1.0, 3);
        assert_eq!(result.len(), 3);
        assert_relative_eq!(result[0], -1.0);
        assert_relative_eq!(result[1], 0.0);
        assert_relative_eq!(result[2], 1.0);
    }

    #[test]
    #[should_panic]
    fn test_linspace_invalid_range() {
        linspace(1.0, 0.0, 5);
    }

    #[test]
    fn test_geographical_grid_basic() {
        let lat_range = 0.0..90.0;
        let lat_steps = 3;
        let lon_range = 0.0..180.0;
        let lon_steps = 2;
        let altitude = 100.0;
        let result = geographical_grid(lat_range, lat_steps, lon_range, lon_steps, altitude);
        assert_eq!(result.len(), 6);
        assert_relative_eq!(result[0].latitude, 0.0);
        assert_relative_eq!(result[0].longitude, 0.0);
        assert_relative_eq!(result[0].altitude, altitude);
        assert_relative_eq!(result[5].latitude, 90.0);
        assert_relative_eq!(result[5].longitude, 180.0);
        assert_relative_eq!(result[5].altitude, altitude);
    }
}
