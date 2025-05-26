use std::ops::Range;

// Earth radius in meters
pub const R_EARTH: f32 = 6371e3;

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct GeographicalPoint {
    /// The longitude in degrees
    pub lon: f32,
    /// The latitude in degrees
    pub lat: f32,
}

impl GeographicalPoint {
    pub fn new(latitude: f32, longitude: f32) -> Self {
        Self {
            lat: latitude,
            lon: longitude,
        }
    }

    /// Returns latitude in radians
    pub fn lat_rad(&self) -> f32 {
        self.lat.to_radians()
    }

    /// Returns longitude in radians
    pub fn lon_rad(&self) -> f32 {
        self.lon.to_radians()
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
) -> Vec<GeographicalPoint> {
    let latitudes = linspace(lat_range.start, lat_range.end, lat_steps);
    let longitudes = linspace(lon_range.start, lon_range.end, lon_steps);
    let mut result = Vec::with_capacity(lat_steps * lon_steps);

    for lat in latitudes {
        for &lon in &longitudes {
            result.push(GeographicalPoint { lon, lat })
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
        let result = geographical_grid(lat_range, lat_steps, lon_range, lon_steps);
        assert_eq!(result.len(), 6);
        assert_relative_eq!(result[0].lat, 0.0);
        assert_relative_eq!(result[0].lon, 0.0);
        assert_relative_eq!(result[5].lat, 90.0);
        assert_relative_eq!(result[5].lon, 180.0);
    }
}
