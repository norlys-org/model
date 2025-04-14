use std::ops::Range;

/// Radius of the earth in meters
const EARTH_RADIUS: f32 = 6371e3;

#[derive(Debug, Clone, Copy)]
struct GeographicalPoint {
    /// The longitude in degrees.
    longitude: f32,
    /// The latitude in degrees.
    latitude: f32,
    /// The radius from the Earth's center in meters
    radius: f32,
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
pub fn geographical_point(
    lat_range: Range<f32>,
    lat_steps: usize,
    lon_range: Range<f32>,
    lon_steps: usize,
    altitude: f32,
) -> Vec<GeographicalPoint> {
    let latitudes = linspace(lat_range.start, lat_range.end, lat_steps);
    let longitudes = linspace(lon_range.start, lon_range.end, lon_steps);
    let mut result = Vec::with_capacity(lat_steps * lon_steps);

    let radius = EARTH_RADIUS + altitude;

    for lat in latitudes {
        for &lon in &longitudes {
            result.push(GeographicalPoint {
                longitude: lon,
                latitude: lat,
                radius,
            })
        }
    }

    result
}
