use crate::grid::geographical_point;
use std::ops::Range;

pub struct Observation {
    /// The longitude in degrees.
    longitude: f32,
    /// The latitude in degrees.
    latitude: f32,
    // i vector (usually x magnetometer component) in nano teslas
    i: f32,
    // j vector (usually y magnetometer component) in nano teslas
    j: f32,
    // Radius from center of the earth to the obsevation (usually just the radius of the earth
    // since the measurements are conducted on the ground)
    radius: f32,
}

pub fn secs_interpolate(
    observations: &[Observation],
    lat_range: Range<f32>,
    lat_steps: usize,
    lon_range: Range<f32>,
    lon_steps: usize,
    prediction_altitude: f32,
) {
    let points = geographical_point(0.0..10.0, 11, 0.0..10.0, 11, 110000f32);
}
