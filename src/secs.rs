use crate::{
    grid::{geographical_grid, GeographicalPoint},
    matrix::t_df,
};
use std::ops::Range;

pub const R_EARTH: f32 = 6371e3;

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
    let secs_locs = geographical_grid(0.0..10.0, 11, 0.0..10.0, 11, 110000f32);
    let obs_locs: Vec<GeographicalPoint> = observations
        .iter()
        .map(|obs| GeographicalPoint {
            latitude: obs.latitude,
            longitude: obs.longitude,
            radius: R_EARTH, // Assume observations are at Earth's surface
        })
        .collect();

    t_df(&obs_locs, &secs_locs);
}
