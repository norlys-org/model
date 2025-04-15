use crate::sphere::{angular_distance, bearing};

pub struct Point {
    /// The longitude in degrees.
    longitude: f32,
    /// The latitude in degrees.
    latitude: f32,
    /// Radius from center of the earth
    radius: f32,
}

pub fn T_df(obs_locs: &[Point], secs_locs: &[Point]) {
    let obs_lat_lon: Vec<(f32, f32)> = obs_locs.iter().map(|p| (p.latitude, p.longitude)).collect();

    let secs_lat_lon: Vec<(f32, f32)> = secs_locs
        .iter()
        .map(|p| (p.latitude, p.longitude))
        .collect();

    let theta = angular_distance(&obs_lat_lon, &secs_lat_lon);
    let alpha = bearing(&obs_lat_lon, &secs_lat_lon);
}
