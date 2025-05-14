use crate::{
    grid::GeographicalPoint,
    sphere::{angular_distance, bearing},
};

pub fn t_df(obs_locs: &[GeographicalPoint], secs_locs: &[GeographicalPoint]) -> Vec<Vec<Vec<f64>>> {
    let nobs = obs_locs.len();
    let nsec = secs_locs.len();

    // Convert location data for calculations
    let obs_lat_lon: Vec<(f64, f64)> = obs_locs.iter().map(|p| (p.latitude, p.longitude)).collect();
    let secs_lat_lon: Vec<(f64, f64)> = secs_locs
        .iter()
        .map(|p| (p.latitude, p.longitude))
        .collect();

    let theta = angular_distance(&obs_lat_lon, &secs_lat_lon);
    let alpha = bearing(&obs_lat_lon, &secs_lat_lon);

    let sin_theta = theta
        .iter()
        .map(|row| row.iter().map(|&x| x.sin()).collect())
        .collect();
    let cos_theta = theta
        .iter()
        .map(|row| row.iter().map(|&x| x.cos()).collect())
        .collect();

    // only implement lower
}
