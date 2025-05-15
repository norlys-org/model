use crate::{grid::GeographicalPoint, secs::R_EARTH, sphere::calc_angular_distance_and_bearing};

const MU0: f64 = 1e-7;

pub fn t_df(obs_locs: &[GeographicalPoint], secs_locs: &[GeographicalPoint]) {
    let nobs = obs_locs.len();
    let nsec = secs_locs.len();

    // Convert location data for calculations
    let obs_lat_lon: Vec<(f64, f64)> = obs_locs.iter().map(|p| (p.latitude, p.longitude)).collect();
    let secs_lat_lon: Vec<(f64, f64)> = secs_locs
        .iter()
        .map(|p| (p.latitude, p.longitude))
        .collect();

    let (theta, alpha) = calc_angular_distance_and_bearing(&obs_lat_lon, &secs_lat_lon);

    let sin_theta = theta
        .iter()
        .map(|row| row.iter().map(|&x| x.sin()).collect())
        .collect();
    let cos_theta = theta
        .iter()
        .map(|row| row.iter().map(|&x| x.cos()).collect())
        .collect();

    let x: Vec<f64> = obs_locs
        .iter()
        .zip(secs_locs.iter())
        .map(|(obs, sec)| obs.altitude / sec.altitude)
        .collect(); // only implement lower
}
