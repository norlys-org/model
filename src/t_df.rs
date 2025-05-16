use crate::{grid::GeographicalPoint, sphere::calc_angular_distance_and_bearing};
use ndarray::{Array1, Array2, ArrayView1};

const MU0: f64 = 1e-7;
pub const R_EARTH: f64 = 6371e3;

fn calc_t_df_under(
    obs_r: &Array1<f64>,
    sec_r: &Array1<f64>,
    cos_theta: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let x = sec_r / obs_r;
    let factor =
        1.0 / (1.0 - 2.0 * x.clone() * cos_theta + x.mapv(|val| val.powi(2))).mapv(f64::sqrt);

    // Amm & Viljanen: Equation 9
    let br = MU0 / obs_r * (factor.clone() - 1.0);

    // Amm & Viljanen: Equation 10
    let btheta = -MU0 / obs_r * (factor * (x - cos_theta) + cos_theta);

    (br, btheta)
}

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

    let theta_array = Array2::from_shape_fn((nobs, nsec), |(i, j)| theta[i][j]);
    let sin_theta = theta_array.mapv(|x| x.sin());
    // Flatten + cos
    let cos_theta = Array1::from_iter(theta.iter().flat_map(|row| row.iter().map(|&x| x.cos())));

    let obs_r = Array1::from_iter(obs_locs.iter().map(|obs| R_EARTH + obs.altitude));
    let sec_r = Array1::from_iter(secs_locs.iter().map(|sec| R_EARTH + sec.altitude));

    let (br, btheta) = calc_t_df_under(&obs_r, &sec_r, &cos_theta);

    println!("{:?}", sin_theta);
    println!("{:?}", cos_theta);
    println!("Br: {:?}", br);
    println!("Btheta: {:?}", btheta);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t_df() {
        t_df(
            &[
                GeographicalPoint {
                    latitude: 10.0,
                    longitude: 20.0,
                    altitude: 3000.0,
                },
                GeographicalPoint {
                    latitude: 11.0,
                    longitude: 21.0,
                    altitude: 3000.0,
                },
            ],
            &[GeographicalPoint {
                latitude: 40.0,
                longitude: 60.0,
                altitude: 4000.0,
            }],
        );
    }
}
