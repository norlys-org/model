use crate::{grid::GeographicalPoint, sphere::calc_angular_distance_and_bearing};
use ndarray::{Array1, Array2, Array3, ArrayView1, Zip};

const MU0: f64 = 1e-7;
pub const R_EARTH: f64 = 6371e3;

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

    let obs_r = Array1::from_iter(
        obs_locs
            .iter()
            .flat_map(|obs| vec![R_EARTH + obs.altitude; secs_locs.len()]),
    );
    let sec_r = Array1::from_iter(
        secs_locs
            .iter()
            .flat_map(|sec| vec![R_EARTH + sec.altitude; obs_locs.len()]),
    );

    // MARK: calc_t_df_under
    let x = &sec_r / &obs_r;
    let factor =
        1.0 / (1.0 - 2.0 * x.clone() * &cos_theta + x.mapv(|val| val.powi(2))).mapv(f64::sqrt);

    // Amm & Viljanen: Equation 9
    let br: Array1<f64> = MU0 / &obs_r * (factor.clone() - 1.0);

    // Amm & Viljanen: Equation 10
    let b_theta: Array1<f64> = -MU0 / &obs_r * (factor * (x - &cos_theta) + &cos_theta);

    let br = br.into_shape((obs_locs.len(), secs_locs.len())).unwrap();
    let b_theta = b_theta
        .into_shape((obs_locs.len(), secs_locs.len()))
        .unwrap();

    let mut b_theta_divided = Array2::<f64>::zeros(sin_theta.dim());
    Zip::from(&mut b_theta_divided)
        .and(&b_theta)
        .and(&sin_theta)
        .for_each(|result_val, &a, &b| {
            *result_val = if b == 0.0 { 0.0 } else { a / b };
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t_df() {
        t_df(
            &[
                GeographicalPoint {
                    latitude: 50.0,
                    longitude: 20.0,
                    altitude: 3000.0,
                },
                GeographicalPoint {
                    latitude: 51.0,
                    longitude: 21.0,
                    altitude: 3000.0,
                },
            ],
            &[
                GeographicalPoint {
                    latitude: 10.0,
                    longitude: 30.0,
                    altitude: 4000.0,
                },
                GeographicalPoint {
                    latitude: 11.0,
                    longitude: 31.0,
                    altitude: 3000.0,
                },
                GeographicalPoint {
                    latitude: 12.0,
                    longitude: 32.0,
                    altitude: 3000.0,
                },
            ],
        );
    }
}
