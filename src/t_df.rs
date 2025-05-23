use crate::geo::{GeographicalPoint, R_EARTH};
use crate::sphere::angular_distance_and_bearing;
use ndarray::{Array1, Array2, Array3, Zip};

/// Physical constant: permeability of free space (µ0)
const MU0: f64 = 1e-7;

/// Calculates the "Transfer Matrix" (T) for Divergence-Free Spherical Elementary Current Systems (SECS).
///
/// **What it does:**
/// This function determines the magnetic field influence that each hypothetical "elementary current"
/// located high above the Earth – ~110km – (at `secs_locs`) would have on each ground-based observation
/// point (`obs_locs`). It essentially builds a map of potential influences.
/// This function deals with "divergence-free" SECS, which represent currents that flow in closed loops.
///
/// **The Physics:**
/// The calculation is based on the fundamental principle that electric currents create magnetic fields.
/// This relationship is formally described by the Biot-Savart Law.
/// (See: https://en.wikipedia.org/wiki/Biot%E2%80%93Savart_law)
///
/// **Specific Formulas:**
/// The exact mathematical formulas implemented here are analytical solutions derived from the
/// Biot-Savart law specifically for the geometry of these spherical elementary currents. They come
/// from the work of Amm & Viljanen in their paper on using SECS for ionospheric field continuation.
/// The comments below reference specific equations from that paper (e.g., Eq. 9, 10, A.7, A.8).
/// (See: https://link.springer.com/content/pdf/10.1007/978-3-030-26732-2.pdf)
///
/// **The Transfer Matrix (Output):**
/// The function returns a 3D matrix `T`. Each element `T[i][k][j]` represents the magnetic field's
/// k-th component (where 0=Bx/North, 1=By/East, 2=Bz/Down) measured at the i-th observation point (`obs_locs[i]`),
/// *caused by* the j-th elementary current (`secs_locs[j]`) *if that current had a standard strength of 1 Ampere*.
///
/// # Arguments
/// * `obs_locs` - A slice of `GeographicalPoint` structures representing the observation locations (e.g., ground magnetometers).
/// * `obs_altitude` - Altitude above the Earth's surface at which observations are located in meters.
/// * `secs_locs` - A slice of `GeographicalPoint` structures representing the locations (poles) of the hypothetical Spherical Elementary Currents.
/// * `secs_altitude` - Altitude above the Earth's surface at which poles are located in meters.
///
/// # Returns
/// `Array3<f64>` representing the transfer matrix T, with dimensions [nobs][3][nsec].
pub fn t_df(
    obs_locs: &[GeographicalPoint],
    obs_altitude: f64,
    secs_locs: &[GeographicalPoint],
    secs_altitude: f64,
) -> Array3<f64> {
    let nobs = obs_locs.len();
    let nsec = secs_locs.len();

    // Convert location data for calculations
    let obs_lat_lon: Vec<(f64, f64)> = obs_locs.iter().map(|p| (p.latitude, p.longitude)).collect();
    let secs_lat_lon: Vec<(f64, f64)> = secs_locs
        .iter()
        .map(|p| (p.latitude, p.longitude))
        .collect();

    let (theta, alpha) = angular_distance_and_bearing(&obs_lat_lon, &secs_lat_lon);

    let sin_theta = theta.mapv(|x| x.sin());
    let cos_theta = theta.mapv(|x| x.cos()).into_shape((nobs * nsec,)).unwrap(); // cos + flatten

    let obs_r = Array1::from_iter(
        obs_locs
            .iter()
            .flat_map(|_| vec![obs_altitude + R_EARTH; secs_locs.len()]),
    );
    let sec_r = Array1::from_iter(
        secs_locs
            .iter()
            .flat_map(|_| vec![secs_altitude + R_EARTH; obs_locs.len()]),
    );

    let over = obs_altitude > secs_altitude;
    let x: Array1<f64> = if over {
        &sec_r / &obs_r
    } else {
        &obs_r / &sec_r
    };

    let factor: Array1<f64> =
        1.0 / (1.0 - 2.0 * x.clone() * &cos_theta + x.mapv(|val| val.powi(2))).mapv(f64::sqrt);

    let br: Array1<f64> = if over {
        // Amm & Viljanen: Equation A.7
        MU0 * &x / &obs_r * (factor.clone() - 1.0)
    } else {
        // Amm & Viljanen: Equation 9
        MU0 / &obs_r * (factor.clone() - 1.0)
    };

    let b_theta: Array1<f64> = if over {
        // Amm & Viljanen: Equation A.8
        -MU0 / &obs_r
            * ((&obs_r - &sec_r * &cos_theta)
                / (&obs_r.mapv(|v| v.powi(2)) - &obs_r.mapv(|v| v * 2.0) * &sec_r * &cos_theta
                    + &sec_r.mapv(|v| v.powi(2)))
                    .mapv(f64::sqrt))
            .mapv(|v| v - 1.0)
    } else {
        // Amm & Viljanen: Equation 10
        -MU0 / &obs_r * (factor * (x - &cos_theta) + &cos_theta)
    };

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

    let mut t = Array3::<f64>::zeros((obs_locs.len(), 3, secs_locs.len()));

    for i in 0..nobs {
        for j in 0..nsec {
            t[[i, 0, j]] = -b_theta_divided[[i, j]] * alpha[[i, j]].sin();
            t[[i, 1, j]] = -b_theta_divided[[i, j]] * alpha[[i, j]].cos();
            t[[i, 2, j]] = -br[[i, j]];
        }
    }

    t
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_t_df_few_points_under() {
        let t: Array3<f64> = t_df(
            // Substract R_EARTH in order to have same results as python code as `t_df`
            // adds R_EARTH
            &[
                GeographicalPoint::new(50.0, 20.0),
                GeographicalPoint::new(51.0, 21.0),
            ],
            3000.0 - R_EARTH,
            &[
                GeographicalPoint::new(10.0, 30.0),
                GeographicalPoint::new(11.0, 31.0),
                GeographicalPoint::new(12.0, 32.0),
            ],
            4000.0 - R_EARTH,
        );

        // values generated from the python code
        let expected: Array3<f64> = Array3::from_shape_vec(
            (2, 3, 3),
            vec![
                -3.67250861e-11,
                -3.67476077e-11,
                -3.67074095e-11,
                9.94787927e-12,
                1.11826106e-11,
                1.24566687e-11,
                -1.76265553e-11,
                -1.84618364e-11,
                -1.92994257e-11,
                -3.66568665e-11,
                -3.67415083e-11,
                -3.67687370e-11,
                8.73331388e-12,
                9.92220548e-12,
                1.11521060e-11,
                -1.68114853e-11,
                -1.76472830e-11,
                -1.84884113e-11,
            ],
        )
        .unwrap();

        assert_eq!(t.shape(), expected.shape(), "T have different shapes");
        assert_relative_eq!(
            t.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            max_relative = 1e-15
        );
    }

    #[test]
    fn test_t_df_few_points_over() {
        let t: Array3<f64> = t_df(
            // Substract R_EARTH in order to have same results as python code as `t_df`
            // adds R_EARTH
            &[
                GeographicalPoint::new(50.0, 20.0),
                GeographicalPoint::new(51.0, 21.0),
            ],
            4000.0 - R_EARTH,
            &[
                GeographicalPoint::new(10.0, 30.0),
                GeographicalPoint::new(11.0, 31.0),
                GeographicalPoint::new(12.0, 32.0),
            ],
            3000.0 - R_EARTH,
        );

        // values generated from the python code
        let expected: Array3<f64> = Array3::from_shape_vec(
            (2, 3, 3),
            vec![
                1.248881305890098e-11,
                1.257667569294579e-11,
                1.264169224772104e-11,
                -3.382897573340390e-12,
                -3.827189728374329e-12,
                -4.289961456907339e-12,
                -9.914937352303397e-12,
                -1.038478295394393e-11,
                -1.085592698421099e-11,
                1.238600079569024e-11,
                1.249640680745045e-11,
                1.258643444730852e-11,
                -2.950902327063451e-12,
                -3.374709473974890e-12,
                -3.817516262669339e-12,
                -9.456460477747721e-12,
                -9.926596671910809e-12,
                -1.039973133651685e-11,
            ],
        )
        .unwrap();

        assert_eq!(t.shape(), expected.shape(), "T have different shapes");
        assert_relative_eq!(
            t.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            max_relative = 1e-15
        );
    }
}
