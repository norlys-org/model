import numpy as np


def _calc_angular_distance_and_bearing(
    latlon1: np.ndarray, latlon2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the angular distance and bearing between two points.

    This function calculates the angular distance in radians
    between two latitude and longitude points. It also calculates
    the bearing (direction) from point 1 to point 2 going from the
    cartesian x-axis towards the cartesian y-axis.

    Parameters
    ----------
    latlon1 : ndarray of shape (N, 2)
        Array of N (latitude, longitude) points.
    latlon2 : ndarray of shape (M, 2)
        Array of M (latitude, longitude) points.

    Returns
    -------
    theta : ndarray of shape (N, M)
        Angular distance in radians between each point in latlon1 and latlon2.
    alpha : ndarray of shape (N, M)
        Bearing from each point in latlon1 to each point in latlon2.
    """
    latlon1_rad = np.deg2rad(latlon1)
    latlon2_rad = np.deg2rad(latlon2)

    lat1 = latlon1_rad[:, 0][:, None]
    lon1 = latlon1_rad[:, 1][:, None]
    lat2 = latlon2_rad[:, 0][None, :]
    lon2 = latlon2_rad[:, 1][None, :]

    cos_lat1 = np.cos(lat1)
    sin_lat1 = np.sin(lat1)
    cos_lat2 = np.cos(lat2)
    sin_lat2 = np.sin(lat2)

    dlon = lon2 - lon1
    cos_dlon = np.cos(dlon)
    sin_dlon = np.sin(dlon)

    # theta == angular distance between two points
    theta = np.arccos(sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_dlon)

    # alpha == bearing, going from point1 to point2
    #          angle (from cartesian x-axis (By), going towards y-axis (Bx))
    # Used to rotate the SEC coordinate frame into the observation coordinate
    # frame.
    # SEC coordinates are: theta (colatitude (+ away from North Pole)),
    #                      phi (longitude, + east), r (+ out)
    # Obs coordinates are: X (+ north), Y (+ east), Z (+ down)
    x = cos_lat2 * sin_dlon
    y = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_dlon
    alpha = np.pi / 2 - np.arctan2(x, y)

    return theta, alpha


def _calc_T_df_under(
    obs_r: np.ndarray, sec_r: np.ndarray, cos_theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """T matrix for over locations (obs_r <= sec_r)."""
    mu0_over_4pi = 1e-7
    # print(obs_r, sec_r)
    x = obs_r / sec_r
    print(x)
    print(cos_theta)
    factor = 1.0 / np.sqrt(1 - 2 * x * cos_theta + x**2)

    # Amm & Viljanen: Equation 9
    Br = mu0_over_4pi / obs_r * (factor - 1)

    # Amm & Viljanen: Equation 10
    Btheta = -mu0_over_4pi / obs_r * (factor * (x - cos_theta) + cos_theta)

    return Br, Btheta


def _calc_T_df_over(
    obs_r: np.ndarray, sec_r: np.ndarray, cos_theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """T matrix for over locations (obs_r > sec_r)."""
    mu0_over_4pi = 1e-7
    x = sec_r / obs_r
    factor = 1.0 / np.sqrt(1 - 2 * x * cos_theta + x**2)

    # Amm & Viljanen: Equation A.7
    Br = mu0_over_4pi * x / obs_r * (factor - 1)

    # Amm & Viljanen: Equation A.8
    Btheta = (
        -mu0_over_4pi
        / obs_r
        * (
            (obs_r - sec_r * cos_theta)
            / np.sqrt(obs_r**2 - 2 * obs_r * sec_r * cos_theta + sec_r**2)
            - 1
        )
    )

    return Br, Btheta


def T_df(obs_loc: np.ndarray, sec_loc: np.ndarray) -> np.ndarray:
    """Calculate the divergence free magnetic field transfer function.

    The transfer function goes from SEC location to observation location
    and assumes unit current SECs at the given locations.

    Parameters
    ----------
    obs_loc : ndarray (nobs, 3 [lat, lon, r])
        The locations of the observation points.

    sec_loc : ndarray (nsec, 3 [lat, lon, r])
        The locations of the SEC points.

    Returns
    -------
    ndarray (nobs, 3, nsec)
        The T transfer matrix.
    """
    nobs = len(obs_loc)
    nsec = len(sec_loc)

    theta, alpha = _calc_angular_distance_and_bearing(obs_loc[:, :2], sec_loc[:, :2])
    print(theta)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    Br = np.empty((nobs, nsec))
    Btheta = np.empty((nobs, nsec))

    # Over locations: obs_r > sec_r
    over_locs = obs_loc[:, 2][:, np.newaxis] > sec_loc[:, 2][np.newaxis, :]
    if np.any(over_locs):
        # We use np.where because we are broadcasting 1d arrays
        # over_locs is a 2d array of booleans
        over_indices = np.where(over_locs)
        obs_r = obs_loc[over_indices[0], 2]
        sec_r = sec_loc[over_indices[1], 2]
        Br[over_locs], Btheta[over_locs] = _calc_T_df_over(
            obs_r, sec_r, cos_theta[over_locs]
        )

    # Under locations: obs_r <= sec_r
    under_locs = ~over_locs
    if np.any(under_locs):
        # We use np.where because we are broadcasting 1d arrays
        # over_locs is a 2d array of booleans
        under_indices = np.where(under_locs)
        obs_r = obs_loc[under_indices[0], 2]
        sec_r = sec_loc[under_indices[1], 2]
        print(cos_theta)
        print(cos_theta[under_locs])
        Br[under_locs], Btheta[under_locs] = _calc_T_df_under(
            obs_r, sec_r, cos_theta[under_locs]
        )

    # If sin(theta) == 0: Btheta = 0
    # There is a possible 0/0 in the expansion when sec_loc == obs_loc
    Btheta = np.divide(
        Btheta, sin_theta, out=np.zeros_like(sin_theta), where=sin_theta != 0
    )

    # Transform back to Bx, By, Bz at each local point
    T = np.empty((nobs, 3, nsec))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))
    T[:, 0, :] = -Btheta * np.sin(alpha)
    T[:, 1, :] = -Btheta * np.cos(alpha)
    T[:, 2, :] = -Br

    return T


T_df(
    np.array([np.array([50, 20, 3000]), np.array([51, 21, 3000])]),
    np.array([np.array([10, 30, 4000]), np.array([11, 31, 4000])]),
)
