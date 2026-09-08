# -*- coding: utf-8 -*-

"""
File Name:              rayleigh_engine.py
Description:            This code-file contains functions needed to process the data from TOA to the final Rayleigh-corrected reflectance product. 
                        The functions are designed to be modular and can be used in a variety of workflows.

Date Created:           August 25th, 2026
Author:                 William Wainwright, Arun M. Saranathan
Email:                  william.wainwright@ssaihq.com/william.wainwright@nasa.gov
"""


import numpy as np
import pyproj
import rasterio
import xarray as xr
from typing import Callable
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
import geopandas as gpd
from shapely.geometry import Polygon, box
from rasterio.transform import Affine
from scipy.ndimage import zoom
import dask.array as da


from .aq_error import Error_Handler


total_cloud = 0
total_quality = 0
total_ndwi = 0


@Error_Handler
def find_nearest_index(array, value):
    array = np.asarray(array).astype(float)
    idx = (np.abs(array - value)).argmin()
    return idx


@Error_Handler
def resample_NO2_fraction(no2f,height,width,ystart,ystep):
    return zoom(no2f,[height/len(no2f),width/len(no2f[0])], order=1).take(indices=range(ystart, ystart+ystep),axis=0)


def barometric_scale(
    sea_level_pressure: float | np.ndarray, altitude: float | np.ndarray
) -> float | np.ndarray:
    """Calculate atmospheric pressure at a given altitude using the barometric formula.

    Applies the isothermal barometric formula to estimate ambient atmospheric
    pressure based on sea-level pressure and elevation above sea level.

    Parameters
    ----------
    sea_level_pressure : float or numpy.ndarray
        Sea-level atmospheric pressure (typically in Pascals or mbar).
        Can be a scalar or a 2D spatial array.
    altitude : float or numpy.ndarray
        Elevation above sea level in meters. Can be a scalar or a 2D surface
        elevation array (e.g., extracted from a rasterio DEM dataset).

    Returns
    -------
    float or numpy.ndarray
        Atmospheric pressure at the specified altitude in the same units as
        `sea_level_pressure`.

    Raises
    ------
    Exception
        Reraises any evaluation exception after logging input argument values.

    Notes
    -----
    The pressure calculation uses the isothermal atmospheric equation:

    .. math::

        P = P_0 \\cdot \\exp\\left( \\frac{-g \\cdot M \\cdot h}{R \\cdot T_0} \\right)

    Where:
    - :math:`P_0` = Sea-level pressure (`sea_level_pressure`)
    - :math:`g` = Standard gravitational acceleration (:math:`9.81\\text{ m/s}^2`)
    - :math:`M` = Molar mass of dry air (:math:`0.02896\\text{ kg/mol}`)
    - :math:`h` = Altitude in meters (`altitude`)
    - :math:`R` = Universal gas constant (:math:`8.3145\\text{ J/(mol}\\cdot\\text{K)}`)
    - :math:`T_0` = Reference temperature (:math:`293\\text{ K}`)

    Examples
    --------
    Extract elevation using `rasterio` and convert coordinates via `pyproj`:

    >>> import pyproj
    >>> import rasterio
    >>> transformer = pyproj.Transformer.from_crs(
    ...     "EPSG:4326", "EPSG:32618", always_xy=True
    ... )
    >>> x_utm, y_utm = transformer.transform(-75.16, 39.95)
    >>> with rasterio.open("dem.tif") as src:
    ...     elev = list(src.sample([(x_utm, y_utm)]))[0][0]
    >>> pressure = barometric_scale(101325.0, elev)
    """
    temp_ref = 293.0  # Reference temperature in Kelvin (20 °C)
    grav_accel = 9.81  # Gravitational acceleration in m/s^2
    mol_mass = 0.02896  # Molar mass of dry air in kg/mol
    R_gas = 8.3145  # Universal gas constant in J/(mol*K)

    try:
        pressure = sea_level_pressure * np.exp(
            (-1.0 * grav_accel * mol_mass * altitude) / (R_gas * temp_ref)
        )
    except Exception:
        print(
            f"Error calculating barometric pressure for sea_level_pressure={sea_level_pressure}, altitude={altitude}"
        )
        raise

    return pressure


def get_safe_wavelength(
    target: int | float | str,
    available_keys: list[int | float | str] | set[int | float | str],
    threshold: int | float = 20,
) -> str:
    """Find the nearest wavelength key within a specified tolerance threshold.

    Searches a collection of available wavelength keys for the nearest entry to
    a target wavelength. If the closest match falls outside the acceptable
    nanometer threshold, a ValueError is raised.

    Parameters
    ----------
    target : int, float, or str
        Target wavelength in nanometers to match.
    available_keys : iterable of int, float, or str
        Collection of available wavelength keys (e.g., LUT keys or sensor band metadata).
    threshold : int or float, default=20
        Maximum allowable distance in nanometers between `target` and the nearest key.

    Returns
    -------
    str
        The matching nearest wavelength key formatted as a string.

    Raises
    ------
    ValueError
        If `available_keys` is empty, or if the absolute difference between `target`
        and the nearest available key exceeds `threshold`.

    Notes
    -----
    Inputs are rounded/converted to integer values prior to distance comparisons
    to ensure reliable string key mapping across dictionary lookups.

    Examples
    --------
    >>> available_luts = ["443", "555", "670", "865"]
    >>> get_safe_wavelength(440, available_luts, threshold=10)
    '443'
    """
    if not available_keys:
        raise ValueError("Cannot match target wavelength: 'available_keys' is empty.")

    target_int = int(round(float(target)))

    # Convert keys to integers and find the closest match to target
    available_ints = [int(round(float(k))) for k in available_keys]
    nearest = min(available_ints, key=lambda x: abs(x - target_int))

    # Enforce threshold boundary
    if abs(nearest - target_int) > threshold:
        raise ValueError(
            f"Wavelength mismatch: Target {target} nm is too far from nearest available "
            f"{nearest} nm (Threshold: +/-{threshold} nm)"
        )

    return str(nearest)


@Error_Handler
def interpolate(array: np.ndarray, step: int) -> np.ndarray:
    """Interpolate a coarse sub-sampled grid back to full 2D array resolution.

    Uses `scipy.interpolate.RegularGridInterpolator` for unbiased bilinear 2D
    grid interpolation. To prevent NoData edge-bleeding artifacts along spatial
    boundaries, invalid/NaN nodes in the coarse grid are filled with nearest valid
    neighbor values via Euclidean distance transform prior to interpolation.

    Parameters
    ----------
    array : numpy.ndarray
        2D target array containing coarse values populated at subsampled interval
        slices (e.g., `array[::step, ::step]`).
    step : int
        Subsampling interval stride (in pixels) used to extract the coarse grid.

    Returns
    -------
    numpy.ndarray
        Dense 2D interpolated array matching the shape of `array`.

    Raises
    ------
    ValueError
        If `array` is not a 2D matrix or if `step` is less than 1.

    Notes
    -----
    Requires `scipy.interpolate.RegularGridInterpolator` and
    `scipy.ndimage.distance_transform_edt`.
    """
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D array for spatial interpolation, got {array.ndim}D array."
        )

    if step < 1:
        raise ValueError(f"Step size must be an integer >= 1, got {step}.")

    # Fast path: step of 1 requires no subsampling or interpolation
    if step == 1:
        return array.copy()

    height, width = array.shape
    array = array.astype(float, copy=True)  # Ensure float type for NaN handling

    # Define coarse grid pixel coordinates
    rows = np.arange(0, height, step)
    cols = np.arange(0, width, step)

    # Extract the subsampled coarse grid slice
    coarse_grid = array[np.ix_(rows, cols)].copy()

    # Identify invalid/NoData nodes (NaNs or non-positive values)
    invalid_mask = np.isnan(coarse_grid) | (coarse_grid <= 0)

    # Propagate nearest valid node values to prevent edge-bleeding if NaNs exist
    if invalid_mask.any() and not invalid_mask.all():
        _, indices = distance_transform_edt(
            invalid_mask, return_distances=True, return_indices=True
        )
        coarse_grid = coarse_grid[tuple(indices)]

    # Construct 2D linear interpolator
    interp = RegularGridInterpolator(
        (rows, cols),
        coarse_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    # Broadcast full-resolution row and column indices for evaluation
    full_rows = np.arange(height)[:, None]
    full_cols = np.arange(width)[None, :]

    # Evaluate bilinear interpolation over the dense grid
    interpolated_array = interp((full_rows, full_cols))

    return interpolated_array


@Error_Handler
def interpolate_and_fill(
    rho_rc: np.ndarray, stepsize: int, fill: float | int = -32767
) -> np.ndarray:
    """Sub-sample interpolate a 2D array and restore NoData fill values.

    Masks non-positive and NaN values in the input array, applies spatial 2D
    interpolation if `stepsize > 1`, and guarantees that all invalid or original
    NoData pixel locations are uniformly assigned to a designated fill value.

    Parameters
    ----------
    rho_rc : numpy.ndarray
        Input 2D surface reflectance or radiance array.
    stepsize : int
        Subsampling interval stride (in pixels) for coarse-grid interpolation.
        If `stepsize <= 1`, interpolation is bypassed.
    fill : float or int, default=-32767
        NoData sentinel value assigned to uncomputed or invalid pixels.

    Returns
    -------
    numpy.ndarray
        Fully interpolated array with invalid locations set to the `fill` value.

    Notes
    -----
    This function modifies non-positive values (`<= 0`) into `np.nan` prior to
    interpolation to ensure non-physical optical values do not skew spatial results.
    """
    # Create copy to prevent mutating the caller's array in-place
    processed_array = rho_rc.astype(float, copy=True)

    # Convert non-positive values to NaNs to establish valid observation mask
    processed_array[processed_array <= 0] = np.nan
    valid_pixel_mask = ~np.isnan(processed_array)

    # Perform 2D grid interpolation for coarse sampling strides
    if stepsize > 1:
        processed_array = interpolate(processed_array, stepsize)

    # Re-apply fill values to originally invalid pixel coordinates and residual NaNs
    processed_array[~valid_pixel_mask] = fill
    processed_array[np.isnan(processed_array)] = fill

    return processed_array

@Error_Handler
def rayleigh_correction(
    wavelength: float | int,
    lut: dict,
    solar_irradiance: float | np.ndarray,
    airmass: float | np.ndarray,
    solar_zenith: float | np.ndarray,
    view_zenith: float | np.ndarray,
    relative_azimuth: float | np.ndarray,
    pressure: float | np.ndarray,
    wind_speed: float | np.ndarray,
    polarize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Rayleigh radiance/reflectance and atmospheric transmittance using LUT interpolation.

    Performs multi-dimensional linear interpolation over pre-computed Rayleigh
    Look-Up Tables (LUT) as a function of geometry (solar zenith, viewing zenith,
    and relative azimuth angles) and surface wind speed. Applies pressure-scaling
    corrections to account for non-standard sea-level atmospheric pressure.

    Parameters
    ----------
    wavelength : float or int
        Central spectral wavelength in nanometers.
    lut : dict
        Rayleigh lookup table dictionary containing grid axes (`theta_solar_zenith`,
        `theta_viewing`, `sigma_wind`, `optical_thickness`) and multi-dimensional
        Stokes vector arrays (`I`, `Q`, `U`).
    solar_irradiance : float or numpy.ndarray
        Top-of-atmosphere solar irradiance for the band, scaled for Earth-Sun distance.
    airmass : float or numpy.ndarray
        Airmass factor calculated from solar and viewing zenith angles.
    solar_zenith : float or numpy.ndarray
        Solar zenith angle(s) in degrees.
    view_zenith : float or numpy.ndarray
        Viewing (sensor) zenith angle(s) in degrees.
    relative_azimuth : float or numpy.ndarray
        Relative azimuth angle(s) between sensor and sun in degrees.
    pressure : float or numpy.ndarray
        Ambient atmospheric pressure in Pascals.
    wind_speed : float or numpy.ndarray
        Surface wind speed in meters per second.
    polarize : bool, default=False
        If True, returns full Stokes vector components (`I`, `Q`, `U`).
        If False, returns only intensity `I`.

    Returns
    -------
    tuple of numpy.ndarray
        If `polarize=False`:
            - **rayleigh_I**: Pressure-corrected Rayleigh radiance/reflectance (Stokes I).
            - **solar_correction_factor**: Downward solar atmospheric transmittance.
            - **view_correction_factor**: Upward sensor atmospheric transmittance.
        If `polarize=True`:
            - **rayleigh_I**: Stokes I component.
            - **rayleigh_Q**: Stokes Q polarization component.
            - **rayleigh_U**: Stokes U polarization component.
            - **solar_correction_factor**: Downward solar atmospheric transmittance.
            - **view_correction_factor**: Upward sensor atmospheric transmittance.

    Notes
    -----
    The pressure correction follows NASA atmospheric correction formulations
    (Section 6.1.2 of NASA Atmospheric Correction document):

    .. math::

        C_p = \\left( -(0.6543 - 1.608 \\tau_r) + (0.8192 - 1.2541 \\tau_r) \\ln(M) \\right) \\tau_r M

    where :math:`\\tau_r` is the baseline optical thickness and :math:`M` is the airmass.
    """
    # Extract LUT grid axes
    solz_list = np.asarray(lut["theta_solar_zenith"])
    view_list = np.asarray(lut["theta_viewing"])
    sigma_list = np.asarray(lut["sigma_wind"])

    # Locate nearest grid indices for solar zenith angle
    solz_index_lower_bound = np.clip(
        np.searchsorted(solz_list, solar_zenith) - 1, 0, len(solz_list) - 2
    )
    solz_index_upper_bound = solz_index_lower_bound + 1

    # Locate nearest grid indices for viewing zenith angle
    view_index_lower_bound = np.clip(
        np.searchsorted(view_list, view_zenith) - 1, 0, len(view_list) - 2
    )
    view_index_upper_bound = view_index_lower_bound + 1

    # Compute interpolation weights for solar zenith and viewing angles
    solz_gap = (solar_zenith - solz_list[solz_index_lower_bound]) / (
        solz_list[solz_index_upper_bound] - solz_list[solz_index_lower_bound]
    )
    view_gap = (view_zenith - view_list[view_index_lower_bound]) / (
        view_list[view_index_upper_bound] - view_list[view_index_lower_bound]
    )

    # Convert surface wind speed to sea-surface roughness parameter (sigma)
    sigma = 0.0731 * np.sqrt(np.abs(wind_speed))

    # Locate upper and lower bounds in sigma grid
    sigma_index_lower_bound = np.clip(
        np.searchsorted(sigma_list, sigma) - 1, 0, len(sigma_list) - 2
    )
    sigma_index_upper_bound = sigma_index_lower_bound + 1

    # Cumulative accumulators for Fourier order expansion
    ray_I_upper_sigma = np.zeros_like(solar_zenith)
    ray_I_lower_sigma = np.zeros_like(solar_zenith)

    if polarize:
        ray_Q_upper_sigma = np.zeros_like(solar_zenith)
        ray_Q_lower_sigma = np.zeros_like(solar_zenith)
        ray_U_upper_sigma = np.zeros_like(solar_zenith)
        ray_U_lower_sigma = np.zeros_like(solar_zenith)

    # Sum Fourier azimuthal orders (0, 1, 2)
    for order in range(0, 3):
        # Stokes I corner values (lower sigma bound)
        lower_sigma_I_low_solz_low_view = lut["I"][
            sigma_index_lower_bound, solz_index_lower_bound, order, view_index_lower_bound
        ]
        lower_sigma_I_low_solz_high_view = lut["I"][
            sigma_index_lower_bound, solz_index_lower_bound, order, view_index_upper_bound
        ]
        lower_sigma_I_high_solz_low_view = lut["I"][
            sigma_index_lower_bound, solz_index_upper_bound, order, view_index_lower_bound
        ]
        lower_sigma_I_high_solz_high_view = lut["I"][
            sigma_index_lower_bound, solz_index_upper_bound, order, view_index_upper_bound
        ]

        # Stokes I corner values (upper sigma bound)
        upper_sigma_I_low_solz_low_view = lut["I"][
            sigma_index_upper_bound, solz_index_lower_bound, order, view_index_lower_bound
        ]
        upper_sigma_I_low_solz_high_view = lut["I"][
            sigma_index_upper_bound, solz_index_lower_bound, order, view_index_upper_bound
        ]
        upper_sigma_I_high_solz_low_view = lut["I"][
            sigma_index_upper_bound, solz_index_upper_bound, order, view_index_lower_bound
        ]
        upper_sigma_I_high_solz_high_view = lut["I"][
            sigma_index_upper_bound, solz_index_upper_bound, order, view_index_upper_bound
        ]

        if polarize:
            # Stokes Q corner values (lower sigma bound)
            lower_sigma_Q_low_solz_low_view = lut["Q"][
                sigma_index_lower_bound, solz_index_lower_bound, order, view_index_lower_bound
            ]
            lower_sigma_Q_low_solz_high_view = lut["Q"][
                sigma_index_lower_bound, solz_index_lower_bound, order, view_index_upper_bound
            ]
            lower_sigma_Q_high_solz_low_view = lut["Q"][
                sigma_index_lower_bound, solz_index_upper_bound, order, view_index_lower_bound
            ]
            lower_sigma_Q_high_solz_high_view = lut["Q"][
                sigma_index_lower_bound, solz_index_upper_bound, order, view_index_upper_bound
            ]

            # Stokes Q corner values (upper sigma bound)
            upper_sigma_Q_low_solz_low_view = lut["Q"][
                sigma_index_upper_bound, solz_index_lower_bound, order, view_index_lower_bound
            ]
            upper_sigma_Q_low_solz_high_view = lut["Q"][
                sigma_index_upper_bound, solz_index_lower_bound, order, view_index_upper_bound
            ]
            upper_sigma_Q_high_solz_low_view = lut["Q"][
                sigma_index_upper_bound, solz_index_upper_bound, order, view_index_lower_bound
            ]
            upper_sigma_Q_high_solz_high_view = lut["Q"][
                sigma_index_upper_bound, solz_index_upper_bound, order, view_index_upper_bound
            ]

            # Stokes U corner values (lower sigma bound)
            lower_sigma_U_low_solz_low_view = lut["U"][
                sigma_index_lower_bound, solz_index_lower_bound, order, view_index_lower_bound
            ]
            lower_sigma_U_low_solz_high_view = lut["U"][
                sigma_index_lower_bound, solz_index_lower_bound, order, view_index_upper_bound
            ]
            lower_sigma_U_high_solz_low_view = lut["U"][
                sigma_index_lower_bound, solz_index_upper_bound, order, view_index_lower_bound
            ]
            lower_sigma_U_high_solz_high_view = lut["U"][
                sigma_index_lower_bound, solz_index_upper_bound, order, view_index_upper_bound
            ]

            # Stokes U corner values (upper sigma bound)
            upper_sigma_U_low_solz_low_view = lut["U"][
                sigma_index_upper_bound, solz_index_lower_bound, order, view_index_lower_bound
            ]
            upper_sigma_U_low_solz_high_view = lut["U"][
                sigma_index_upper_bound, solz_index_lower_bound, order, view_index_upper_bound
            ]
            upper_sigma_U_high_solz_low_view = lut["U"][
                sigma_index_upper_bound, solz_index_upper_bound, order, view_index_lower_bound
            ]
            upper_sigma_U_high_solz_high_view = lut["U"][
                sigma_index_upper_bound, solz_index_upper_bound, order, view_index_upper_bound
            ]

        # Bilinear angular interpolation weighted by cosine/sine azimuth expansion
        azimuth_rad = relative_azimuth * order * np.pi / 180.0

        ray_I_upper_sigma += (
            (1 - solz_gap) * (1 - view_gap) * upper_sigma_I_low_solz_low_view
            + solz_gap * view_gap * upper_sigma_I_high_solz_high_view
            + solz_gap * (1 - view_gap) * upper_sigma_I_high_solz_low_view
            + (1 - solz_gap) * view_gap * upper_sigma_I_low_solz_high_view
        ) * np.cos(azimuth_rad)

        ray_I_lower_sigma += (
            (1 - solz_gap) * (1 - view_gap) * lower_sigma_I_low_solz_low_view
            + solz_gap * view_gap * lower_sigma_I_high_solz_high_view
            + solz_gap * (1 - view_gap) * lower_sigma_I_high_solz_low_view
            + (1 - solz_gap) * view_gap * lower_sigma_I_low_solz_high_view
        ) * np.cos(azimuth_rad)

        if polarize:
            ray_Q_upper_sigma += (
                (1 - solz_gap) * (1 - view_gap) * upper_sigma_Q_low_solz_low_view
                + solz_gap * view_gap * upper_sigma_Q_high_solz_high_view
                + solz_gap * (1 - view_gap) * upper_sigma_Q_high_solz_low_view
                + (1 - solz_gap) * view_gap * upper_sigma_Q_low_solz_high_view
            ) * np.cos(azimuth_rad)

            ray_Q_lower_sigma += (
                (1 - solz_gap) * (1 - view_gap) * lower_sigma_Q_low_solz_low_view
                + solz_gap * view_gap * lower_sigma_Q_high_solz_high_view
                + solz_gap * (1 - view_gap) * lower_sigma_Q_high_solz_low_view
                + (1 - solz_gap) * view_gap * lower_sigma_Q_low_solz_high_view
            ) * np.cos(azimuth_rad)

            ray_U_upper_sigma += (
                (1 - solz_gap) * (1 - view_gap) * upper_sigma_U_low_solz_low_view
                + solz_gap * view_gap * upper_sigma_U_high_solz_high_view
                + solz_gap * (1 - view_gap) * upper_sigma_U_high_solz_low_view
                + (1 - solz_gap) * view_gap * upper_sigma_U_low_solz_high_view
            ) * np.sin(azimuth_rad)

            ray_U_lower_sigma += (
                (1 - solz_gap) * (1 - view_gap) * lower_sigma_U_low_solz_low_view
                + solz_gap * view_gap * lower_sigma_U_high_solz_high_view
                + solz_gap * (1 - view_gap) * lower_sigma_U_high_solz_low_view
                + (1 - solz_gap) * view_gap * lower_sigma_U_low_solz_high_view
            ) * np.sin(azimuth_rad)

    # Linear interpolation across surface roughness (wind speed / sigma)
    sig_diff = sigma_list[sigma_index_upper_bound] - sigma_list[sigma_index_lower_bound]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(
            sig_diff < 1e-5, 0.0, (sigma - sigma_list[sigma_index_lower_bound]) / sig_diff
        )

    interpolated_I = ray_I_lower_sigma + h * (ray_I_upper_sigma - ray_I_lower_sigma)

    if polarize:
        interpolated_Q = ray_Q_lower_sigma + h * (ray_Q_upper_sigma - ray_Q_lower_sigma)
        interpolated_U = ray_U_lower_sigma + h * (ray_U_upper_sigma - ray_U_lower_sigma)

    # Standard atmosphere baseline properties
    optical_thickness = float(lut["optical_thickness"])
    standard_pressure = 101325.0  # Pascals

    # Apply barometric pressure scaling adjustment
    coeff_p = (
        -(0.6543 - 1.608 * optical_thickness)
        + (0.8192 - 1.2541 * optical_thickness) * np.log(airmass)
    ) * optical_thickness * airmass
    pressure_factor = (1.0 - np.exp(-1.0 * coeff_p * pressure / standard_pressure)) / (
        1.0 - np.exp(-1.0 * coeff_p)
    )

    # Calculate solar and viewing path transmittance factors
    solar_correction_factor = np.exp(
        -0.5 * pressure / standard_pressure * optical_thickness / np.cos(np.deg2rad(solar_zenith))
    )
    view_correction_factor = np.exp(
        -0.5 * pressure / standard_pressure * optical_thickness / np.cos(np.deg2rad(view_zenith))
    )

    # Scale interpolated Stokes values by solar irradiance and pressure factor
    rayleigh_I = solar_irradiance * pressure_factor * interpolated_I

    if polarize:
        rayleigh_Q = solar_irradiance * pressure_factor * interpolated_Q
        rayleigh_U = solar_irradiance * pressure_factor * interpolated_U
        return rayleigh_I, rayleigh_Q, rayleigh_U, solar_correction_factor, view_correction_factor

    return rayleigh_I, solar_correction_factor, view_correction_factor


@Error_Handler
def gaseous_absorption(
    ancillary_data: dict, wavelength: float | int
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, float]]:
    """Factory function generating gaseous transmittance calculations for a wavelength.

    Pre-computes optical thickness parameters for Ozone (O3) and Nitrogen Dioxide (NO2)
    based on spectral band opacity lookup tables and atmospheric ancillary inputs. Returns
    a inner calculation closure that evaluates spatial geometric transmittance arrays.

    Parameters
    ----------
    ancillary_data : dict
        Auxiliary atmospheric inputs dictionary containing:
        - `ozone`: Column density in Dobson Units (DU).
        - `NO2_stratosphere`: Stratospheric NO2 concentration (in 1e15 mol/cm^2).
        - `NO2_troposphere`: Tropospheric NO2 concentration (in 1e15 mol/cm^2).
        - `NO2_fraction_200m`: Fraction of tropospheric NO2 above 200m altitude.
        - `sensor_data`: Nested dictionary of spectral band properties including
          `ozone_opacity` and `NO2_opacity`.
    wavelength : float or int
        Central band spectral wavelength in nanometers.

    Returns
    -------
    Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray, float]]
        A parameterized inner closure function `gas_func` taking spatial arrays
        `(solar_zenith, view_zenith, airmass)` and returning atmospheric transmittances.

    Notes
    -----
    - Ozone column density is converted from Dobson Units (DU) to :math:`\\text{atm}\\cdot\\text{cm}`
      by dividing by 1000.
    - NO2 absorption cross-sections are temperature-corrected to 285 K (troposphere)
      and 225 K (stratosphere) following standard NASA `l2gen` formulations.
    """
    # Safe lookup for the closest available sensor band
    sensor_wave = get_safe_wavelength(wavelength, ancillary_data["sensor_data"].keys(), threshold=10)

    # ------------------------------------------------------------------
    # 1. Ozone (O3) Optical Thickness Pre-computation
    # ------------------------------------------------------------------
    # Convert ozone column density from Dobson Units (DU) to atm-cm (1 DU = 1e-3 atm-cm)
    ozone_concentration = ancillary_data["ozone"] / 1000.0
    ozone_absorption = ancillary_data["sensor_data"][sensor_wave]["ozone_opacity"]
    ozone_optical_thickness = (ozone_concentration * ozone_absorption).astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Nitrogen Dioxide (NO2) Optical Thickness Pre-computation
    # ------------------------------------------------------------------
    no2_stratosphere = ancillary_data["NO2_stratosphere"] * 1e15
    no2_troposphere = ancillary_data["NO2_troposphere"] * 1e15
    no2_opacity = ancillary_data["sensor_data"][sensor_wave]["NO2_opacity"]
    no2_fraction_200m = ancillary_data["NO2_fraction_200m"]

    # Compute total tropospheric NO2 column content residing above 200m
    no2_above_200m = (no2_fraction_200m * no2_troposphere).astype(np.float32)

    # Temperature-corrected NO2 absorption cross-sections (l2gen standard)
    a_285 = (no2_opacity * (1.0 - 0.003 * (285.0 - 294.0))).astype(np.float32)  # Tropospheric layer (~285 K)
    a_225 = (no2_opacity * (1.0 - 0.003 * (225.0 - 294.0))).astype(np.float32)  # Stratospheric layer (~225 K)

    # Optical thickness for NO2 above 200m
    tau_to_200 = (a_285 * no2_above_200m + a_225 * no2_stratosphere).astype(np.float32)

    def gas_func(
        solar_zenith: np.ndarray, view_zenith: np.ndarray, airmass: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Calculate spatial Ozone and NO2 atmospheric transmittances.

        Parameters
        ----------
        solar_zenith : numpy.ndarray
            Solar zenith angle array in degrees.
        view_zenith : numpy.ndarray
            Viewing zenith angle array in degrees.
        airmass : numpy.ndarray
            Relative optical atmospheric path (airmass factor).

        Returns
        -------
        total_ozone_transmittance : numpy.ndarray
            Combined two-way (solar + sensor path) ozone transmittance grid.
        total_no2_transmittance : numpy.ndarray
            Combined two-way (solar + sensor path) NO2 transmittance grid.
        no2_fraction_200m : float
            Fraction of tropospheric NO2 column above 200m used in calculation.
        """
        # Ozone downward (solar) and upward (view) path transmittances
        ozone_trans_sol = np.exp(
            -1.0 * ozone_optical_thickness / np.cos(np.deg2rad(solar_zenith))
        )
        ozone_trans_view = np.exp(
            -1.0 * ozone_optical_thickness / np.cos(np.deg2rad(view_zenith))
        )
        total_ozone_transmittance = ozone_trans_sol * ozone_trans_view

        # NO2 downward (solar) and upward (view) path transmittances
        no2_trans_sol = (np.exp(-1.0 * tau_to_200 / np.cos(np.deg2rad(solar_zenith)))).astype(np.float32)
        no2_trans_view = (np.exp(-1.0 * tau_to_200 / np.cos(np.deg2rad(view_zenith)))).astype(np.float32)
        total_no2_transmittance = (no2_trans_sol * no2_trans_view).astype(np.float32)

        return total_ozone_transmittance, total_no2_transmittance, no2_fraction_200m

    return gas_func


@Error_Handler
def resample_gmao_ancillary(
    ancillary_data: dict,
    transform: Affine,
    width: int,
    height: int,
    ystart: int,
    ystep: int,
    crs: str = "EPSG:4326",
) -> dict:
    """
    Crop and resample GMAO and GEOS ancillary atmospheric grids to scene dimensions.

    Determines the geographic bounding coordinates of an image's four corners
    using `rasterio` and `pyproj`, slices the coarse GMAO and GEOS atmospheric
    grids to the scene's bounding region, and bilinearly interpolates (zooms)
    them to match the target scene spatial grid without relying on `osgeo`.

    Parameters
    ----------
    ancillary_data : dict
        Dictionary containing input atmospheric grids and coordinate vectors:
        - ``gmao_lat``, ``gmao_long`` : 1D arrays of GMAO coordinate axes.
        - ``geos_lat``, ``geos_long`` : 1D arrays of GEOS NO2 coordinate axes.
        - Atmospheric 2D variables: ``wind_speed``, ``pressure_sea_level``,
          ``relative_humidity``, ``precipitable_water``, ``ozone``,
          ``water_vapor``, ``NO2_stratosphere``, and ``NO2_troposphere``.
    transform : rasterio.transform.Affine
        Affine transformation matrix mapping image pixel coordinates to the CRS.
    width : int
        Horizontal dimension (number of columns) of the target image raster.
    height : int
        Vertical dimension (number of rows) of the target image raster.
    ystart : int
        Starting row index for slicing along the vertical (y) axis.
    ystep : int
        Number of rows to slice along the vertical (y) axis from `ystart`.
    crs : str, optional
        Coordinate reference system of the `transform`. Defaults to "EPSG:4326".

    Returns
    -------
    ancillary_data : dict
        Updated dictionary where atmospheric 2D variable arrays are cropped,
        resampled to shape `(ystep, width)`, and converted to `np.ndarray`.

    Notes
    -----
    Coordinate projections are calculated using `rasterio.transform.xy` and
    `pyproj.Transformer` to completely avoid dependencies on `osgeo` / GDAL.
    """
    # 1. Compute Geographic Corners using Rasterio & PyProj
    rows = [0, height, 0, height]
    cols = [0, 0, width, width]

    proj_x, proj_y = rasterio.transform.xy(transform, rows, cols, offset="center")

    if crs != "EPSG:4326":
        transformer = pyproj.Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True
        )
        long_corners, lat_corners = transformer.transform(proj_x, proj_y)
    else:
        long_corners, lat_corners = proj_x, proj_y

    # 2. Extract Lat/Lon Bounding Indices in GMAO Space
    gmao_lat = ancillary_data["gmao_lat"]
    gmao_long = ancillary_data["gmao_long"]

    gmao_lat_idx = sorted([find_nearest_index(gmao_lat, a) for a in lat_corners])
    gmao_lon_idx = sorted([find_nearest_index(gmao_long, a) for a in long_corners])

    gmao_row_min, gmao_row_max = min(gmao_lat_idx), max(gmao_lat_idx) + 1
    gmao_col_min, gmao_col_max = min(gmao_lon_idx), max(gmao_lon_idx) + 1

    # 3. Extract Lat/Lon Bounding Indices in GEOS Space
    geos_lat = ancillary_data["geos_lat"]
    geos_long = ancillary_data["geos_long"]

    geos_lat_idx = sorted([find_nearest_index(geos_lat, a) for a in lat_corners])
    geos_lon_idx = sorted([find_nearest_index(geos_long, a) for a in long_corners])

    geos_row_min, geos_row_max = min(geos_lat_idx), max(geos_lat_idx) + 1
    geos_col_min, geos_col_max = min(geos_lon_idx), max(geos_lon_idx) + 1

    # 4. Crop and Resample Standard GMAO Variables
    gmao_vars = [
        "wind_speed",
        "pressure_sea_level",
        "relative_humidity",
        "precipitable_water",
        "ozone",
        "water_vapor",
    ]

    for variable in gmao_vars:
        var_array = ancillary_data[variable][
            gmao_row_min:gmao_row_max, gmao_col_min:gmao_col_max
        ]
        zoom_factors = (height / var_array.shape[0], width / var_array.shape[1])

        resampled = zoom(var_array, zoom_factors, order=1)
        ancillary_data[variable] = resampled[ystart : ystart + ystep, :]

    # 5. Crop and Resample GEOS NO2 Variables
    geos_vars = ["NO2_stratosphere", "NO2_troposphere"]

    for variable in geos_vars:
        var_array = ancillary_data[variable][
            geos_row_min:geos_row_max, geos_col_min:geos_col_max
        ]
        zoom_factors = (height / var_array.shape[0], width / var_array.shape[1])

        resampled = zoom(var_array, zoom_factors, order=1)
        ancillary_data[variable] = resampled[ystart : ystart + ystep, :]

    print("Resampled ancillary arrays successfully.")

    return ancillary_data



@Error_Handler
def resample_old_ancillary(
    ancillary_data: dict,
    transform: Affine,
    width: int,
    height: int,
    ystart: int,
    ystep: int,
    crs: str = "EPSG:4326",
) -> dict:
    """Crop and resample standard 1-degree and 0.25-degree NO2 ancillary grids.

    Calculates the geographic corner coordinates using `rasterio` and `pyproj`,
    maps latitude/longitude values to array index coordinates based on standard
    global raster indexing rules, slices the bounding regions, and bilinearly
    interpolates (zooms) them to match target scene dimensions.

    Parameters
    ----------
    ancillary_data : dict
        Dictionary containing input atmospheric 2D grid arrays:
        - Standard 1-degree variables: ``wind_speed``, ``pressure_sea_level``,
          ``relative_humidity``, ``precipitable_water``, ``ozone``,
          ``water_vapor``.
        - 0.25-degree NO2 variables: ``NO2_stratosphere``, ``NO2_troposphere``.
    transform : rasterio.transform.Affine
        Affine transformation matrix mapping image pixel coordinates to the CRS.
    width : int
        Horizontal dimension (number of columns) of the target image raster.
    height : int
        Vertical dimension (number of rows) of the target image raster.
    ystart : int
        Starting row index for slicing along the vertical (y) axis.
    ystep : int
        Number of rows to slice along the vertical (y) axis from `ystart`.
    crs : str, optional
        Coordinate reference system of the `transform`. Defaults to "EPSG:4326".

    Returns
    -------
    ancillary_data : dict
        Updated dictionary where atmospheric 2D variable arrays are cropped,
        resampled to shape `(ystep, width)`, and converted to `np.ndarray`.

    Notes
    -----
    Indexing conventions applied:
    - Standard 1-degree grid: `row = abs(round(lat) - 90)`, `col = round(long) + 180`
    - NO2 0.25-degree grid: `row = abs(round(4 * (lat - 90)))`, `col = round(4 * (long + 180))`
    This function avoids `osgeo` dependencies by using native `rasterio` and `pyproj`.
    """
    # 1. Calculate Geographic Corner Coordinates via Rasterio & PyProj
    rows = [0, height, 0, height]
    cols = [0, 0, width, width]

    proj_x, proj_y = rasterio.transform.xy(transform, rows, cols, offset="center")

    if crs != "EPSG:4326":
        transformer = pyproj.Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True
        )
        long_corners, lat_corners = transformer.transform(proj_x, proj_y)
    else:
        long_corners, lat_corners = proj_x, proj_y

    # 2. Compute Array Indices for 1-Degree Global Grids
    corner_lat_idx = [abs(round(lat) - 90) for lat in lat_corners]
    corner_long_idx = [round(long) + 180 for long in long_corners]

    lat_min, lat_max = min(corner_lat_idx), max(corner_lat_idx) + 1
    long_min, long_max = min(corner_long_idx), max(corner_long_idx) + 1

    # 3. Compute Array Indices for 0.25-Degree NO2 Grids
    corner_lat_idx_no2 = [abs(round(4 * (lat - 90))) for lat in lat_corners]
    corner_long_idx_no2 = [round(4 * (long + 180)) for long in long_corners]

    no2_lat_min, no2_lat_max = min(corner_lat_idx_no2), max(corner_lat_idx_no2) + 1
    no2_long_min, no2_long_max = min(corner_long_idx_no2), max(corner_long_idx_no2) + 1

    # 4. Crop and Resample Standard 1-Degree Grids
    standard_vars = [
        "wind_speed",
        "pressure_sea_level",
        "relative_humidity",
        "precipitable_water",
        "ozone",
        "water_vapor",
    ]

    for variable in standard_vars:
        var_array = ancillary_data[variable][lat_min:lat_max, long_min:long_max]
        zoom_factors = (height / var_array.shape[0], width / var_array.shape[1])

        resampled = zoom(var_array, zoom_factors, order=1)
        ancillary_data[variable] = resampled[ystart : ystart + ystep, :]

    # 5. Crop and Resample 0.25-Degree NO2 Grids
    no2_vars = ["NO2_stratosphere", "NO2_troposphere"]

    for variable in no2_vars:
        var_array = ancillary_data[variable][
            no2_lat_min:no2_lat_max, no2_long_min:no2_long_max
        ]
        zoom_factors = (height / var_array.shape[0], width / var_array.shape[1])

        resampled = zoom(var_array, zoom_factors, order=1)
        ancillary_data[variable] = resampled[ystart : ystart + ystep, :]

    return ancillary_data


@Error_Handler
def patch_to_process(
    height: int,
    width: int,
    lat: np.ndarray,
    long: np.ndarray,
    stepsize: int,
    coarse_step: int,
    patch: tuple[float, float, float, float]
    | Polygon
    | gpd.GeoDataFrame
    | str
    | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate spatial processing masks for fine and coarse grids.

    Creates 2D binary indicator arrays (containing 1.0 where processing is
    required and NaN elsewhere) spaced at specified subsampling intervals.
    Optionally clips the active processing domain using a bounding box,
    Shapely polygon, GeoDataFrame, or vector file path using Shapely/GeoPandas
    without `osgeo` dependencies.

    Parameters
    ----------
    height : int
        Vertical dimension (number of rows) of the target image array.
    width : int
        Horizontal dimension (number of columns) of the target image array.
    lat : numpy.ndarray
        2D array of geographic latitudes in degrees North, shape `(height, width)`.
    long : numpy.ndarray
        2D array of geographic longitudes in degrees East, shape `(height, width)`.
    stepsize : int
        Sampling interval in pixels for the fine processing mask.
    coarse_step : int
        Sampling interval in pixels for the coarse processing mask.
    patch : tuple of float, shapely.geometry.Polygon, geopandas.GeoDataFrame, str, optional
        Spatial ROI geometry used to constrain the mask area. Can be provided as:
        - Bounding box tuple: `(min_lon, min_lat, max_lon, max_lat)`
        - `shapely.geometry.Polygon` object
        - `geopandas.GeoDataFrame` object
        - File path string to a vector file (e.g., GeoJSON, Shapefile) read via GeoPandas.
        Defaults to `None` (processes the entire grid).

    Returns
    -------
    mask : numpy.ndarray
        2D array of shape `(height, width)` containing 1.0 at active fine grid
        nodes and NaN elsewhere.
    coarse_mask : numpy.ndarray
        2D array of shape `(height, width)` containing 1.0 at active coarse grid
        nodes and NaN elsewhere.

    Notes
    -----
    Avoids `osgeo` / GDAL modules by leveraging native `numpy` boolean indexing
    and `shapely` / `geopandas` for vector operations.
    """
    # 1. Initialize Base Subsampled Grids
    mask = np.full((height, width), np.nan, dtype=np.float32)
    mask[::stepsize, ::stepsize] = 1.0

    coarse_mask = np.full((height, width), np.nan, dtype=np.float32)
    coarse_mask[::coarse_step, ::coarse_step] = 1.0

    # 2. Restrict Domain by Spatial Patch (if provided)
    if patch is not None:
        patch_geom = None

        if isinstance(patch, (list, tuple)) and len(patch) == 4:
            # Bounding box: (min_lon, min_lat, max_lon, max_lat)
            patch_geom = box(*patch)
        elif isinstance(patch, Polygon):
            patch_geom = patch
        elif isinstance(patch, gpd.GeoDataFrame):
            patch_geom = patch.unary_union
        elif isinstance(patch, str):
            gdf = gpd.read_file(patch)
            patch_geom = gdf.unary_union

        if patch_geom is not None:
            # Polygon or Box spatial filtering via pure Shapely / NumPy
            min_lon, min_lat, max_lon, max_lat = patch_geom.bounds

            in_patch = (
                (long >= min_lon)
                & (long <= max_lon)
                & (lat >= min_lat)
                & (lat <= max_lat)
            )

            # Mask out pixels outside the bounding box
            mask[~in_patch] = np.nan
            coarse_mask[~in_patch] = np.nan

    return mask, coarse_mask

def _process_chunk_block(
    image: np.ndarray,
    sol_zen: np.ndarray,
    sol_az: np.ndarray,
    view_zen: np.ndarray,
    view_az: np.ndarray,
    lat: np.ndarray,
    long: np.ndarray,
    *ancillary_chunks: np.ndarray,
    wavelength: float | int | None = None,
    lut_dict: dict | None = None,
    solar_irradiance: float | None = None,
    altitude: float = 0,
    no_gas_absorption: bool = False,
    coarse_step: int = 10,
    stepsize: int = 1,
    ancillary_keys: list[str] | None = None,
    ancillary_scalars: dict | None = None,
    **kwargs,
) -> np.ndarray:
    """Process a single 2D spatial chunk for Rayleigh and atmospheric correction.

    Performs coarse-grid lookup table (LUT) Rayleigh radiance calculations,
    gaseous absorption estimation, spatial interpolation, and final surface
    reflectance (`rho_rc`) computation on a 2D chunk.

    Parameters
    ----------
    image : numpy.ndarray
        2D sub-array of Top-Of-Atmosphere (TOA) reflectance for the target band.
    sol_zen : numpy.ndarray
        2D sub-array of solar zenith angles in degrees.
    sol_az : numpy.ndarray
        2D sub-array of solar azimuth angles in degrees.
    view_zen : numpy.ndarray
        2D sub-array of viewing zenith angles in degrees.
    view_az : numpy.ndarray
        2D sub-array of viewing azimuth angles in degrees.
    lat : numpy.ndarray
        2D sub-array of latitude coordinates in decimal degrees.
    long : numpy.ndarray
        2D sub-array of longitude coordinates in decimal degrees.
    *ancillary_chunks : numpy.ndarray
        Variable positional 2D sub-arrays corresponding to spatial ancillary
        rasters (e.g., pressure_sea_level, wind_speed, ozone, no2). The keys for
        these arrays are provided in `ancillary_keys`.
    wavelength : float or int, optional
        Target band wavelength in nanometers.
    lut_dict : dict, optional
        Look-up table parameters for Rayleigh scattering at the target wavelength.
    solar_irradiance : float, optional
        Solar irradiance for the target band corrected for Earth-Sun distance.
    altitude : float, default=0
        Surface altitude in meters above sea level.
    no_gas_absorption : bool, default=False
        If True, skips gaseous absorption transmittance calculation and sets it to 1.0.
    coarse_step : int, default=10
        Sub-sampling step size used to evaluate Rayleigh LUT on a coarse grid.
    stepsize : int, default=1
        Spatial step size of the input image pixel grid.
    ancillary_keys : list of str, optional
        List of dictionary keys mapping sequentially to each array in `ancillary_chunks`.
    ancillary_scalars : dict, optional
        Dictionary containing non-spatial ancillary parameters and metadata.
    **kwargs : dict
        Additional unused keyword arguments passed by `xr.apply_ufunc`.

    Returns
    -------
    numpy.ndarray
        2D float32 array of Rayleigh-corrected reflectance (`rho_rc`) matching
        the input chunk dimensions `(height, width)`.
    """
    height, width = image.shape

    # Reconstruct ancillary dictionary for this specific chunk
    ancillary_chunk_dict = dict(ancillary_scalars) if ancillary_scalars else {}
    if ancillary_keys:
        for key, chunk_arr in zip(ancillary_keys, ancillary_chunks):
            ancillary_chunk_dict[key] = chunk_arr

    # Extract required fields directly from reconstructed dictionary
    sea_level_pressure = ancillary_chunk_dict["pressure_sea_level"]
    wind_speed = ancillary_chunk_dict["wind_speed"]

    # 1. Spatial Patch Masking
    mask, coarse_mask = patch_to_process(height, width, lat, long, stepsize, coarse_step)

    # 2. Barometric Scale & Pressure Calculation
    pressure = barometric_scale(sea_level_pressure, altitude)

    # 3. Geometric Angle Conversions
    sol_zen_rad = np.radians(sol_zen)
    cos_sol_zen = np.cos(sol_zen_rad)
    view_zen_rad = np.radians(view_zen)
    rel_az = np.abs(view_az - sol_az - 180)
    airmass = (1.0 / cos_sol_zen) + (1.0 / np.cos(view_zen_rad))

    # 4. Rayleigh Coarse-Grid Evaluation
    rayleigh_radiance = np.full((height, width), np.nan, dtype=np.float32)
    rayleigh_transmittance = np.full((height, width), np.nan, dtype=np.float32)
    c_slice = np.s_[::coarse_step, ::coarse_step]

    valid_coarse = (coarse_mask[c_slice] == 1) & ~np.isnan(view_zen[c_slice])

    if np.any(valid_coarse):
        ray_I, sol_trans, view_trans = rayleigh_correction(
            wavelength,
            lut_dict,
            solar_irradiance,
            airmass[c_slice][valid_coarse],
            sol_zen[c_slice][valid_coarse],
            view_zen[c_slice][valid_coarse],
            rel_az[c_slice][valid_coarse],
            pressure[c_slice][valid_coarse],
            wind_speed[c_slice][valid_coarse],
        )

        temp_rad = np.full(valid_coarse.shape, np.nan, dtype=np.float32)
        temp_trans = np.full(valid_coarse.shape, np.nan, dtype=np.float32)

        temp_rad[valid_coarse] = ray_I
        temp_trans[valid_coarse] = sol_trans * view_trans

        rayleigh_radiance[c_slice] = temp_rad
        rayleigh_transmittance[c_slice] = temp_trans

    # 5. Gaseous Absorption Transmittance
    if no_gas_absorption:
        diffuse_transmittance = 1.0
    else:
        gas_func = gaseous_absorption(ancillary_chunk_dict, wavelength)
        ozone_trans, no2_trans, _ = gas_func(sol_zen, view_zen, airmass)
        diffuse_transmittance = ozone_trans * no2_trans

    # 6. Spatial 2D Grid Interpolation
    rayleigh_radiance = interpolate(rayleigh_radiance, coarse_step).astype(np.float32)
    rayleigh_transmittance = interpolate(rayleigh_transmittance, coarse_step).astype(np.float32)

    # 7. Physical Radiance & Rayleigh Reflectance Computation
    toa_radiance = (image * solar_irradiance * cos_sol_zen) / np.pi
    L_rc = (toa_radiance / diffuse_transmittance) - rayleigh_radiance
    rho_rc = (np.pi * L_rc) / (solar_irradiance * cos_sol_zen * rayleigh_transmittance)

    return interpolate_and_fill(rho_rc, stepsize)


def calculate_glint_coefficient(view_zen, sol_zen, rel_az, windspeed):
    """
    Calculate the sunglint reflection coefficient over open water surfaces.

    Computes the normalized glint coefficient based on surface slope 
    statistics (Cox & Munk model) and Fresnel reflectance for a given 
    viewing geometry and wind speed.

    Parameters
    ----------
    view_zen : float or numpy.ndarray or xarray.DataArray
        Sensor (viewing) zenith angle in radians.
    sol_zen : float or numpy.ndarray or xarray.DataArray
        Solar zenith angle in radians.
    rel_az : float or numpy.ndarray or xarray.DataArray
        Relative azimuth angle between sensor and sun in radians.
    windspeed : float or numpy.ndarray or xarray.DataArray
        Wind speed in m/s.

    Returns
    -------
    glint_coefficient : float or numpy.ndarray or xarray.DataArray
        Normalized sunglint directional reflection coefficient.

    References
    ----------
    .. [1] Cox, C., & Munk, W. (1954). Measurement of the Roughness of the 
           Sea Surface from Photographs of the Sun's Glitter. Journal of the 
           Optical Society of America, 44(11), 838-850.
    .. [2] NASA Ocean Level-1 and Atmosphere Processing Group (OBPG). 
           https://oceancolor.gsfc.nasa.gov/docs/ocssw/getglint_8f_source.html
    .. [3] HyspIRI Sunglint Report (2011), Eq. 3.1.1-4.
    """

    # Convert to radians if input is in degrees
    view_zen = np.radians(view_zen)
    sol_zen = np.radians(sol_zen)
    rel_az = np.radians(rel_az)

    # Incident/Scattering angle (omega)
    cos_omega_val = np.clip(
        np.cos(view_zen) * np.cos(sol_zen) - np.sin(view_zen) * np.sin(sol_zen) * np.cos(rel_az),
        -1.0, 1.0
    )
    omega = np.arccos(cos_omega_val) / 2.0

    # Surface wave facet tilt angle (beta)
    cos_omega = np.cos(omega)
    denom_beta = np.where(cos_omega == 0, 1e-6, 2.0 * cos_omega)
    cos_beta_val = np.clip((np.cos(view_zen) + np.cos(sol_zen)) / denom_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta_val)

    # Surface slope variance (Cox & Munk slope parameter)
    sigma = np.sqrt(0.003 + 0.00512 * windspeed)

    # Surface slope probability density function
    Ps = 1.0 / (np.pi * sigma**2) * np.exp(-1.0 * np.tan(beta)**2 / sigma**2)

    # Refraction index ratio for water/air
    refraction = 1.34 #4.0 / 3.0

    # Refracted angle inside water
    refracted_angle = np.arcsin(np.clip(np.sin(omega) / refraction, -1.0, 1.0))

    # Fresnel reflectance calculation with zero-division handling for normal incidence
    omega_minus = omega - refracted_angle
    omega_plus = omega + refracted_angle

    sin_plus = np.sin(omega_plus)
    tan_plus = np.tan(omega_plus)

    fresnel_s = np.where(np.abs(sin_plus) < 1e-6, (refraction - 1.0) / (refraction + 1.0), np.sin(omega_minus) / sin_plus)
    fresnel_p = np.where(np.abs(tan_plus) < 1e-6, (refraction - 1.0) / (refraction + 1.0), np.tan(omega_minus) / tan_plus)

    fresnel_reflectance = (fresnel_s**2 + fresnel_p**2) / 2.0

    # Glint coefficient calculation
    cos_v = np.cos(view_zen)
    cos_s = np.cos(sol_zen)
    cos_b = np.cos(beta)

    denom_glint = 4.0 * cos_v * (cos_b**4)
    glint_coefficient = np.where(
        denom_glint > 1e-6,
        (fresnel_reflectance * Ps) / denom_glint, #(fresnel_reflectance * Ps * np.pi) / denom_glint,
        0.0
    )

    return glint_coefficient


@Error_Handler
def get_mask(glint_coefficient, quality_image, cloudless_mask, nir_image, NDWI, shape, lastChunk):
    """global total_cloud
    global total_quality
    global total_ndwi"""

    # 1. Quality Mask (Bitwise operation replacing bit 7 lookup)
    if quality_image is not None:
        # 1. Cast to uint16/uint8 so bitwise & is valid
        # 2. Extract bit 7 (1 << 7 = 128)
        # 3. Cast the non-zero result to boolean
        notQualityMasked = (quality_image.astype(np.uint16) & (1 << 7)).astype(bool)
    else:
        notQualityMasked = xr.DataArray(True, coords=nir_image.coords, dims=nir_image.dims)

    # 2. Non-empty and Non-land Masks
    notEmpty = nir_image > 0
    notLand = NDWI > 0

    # 3. Cloud Masking
    if cloudless_mask is not None:
        # Interpolate cloudless_mask to match nir_image dimensions dynamically
        if isinstance(cloudless_mask, (xr.DataArray, xr.Dataset)):
            notCloud = cloudless_mask.interp_like(nir_image, method="linear").astype(bool)
        else:
            # Fallback if cloudless_mask is a raw numpy array
            zoom_factors = (shape[0] / cloudless_mask.shape[0], shape[1] / cloudless_mask.shape[1])
            notCloud = xr.DataArray(
                zoom(cloudless_mask, zoom_factors, order=1).astype(bool),
                coords=nir_image.coords,
                dims=nir_image.dims
            )
    else:
        notCloud = nir_image < 0.024

    # 4. Combine all conditions into final mask
    keepPixel = notCloud & notEmpty & notLand & notQualityMasked

    # 5. Compute metrics/sums
    sum_cloud = int(notCloud.sum().compute().item())
    sum_quality = int(notQualityMasked.sum().compute().item())
    sum_ndwi = int(notLand.sum().compute().item())
    sum_keep = int(keepPixel.sum().compute().item())

    # Print Diagnostics
    """print(f"Shape: {shape}")
    print(f"Quality mask sum: {sum_quality}")
    print(f"NDWI mask sum: {sum_ndwi}")
    print(f"Cloud mask sum: {sum_cloud}")
    

    if lastChunk:
        print(f"TOTAL CLOUD: {sum_cloud}")
        print(f"TOTAL QUALITY: {sum_quality}")
        print(f"TOTAL NDWI: {sum_ndwi}")"""

    print(f"{(sum_keep / keepPixel.size) * 100:.2f}% of pixels remaining after masking")

    return keepPixel


def rayleigh_main(ds: xr.Dataset) -> xr.Dataset:
    """
    Perform Rayleigh scattering and gaseous absorption corrections on satellite imagery.

    Extracts geometry and ancillary data from an xarray Dataset, resamples spatial
    ancillary rasters to match the dataset grid, and applies Rayleigh radiometrical
    corrections chunk-by-chunk using Dask parallelization.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing:
        - Spectral reflectance/radiance bands.
        - Geometry layers ('solar_zenith', 'solar_azimuth', 'viewing_zenith', 
          'viewing_azimuth').
        - Spatial dimensions ('x', 'y' or 'lat', 'lon').
        - Global attributes (`sensor`, `earth_sun_distance`, `ancillary_data`, 
          `rayleigh_lut`, `fixed_resolution`, etc.).

    Returns
    -------
    xarray.Dataset
        Output dataset containing:
        - `latitude` : xarray.DataArray (float32)
            Target coordinate latitude grid.
        - `longitude` : xarray.DataArray (float32)
            Target coordinate longitude grid.
        - `solar_zenith` : xarray.DataArray (float32)
            Solar zenith angles across the spatial domain.
        - `solar_azimuth` : xarray.DataArray (float32)
            Solar azimuth angles across the spatial domain.
        - `viewing_zenith` : xarray.DataArray (float32)
            Sensor viewing zenith angles.
        - `viewing_azimuth` : xarray.DataArray (float32)
            Sensor viewing azimuth angles.
        - `rho_rc_<wavelength>` : xarray.DataArray (float32)
            Rayleigh-corrected surface reflectance arrays for each processed band.
        - `attrs["ancillary_data"]` : dict
            Retained dictionary of processed ancillary data.

    Raises
    ------
    KeyError
        If required attributes (`earth_sun_distance`, `ancillary_data`, `rayleigh_lut`)
        or geometry variables are missing from `ds`.

    Notes
    -----
    - All floating-point matrices (coordinates, ancillary rasters, geometry variables)
      are strictly cast to `numpy.float32` prior to block execution to conserve memory
      and maintain consistent precision in lower-level C/Fortran routines.
    - Static invariant arrays (coordinates, solar geometry, ancillary rasters) are 
      chunk-aligned and cast once outside the wavelength loop to prevent Dask task-graph
      re-serialization overhead.
    """
    # --- 1. Unpack Metadata from Dataset Attributes ---
    attrs = dict(ds.attrs)
    sensor = str(attrs.get("sensor", "")).lower()
    earth_sun_distance = attrs["earth_sun_distance"]
    altitude = attrs.get("altitude", 0)
    stepsize = attrs.get("stepsize", 1)
    fixed_resolution = attrs.get("fixed_resolution", 10)
    no_gas_absorption = attrs.get("no_gas_absorption", False)
    ancillary_data = dict(attrs["ancillary_data"])
    lut_list = attrs["rayleigh_lut"]

    # Infer spatial dimensions
    y_dim = "y" if "y" in ds.dims else ("latitude" if "latitude" in ds.dims else "lat")
    x_dim = "x" if "x" in ds.dims else ("longitude" if "longitude" in ds.dims else "lon")

    height = ds.sizes[y_dim]
    width = ds.sizes[x_dim]
    """full_height = attrs.get("full_height", height)
    full_width = attrs.get("full_width", width)"""
    ystart = attrs.get("ystart", 0)
    ystep = attrs.get("ystep", height)

    coarse_size = 100
    coarse_step = max(1, int(coarse_size / fixed_resolution))

    if hasattr(ds, "rio") and ds.rio.transform() is not None:
        transform = ds.rio.transform()
        crs = ds.rio.crs
    else:
        transform = attrs.get("transform", Affine.identity())
        crs = attrs.get("crs", "EPSG:4326")

    # Determine uniform chunking scheme
    target_chunks = ds.chunksizes if ds.chunks else {y_dim: height, x_dim: width}
    y_chunks = target_chunks.get(y_dim, height)
    x_chunks = target_chunks.get(x_dim, width)
    chunk_spec = {y_dim: y_chunks, x_dim: x_chunks}

    # Coordinate Generation (Explicit float32 conversion)
    cols, rows = np.meshgrid(np.arange(0, width, stepsize, dtype=np.float32), np.arange(0, height, stepsize, dtype=np.float32))

    # Compute transform directly 
    long_arr, lat_arr = rasterio.transform.xy(transform, rows, cols, offset="center")

    # Free up meshgrid memory early
    del cols, rows

    # Transform directly into destination arrays in EPSG:4326
    proj_to_wgs84 = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    long_arr, lat_arr = proj_to_wgs84.transform(long_arr, lat_arr)

    # Cast to float32 and reshape in-place
    long_arr = np.asarray(long_arr, dtype=np.float32).reshape((height // stepsize, width // stepsize))
    lat_arr = np.asarray(lat_arr, dtype=np.float32).reshape((height // stepsize, width // stepsize))

    lat_da = (xr.DataArray(np.asarray(lat_arr, dtype=np.float32), dims=(y_dim, x_dim)).chunk(chunk_spec))
    long_da = (xr.DataArray(np.asarray(long_arr, dtype=np.float32), dims=(y_dim, x_dim)).chunk(chunk_spec))

    del long_arr, lat_arr  # Free memory

    # --- Resample Ancillary Data ---
    use_gmao = ancillary_data.get("use_gmao", False)
    if use_gmao:
        ancillary_data = resample_gmao_ancillary(ancillary_data, transform, width, height, ystart, ystep, crs=crs)
    else:
        ancillary_data = resample_old_ancillary(ancillary_data, transform, width, height, ystart, ystep, crs=crs   )
        ancillary_data["pressure_sea_level"] = ( ancillary_data["pressure_sea_level"] * 100 ).astype(np.float32)

    # Resample No2 fraction
    ancillary_data["NO2_fraction_200m"] = resample_NO2_fraction(ancillary_data["NO2_fraction_200m"], height, width, ystart, ystep).astype(np.float32)

    # --- 3. Prepare Invariant Spatial Rasters (Outside Loop) ---
    ancillary_keys = []
    ancillary_rasters_da = []
    ancillary_scalars = {}

    for key, val in ancillary_data.items():
        if isinstance(val, np.ndarray) and val.shape == (height, width):
            arr_contiguous = np.ascontiguousarray(val, dtype=np.float32)
            dask_arr = da.from_array(arr_contiguous, chunks=(y_chunks, x_chunks))
            ancillary_keys.append(key)
            ancillary_rasters_da.append(xr.DataArray(dask_arr, dims=(y_dim, x_dim)))
        else:
            ancillary_scalars[key] = val

    # Static solar geometry (prepared once)
    sol_zen_da = ds["solar_zenith"].chunk(chunk_spec).astype(np.float32)
    sol_az_da = ds["solar_azimuth"].chunk(chunk_spec).astype(np.float32)

    # --- 4. Initialize Output Dataset ---
    attrs["ancillary_data"] = ancillary_data

    ds_out = xr.Dataset(coords=ds.coords, attrs=attrs)
    ds_out["latitude"] = lat_da
    ds_out["longitude"] = long_da
    ds_out["solar_zenith"] = sol_zen_da
    ds_out["solar_azimuth"] = sol_az_da
    ds_out["water_vapor"] = (ancillary_rasters_da[ancillary_keys.index("water_vapor")]).chunk(chunk_spec).astype(np.float32)

    # Non-MSI viewing geometry is band-invariant (assign once)
    if sensor != "msi" or "wavelength" not in ds["viewing_zenith"].dims:
        ds_out["viewing_zenith"] = ds["viewing_zenith"].chunk(chunk_spec).astype(np.float32)
        ds_out["viewing_azimuth"] = ds["viewing_azimuth"].chunk(chunk_spec).astype(np.float32)
        
    # --- 5. Loop Wavelengths & Apply Correction ---
    wavelengths = (
        ds["wavelength"].values if "wavelength" in ds.coords else attrs.get("wavelengths", [])
    )

    for wl in wavelengths:
        str_wl = str(wl)
        lut_wave = get_safe_wavelength(wl, lut_list.keys(), threshold=20)
        sensor_wave = get_safe_wavelength(wl, ancillary_scalars["sensor_data"].keys(), threshold=20 )
        solar_irradiance = np.float32(ancillary_scalars["sensor_data"][sensor_wave]["solar_irradiance"]  / (earth_sun_distance**2) )

        if "rhot" in ds and "wavelength" in ds["rhot"].dims:
            img_da = ds["rhot"].sel(wavelength=wl)
        elif f"band_{str_wl}" in ds:
            img_da = ds[f"band_{str_wl}"]
        else:
            img_da = ds["images"][str_wl]

        img_da = img_da.chunk(chunk_spec).astype(np.float32)

        if sensor == "msi" and "wavelength" in ds["viewing_zenith"].dims:
            view_zen_da = ds["viewing_zenith"].sel(wavelength=wl).chunk(chunk_spec).astype(np.float32)
            view_az_da = ds["viewing_azimuth"].sel(wavelength=wl).chunk(chunk_spec).astype(np.float32)
            
            # Store per-band viewing arrays in ds_out if MSI wavelength-dependent
            ds_out[f"viewing_zenith_{str_wl}"] = view_zen_da
            ds_out[f"viewing_azimuth_{str_wl}"] = view_az_da
        else:
            view_zen_da = ds_out["viewing_zenith"]
            view_az_da = ds_out["viewing_azimuth"]

        args = [
            img_da, sol_zen_da, sol_az_da, view_zen_da, view_az_da,
            lat_da, long_da, *ancillary_rasters_da
        ]

        rho_rc_da = xr.apply_ufunc(
            _process_chunk_block,
            *args,
            kwargs={
                "wavelength": wl,
                "lut_dict": lut_list[lut_wave],
                "solar_irradiance": solar_irradiance,
                "altitude": altitude,
                "no_gas_absorption": no_gas_absorption,
                "coarse_step": coarse_step,
                "stepsize": stepsize,
                "ancillary_keys": ancillary_keys,
                "ancillary_scalars": ancillary_scalars,
            },
            input_core_dims=[[]] * len(args),
            output_core_dims=[[]],
            dask="parallelized",
            output_dtypes=[np.float32],
        )

        ds_out[f"rho_rc_{str_wl}"] = rho_rc_da

    # Generate the mask for the image
    # 1. Compute NDWI
    ndwi = (ds["rhot"].sel(wavelength=wavelengths[2]) - ds["rhot"].sel(wavelength=wavelengths[4])) / (ds["rhot"].sel(wavelength=wavelengths[2]) + ds["rhot"].sel(wavelength=wavelengths[4]))
    ndwi = ndwi.rename("NDWI")
    # Assuming view_zen, sol_zen, rel_az, and windspeed are 3D xr.DataArrays
    glint_da = calculate_glint_coefficient(
        view_zen=  ds['viewing_zenith'], # Radians
        sol_zen=   ds['solar_zenith'],   # Radians
        rel_az=    np.abs(ds['viewing_zenith'] - ds['solar_zenith'] - 180),# Radians
        windspeed= (ancillary_rasters_da[ancillary_keys.index("wind_speed")]).chunk(chunk_spec).astype(np.float32)    # m/s
    )
    # Assign metadata name
    glint_da = xr.DataArray(glint_da, coords=ds['viewing_zenith'].coords, dims=ds['viewing_zenith'].dims).chunk(chunk_spec)
    glint_da = glint_da.rename("glint_coefficient")


    # Check existence, cast to float32, and re-chunk if present
    cloud_da = ds['cloudless'].astype(np.float32).chunk(chunk_spec) if 'cloudless' in ds else None
    qa_da = ds['quality_flag'].astype(np.float32).chunk(chunk_spec) if 'quality_flag' in ds else None

    # Get the valid -mask
    mask_da = get_mask( glint_coefficient=glint_da,
                        quality_image=qa_da,                                                        # Pass None if you don't have a QA band
                        cloudless_mask=cloud_da,                                                    # Pass None to use the default NIR threshold (< 0.024)
                        nir_image=ds["rhot"].sel(wavelength=wavelengths[-1]),                       # 2D Xarray DataArray for NIR band
                        NDWI=ndwi,                                                                  # 2D Xarray DataArray for NDWI
                        shape=(height, width),                                                      # Tuple e.g., (height, width)
                        lastChunk=True,                                                             # Set True if processing the final chunk
                        )

    ds_out["water_mask"] = mask_da
    ds_out["glint_coefficient"] = glint_da
    

    if hasattr(ds, "rio") and ds.rio.crs is not None:
        ds_out.rio.write_crs(ds.rio.crs, inplace=True)
        ds_out.rio.write_transform(transform, inplace=True)

    return ds_out