# -*- coding: utf-8 -*-

"""
File Name:              load_rayleigh_data.py
Description:            This code-file contains the list functions to the satellite TOA data and the appropriate ancillary data. 
                        The functions are used to load the data into memory and prepare it for processing.

Date Created:           August 25th, 2026
Author:                 William Wainwright, 
Email:                  william.wainwright@ssaihq.com/william.wainwright@nasa.gov
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import xmltodict
import numpy as np
import pyproj
import rasterio
import xarray as xr
import netCDF4 as nc
import srtm
from datetime import datetime as dt
from dataclasses import dataclass
from scipy.ndimage import zoom
from typing import Callable, Iterable, Union, Optional, Any


from .aq_error import Error_Handler



OLI_wavelengths = {'1' : '443',
                   '2' : '482',
                   '3' : '561',
                   '4' : '655',
                   '5' : '865',
                   '6' : '1609',
                   '7' : '2201',
                   '8' : '0',
                   '9' : '0',
                   '10': '0',
                   '11': '0'
    }

Full_OLI_wavelengths = {'1' : '443',
                        '2' : '482',
                        '3' : '561',
                        '4' : '655',
                        '5' : '865',
                        '6' : '1609',
                        '7' : '2201',
                        '8' : 'Panchromatic',
                        '9' : 'Cirrus',
                        '10': 'TIRS 1',
                        '11': 'TIRS 2'
    }

MSI_wavelengths = {'01' : '443',
                   '02' : '490',
                   '03' : '560',
                   '04' : '665',
                   '05' : '705',
                   '06' : '740',
                   '07' : '780',
                   '08' : '833',
                   '8A' : '864',
                   '09' : '944',
                   '10' : '1375',
                   '11' : '1609',
                   '12' : '2200'
    }


@dataclass
class transformObj:
    """Stores affine transformation metadata and pyproj coordinate transformer."""
    xoffset: float
    px_w: float
    rot1: float
    yoffset: float
    rot2: float
    px_h: float
    crs: Any  # WKT string, EPSG code, or pyproj.CRS object
    pyproj_transformer: pyproj.Transformer = None

    def __post_init__(self):
        if self.pyproj_transformer is None and self.crs is not None:
            # Pre-build transformer to reproject map units to WGS84 (lon, lat)
            self.pyproj_transformer = pyproj.Transformer.from_crs(
                self.crs, "EPSG:4326", always_xy=True
            )


@Error_Handler
def sorted_dict(data_dict):
    return {key:data_dict[key] for key in sorted(data_dict.keys(),key=int)}


@Error_Handler
def reshape_1D(images):
    
    return sorted(list(zip(images.keys(),*np.reshape(images.values(),-1) )),key= lambda x: float(x[0]))


@Error_Handler
def distance(coord,lat,long):
    return np.sqrt( (coord[0] - lat)**2 + (coord[1] - long)**2 )


def read_hdf4_variable(filepath, variable_name):
    try:
        dataset = nc.Dataset(filepath, mode='r')
    except Exception as e:
        raise FileNotFoundError(f"netCDF4 could not open {filepath}. Error: {e}")

    try:
        if variable_name not in dataset.variables:
            available_vars = list(dataset.variables.keys())
            raise ValueError(f"Variable '{variable_name}' not found. Available: {available_vars}")

        # Extract data array using slice notation
        array = dataset.variables[variable_name][:]
        return array
    finally:
        dataset.close()


@Error_Handler
def load_landsat_meta(path: str | Path) -> dict:
    """Parses and loads a Landsat XML metadata file (MTL.xml) into a dictionary.

    Searches the specified scene directory for an MTL XML file, parses its 
    hierarchical tag structure into a standard Python dictionary, and strips 
    the top-level root key for direct access to metadata blocks.

    Args:
        path (str | Path): Path to the directory containing Landsat scene files.

    Returns:
        dict: Parsed metadata hierarchy containing scene attribute dictionaries 
            (e.g., 'IMAGE_ATTRIBUTES', 'PROJECTION_ATTRIBUTES').

    Raises:
        FileNotFoundError: If no file containing 'mtl.xml' is found in `path`.
    """
    path_obj = Path(path)

    # -------------------------------------------------------------------------
    # 1. Locate the MTL XML Metadata File
    # -------------------------------------------------------------------------
    mtl_file = None
    for file_path in path_obj.iterdir():
        if 'mtl.xml' in file_path.name.lower():
            mtl_file = file_path
            break

    if mtl_file is None:
        raise FileNotFoundError(f"No MTL XML metadata file found in directory: {path}")

    print(f"Loading metadata from {mtl_file.name}")

    # -------------------------------------------------------------------------
    # 2. Read and Convert XML Structure to Dictionary
    # -------------------------------------------------------------------------
    root = ET.parse(mtl_file).getroot()
    xml_str = ET.tostring(root, encoding='utf-8', method='xml')
    meta = dict(xmltodict.parse(xml_str))

    # -------------------------------------------------------------------------
    # 3. Prune Root Container Key for Direct Hierarchy Access
    # -------------------------------------------------------------------------
    # Unnest 'LANDSAT_METADATA_FILE' wrapper level if present
    try:
        meta = meta['LANDSAT_METADATA_FILE']
    except KeyError:
        print("Unable to prune upper tree level ('LANDSAT_METADATA_FILE' key not found)")

    return meta


@Error_Handler
def get_earth_sun_distance(date: str) -> float:
    """Retrieves the Earth-Sun distance in Astronomical Units (AU) for a given date.

    Loads a pre-calculated day-of-year Earth-Sun distance lookup table from a CSV
    file and extracts the distance value corresponding to the input date's day of year.

    Args:
        date (str): Date string formatted as 'YYYY-MM-DD' (e.g., '2023-06-21').
        
    Returns:
        float: Earth-Sun distance in Astronomical Units (AU) for the target day.

    Raises:
        ValueError: If `date` does not match the 'YYYY-MM-DD' format.
        FileNotFoundError: If the ancillary lookup CSV file is not found.
    """
    # -------------------------------------------------------------------------
    # 1. Ancillary File Path Setup
    # -------------------------------------------------------------------------
    es_file_path = Path(__file__).resolve().parent / r"ancillary/ESdistance/earth_sun_distance.csv"

    if not es_file_path.is_file():
        raise FileNotFoundError(f"Earth-Sun distance lookup file not found at: {es_file_path}")

    # -------------------------------------------------------------------------
    # 2. Read Day-of-Year Lookup Table
    # -------------------------------------------------------------------------
    # Load distance values (column index 1) ignoring header row
    es_data = np.genfromtxt(es_file_path, delimiter=',', skip_header=1, usecols=1)

    # -------------------------------------------------------------------------
    # 3. Calculate Day of Year & Return Distance
    # -------------------------------------------------------------------------
    # Parse date string and extract ordinal day (1–366)
    day_of_year = dt.strptime(date, "%Y-%m-%d").timetuple().tm_yday

    # Convert 1-based day-of-year index to 0-based array index
    return float(es_data[day_of_year - 1])


@Error_Handler
def select_closest_file(
    files: Iterable[str],
    file_type_check: Callable[[str], bool],
    target_hour: Union[int, str],
    max_delta: int = 6,
) -> Optional[str]:
    """Select the file path closest in time to a target hour, resolving ties by product priority.

    Calculates the absolute hour difference between each candidate file's timestamp
    and `target_hour`. Filters out aerosol files and candidates exceeding `max_delta`.
    If multiple candidate files share the same minimum time delta, ties are broken
    by preferring MERRA-2 over NRT products (IT/FP).

    Parameters
    ----------
    files : iterable of str
        List or iterable of file path strings to search.
    file_type_check : callable
        Function or lambda taking a lowercase file path string and returning
        True if the file matches the desired ancillary type criteria.
    target_hour : int or str
        The reference hour (0-23) used to compute the time delta.
    max_delta : int, default=6
        Maximum allowable hour difference for a file to be considered valid.

    Returns
    -------
    str or None
        The file path string of the selected candidate with the minimal time delta
        and highest product priority, or None if no valid candidate is found.
    """
    gmao_priority = {'gmao_merra2': 0, 'gmao_it': 1, 'gmao_fp': 2}
    candidates = []

    for path in files:
        path_lower = path.lower()

        # Skip aerosols
        if '.aer.' in path_lower:
            continue

        if file_type_check(path_lower):
            try:
                # Extract file hour
                filename = path.split('/')[-1]
                file_hour = int(filename.split('.')[1].split('T')[1][:2])
                delta = abs(file_hour - int(target_hour))

                if delta <= max_delta:
                    # Determine prefix priority ranking (default to low priority if not matched)
                    prio = 99
                    for prefix, rank in gmao_priority.items():
                        if prefix in path_lower:
                            prio = rank
                            break

                    candidates.append((delta, prio, path))
            except (IndexError, ValueError):
                continue

    if not candidates:
        return None

    # Sort primarily by smallest time delta, secondarily by prefix priority rank
    candidates.sort(key=lambda x: (x[0], x[1]))

    # Return the file path with the minimum time delta
    return candidates[0][2]


@Error_Handler
def search_local_ancillary(
    anc_path: str | Path, 
    date: str, 
    hour: int,
    max_delta: int = 6
) -> tuple[list[str], bool]:
    """Searches a local directory for atmospheric ancillary data files matching a date and hour.

    Evaluates local files against accepted dataset types (Ozone, NCEP MET, GMAO FP,
    GEOS-CF) and filters for files whose temporal timestamp falls within the sampling
    interval window for that specific product type.

    Args:
        anc_path (str | Path): Path to the directory containing local ancillary files.
        date (str): Date string formatted as 'YYYY-MM-DD'.
        hour (int): Observation hour as an integer (0–23).
        max_delta (int, optional): Maximum allowed time difference (in hours) for matching files.

    Returns:
        tuple[list[str], bool]: A tuple containing:
            - list[str]: Full file paths of matching ancillary data files.
            - bool: Flag indicating whether GMAO products are used for the given date.
    """
    
    anc_path = Path(anc_path)

    if not anc_path.exists():
        return [], False

    # Gather all files directly inside anc_path
    local_files = [str(f) for f in anc_path.iterdir() if f.is_file()]

    # Select GEOS-CF (Ozone) closest to target hour
    geos_cf_file = select_closest_file(local_files, lambda f: 'gmao_geos-cf.' in f, hour, max_delta=max_delta)

    # Select Surface MET closest to target hour
    met_file = select_closest_file(local_files, lambda f: '.met.' in f and 'gmao_geos-cf.' not in f, hour, max_delta=max_delta)

    # Select PROFILE closest to target hour
    profile_file = select_closest_file(local_files, lambda f: '.profile.' in f and 'gmao_geos-cf.' not in f, hour, max_delta=max_delta)

    # Ensure all 3 components are resolved
    if geos_cf_file and met_file and profile_file:
        selected_files = [geos_cf_file, met_file, profile_file]
        print("Complete 3-file ancillary set identified by closest time delta:")
        for f in selected_files:
            print(f" - {(Path(f)).name}")
        return selected_files, True

    missing = []
    if not geos_cf_file: 
        missing.append("GEOS-CF")
    if not met_file: 
        missing.append("MET")
    if not profile_file: 
        missing.append("PROFILE")

    print(f"Incomplete set. Missing closest match for: {', '.join(missing)}")
    return [], False


@Error_Handler
def read_ancillary(
    date: str,
    hour: int,
    sensor: str,
    files: list[str | bytes | Path],
    use_gmao: bool,
) -> dict[str, Any]:
    """Reads atmospheric ancillary data (meteorology, gases, elevation, and sensor parameters).

    Parses meteorological and gas concentration inputs using either static climatology
    (NCEP MET + OMI O3 + HDF4 NO2) or dynamic GMAO NetCDF models (GMAO MET, profile, 
    and GEOS-CF). Appends shared elevation routines and sensor bandpass characteristics.

    Args:
        date (str): Observation date formatted as 'YYYY-MM-DD'.
        hour (int): Observation hour integer (0-23).
        sensor (str): Satellite sensor name (e.g., 'S2A', 'S2B', 'S2C', 'OLI').
        files (list[str | bytes | Path]): Paths or filenames of local ancillary files.
        use_gmao (bool): Flag indicating whether to use GMAO NetCDF files (True)
            or static/NCEP HDF4 files (False).

    Returns:
        dict[str, Any]: Dictionary containing loaded ancillary variables:
            - Spatial grids: Ozone, NO2 (total, tropo, strato), winds (m, z, speed),
              sea level pressure, precipitable water, water vapor column, relative humidity.
            - Common grids: NO2 fraction above 200m, elevation lambda function.
            - Metadata: Sensor bandpass attributes dictionary mapped by wavelength string.

    Raises:
        IndexError: If required ancillary files (OMI, NCEP, or GMAO) are missing from `files`.
        FileNotFoundError: If NO2 climatology or sensor CSV files do not exist.
    """
    # -------------------------------------------------------------------------
    # 1. Directory Path Definitions & Byte-String Sanitization
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent / r"ancillary"
    no2_dir = base_dir / "NO2"
    sensor_dir = base_dir / "sensor_tables"
    srtm_dir = base_dir / "srtm"
    

    ancillary_data: dict[str, Any] = {"use_gmao": use_gmao}
    file_list: list[str] = []

    # Clean byte strings or convert Path instances to string representations
    for f in files:
        #Strip out the b'' if it occurs
        if str(f).startswith("b\'"):
            file_list.append(str(f)[2:-1])
        else:
            file_list.append(str(f))
        
        
    #print(file_list)

    # -------------------------------------------------------------------------
    # 2. Extract Ancillary Data: Static / NCEP Path (use_gmao == False)
    # -------------------------------------------------------------------------
    if not use_gmao:
        ozone_file = [f for f in file_list if 'omi' in f.lower()][0]
        wind_file = [f for f in file_list if 'ncep' in f.lower()][0]
        nitrogen_file = no2_dir / "no2_climatology_v2013.hdf"
        month = date.split('-')[1]

        # Ozone (DU)
        ancillary_data['ozone'] = read_hdf4_variable(str(ozone_file), 'ozone')

        # NO2 concentrations
        ancillary_data['NO2_total'] = read_hdf4_variable(str(nitrogen_file), f"tot_no2_{month}")
        ancillary_data['NO2_troposphere'] = read_hdf4_variable(str(nitrogen_file), f"trop_no2_{month}")
        ancillary_data['NO2_stratosphere'] = read_hdf4_variable(str(nitrogen_file), f"strat_no2_{month}")

        # Wind speed vectors at 10m (m/s)
        ancillary_data['wind_m'] = read_hdf4_variable(str(wind_file), 'm_wind')
        ancillary_data['wind_z'] = read_hdf4_variable(str(wind_file), 'z_wind')
        ancillary_data['wind_speed'] = np.sqrt(
            ancillary_data['wind_m'] ** 2 + ancillary_data['wind_z'] ** 2
        )

        # Pressure, humidity, and water vapor
        ancillary_data['pressure_sea_level'] = read_hdf4_variable(str(wind_file), 'press')
        ancillary_data['relative_humidity'] = read_hdf4_variable(str(wind_file), 'rel_hum')
        ancillary_data['precipitable_water'] = read_hdf4_variable(str(wind_file), 'p_water')
        ancillary_data['water_vapor'] = ancillary_data['precipitable_water'] / 10.0

    # -------------------------------------------------------------------------
    # 3. Extract Ancillary Data: Dynamic GMAO Path (use_gmao == True)
    # -------------------------------------------------------------------------
    else:
        gmao_met_file = [f for f in file_list if '.met.' in f.lower() and not 'geos' in f.lower()][0]
        gmao_profile_file = [f for f in file_list if '.profile.' in f.lower() and not 'geos' in f.lower()][0]

        # Attempt reading GMAO GEOS-CF NO2 fields; fall back to static climatology if unreadable
        try:
            gmao_geos_file = [f for f in file_list if 'geos' in f.lower()][0]
            temp_geos = nc.Dataset(gmao_geos_file, 'r')
            gmao_geos = temp_geos.variables

            # Lat/Long from gmao geos files
            # Steps of 0.25 x 0.25 degrees
            ancillary_data['geos_lat'] = gmao_geos['lat'][:]
            ancillary_data['geos_long'] = gmao_geos['lon'][:]

            # NO2 in 1e15 molecules/cm^2 (721,1440)
            # Steps of 0.25 x 0.25 degrees
            ancillary_data['NO2_total'] = gmao_geos['TOTCOL_NO2'][:]
            ancillary_data['NO2_troposphere'] = gmao_geos['TROPCOL_NO2'][:]
            ancillary_data['NO2_stratosphere'] = gmao_geos['STRATCOL_NO2'][:]

            temp_geos.close()
        except Exception:
            print("Error loading gmao NO2 climatology, defaulting to static")

            # NO2
            nitrogen_file = f"{NO2dir}no2_climatology_v2013.hdf"
            month = date.split('-')[1]

            ancillary_data['NO2_total'] = read_hdf4_variable(nitrogen_file, f"tot_no2_{month}")
            ancillary_data['NO2_troposphere'] = read_hdf4_variable(nitrogen_file, f"trop_no2_{month}")
            ancillary_data['NO2_stratosphere'] = read_hdf4_variable(nitrogen_file, f"strat_no2_{month}")

        # Extract GMAO meteorology and vertical profile attributes
        # Extract variables and close files
        temp_met = nc.Dataset(gmao_met_file, 'r')
        gmao_met = temp_met.variables

        temp_prof = nc.Dataset(gmao_profile_file, 'r')
        gmao_profile = temp_prof.variables

        # Lat/Long from gmao met and profile files
        # Steps of 0.5 degrees lat by 0.625 degrees long
        ancillary_data['gmao_lat'] = gmao_met['lat'][:]
        ancillary_data['gmao_long'] = gmao_met['lon'][:]

        # Ozone in DU (361,576)
        # Steps of 0.5 degrees lat by 0.625 degrees long
        ancillary_data['ozone'] = gmao_met['TO3'][:]

        # Wind speed at 10 meters
        # V10M is the meridonal wind (N-S)
        # U10M is the zonal wind (E-W)
        ancillary_data['wind_m'] = gmao_met['V10M'][:]
        ancillary_data['wind_z'] = gmao_met['U10M'][:]
        ancillary_data['wind_speed'] = np.sqrt(ancillary_data['wind_m'] ** 2 + ancillary_data['wind_z'] ** 2)

        # Sea-level pressure (Pa)
        ancillary_data['pressure_sea_level'] = gmao_met['SLP'][:]

        # Total precipitable water vapor (kg/m^2)
        ancillary_data['precipitable_water'] = gmao_met['TQV'][:]
        # Water vapor column height (cm) - mass (kg/m^2) / density (1000 kg/m^3) * 100 cm/m
        ancillary_data['water_vapor'] = ancillary_data['precipitable_water'] / 10

        # Relative humidity (fraction)
        ancillary_data['relative_humidity'] = gmao_profile['RH'][0, :, :]

        temp_met.close()
        temp_prof.close()

    # -------------------------------------------------------------------------
    # 4. Common Ancillary Data (NO2 Vertical Fraction & Elevation)
    # -------------------------------------------------------------------------
    # Tropospheric NO2 fraction above 200m elevation (2-degree grid)
    # Fraction of tropospheric NO2 above 200m, 2-degree grid
    nitrogen_fraction_file = no2_dir / f"trop_f_no2_200m.hdf"
    nitrogen_frac = read_hdf4_variable(nitrogen_fraction_file, "f_no2_200m")
    ancillary_data['NO2_fraction_200m'] = np.flipud(nitrogen_frac)

    # SRTM Elevation data
    if not srtm_dir.exists():
        srtm_dir.mkdir(parents=True, exist_ok=True)
    srtm_data = srtm.SrtmService(srtm_dir.as_posix())
    ancillary_data['altitude_tf'] = lambda lat, long: srtm_data.get_elevation(lat, long)

    # Choosing sensor file
    if '2a' in sensor.lower():
        sensor_file = "msi-s2a_bandpass.csv"
    elif '2b' in sensor.lower():
        sensor_file = "msi-s2b_bandpass.csv"
    elif '2c' in sensor.lower():
        sensor_file = "msi-s2b_bandpass.csv"
    else:
        sensor_file = "oli_landsat8.csv"

    # Sensor data table
    sensor_data = np.genfromtxt((sensor_dir / sensor_file), delimiter=',', skip_header=2)
    ancillary_data['sensor_data'] = dict()
    for row in sensor_data:
        wavelength = str(int(row[1]))
        ancillary_data['sensor_data'][wavelength] = {'FWHM': row[3],
                                                     'solar_irradiance': row[4],
                                                     'rayleigh_optical_thickness': row[5],
                                                     'depolarization_factor': row[6],
                                                     'ozone_opacity': row[7],
                                                     'NO2_opacity': row[8]
                                                     }

    return ancillary_data


@Error_Handler
def get_center(
    width: int, 
    height: int, 
    transforms: dict[Any, Any]
) -> tuple[float, float, float] | tuple[int, int, int]:
    """Calculates the geographic/projected center coordinates of an image raster using pyproj.

    Computes pixel coordinates for the raster midpoint, applies affine spatial 
    transformation to calculate projected map coordinates, and reprojects them to 
    geographic coordinates (WGS84 / EPSG:4326) using pyproj.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        transforms (dict[Any, Any]): Dictionary mapping keys (e.g., wavelengths) 
            to spatial transform objects containing affine properties (`px_w`, 
            `px_h`, `rot1`, `rot2`, `xoffset`, `yoffset`) and a pyproj-compatible 
            CRS (`crs` or `pyproj_transformer`).

    Returns:
        tuple[float, float, float] | tuple[int, int, int]: Geographic coordinates of 
            the image center (longitude, latitude, altitude), or `(0, 0, 0)` if 
            spatial transformation fails.
    """
    # -------------------------------------------------------------------------
    # 1. Compute Raster Center Pixel Coordinates
    # -------------------------------------------------------------------------
    center_x = int(width / 2)
    center_y = int(height / 2)

    # Extract spatial metadata from first transform object
    ct = list(transforms.values())[0]

    # -------------------------------------------------------------------------
    # 2. Apply Affine Transformation to Pixel Coordinates
    # -------------------------------------------------------------------------
    # Project pixel indices to ground map coordinates
    pos_x = ct.px_w * center_x + ct.rot1 * center_y + ct.xoffset
    pos_y = ct.rot2 * center_x + ct.px_h * center_y + ct.yoffset

    # Shift to center of target pixel cell
    pos_x += ct.px_w / 2.0
    pos_y += ct.px_h / 2.0

    # -------------------------------------------------------------------------
    # 3. Coordinate Reprojection with pyproj
    # -------------------------------------------------------------------------
    try:
        # Check if transformer object already exists on ct, otherwise build one
        if hasattr(ct, 'pyproj_transformer') and ct.pyproj_transformer is not None:
            transformer = ct.pyproj_transformer
        else:
            transformer = pyproj.Transformer.from_crs(ct.crs, "EPSG:4326", always_xy=True)

        # Transform point: always_xy=True ensures (x, y) -> (lon, lat)
        lon, lat = transformer.transform(pos_x, pos_y)
        center = (lon, lat, 0.0)
    except Exception:
        print(f"pyproj transformation failed for point: ({pos_x}, {pos_y})")
        center = (0, 0, 0)

    return center


@Error_Handler
def loadLandsatLUTs(path: str | Path) -> dict[str, dict[str, Any]]:
    """Loads Landsat Rayleigh Look-Up Tables (LUTs) from HDF4 files into a nested dictionary.

    Searches the target directory for HDF4 LUT files, parses central wavelengths from 
    filenames, and loads Stokes vector components (I, Q, U), Rayleigh optical 
    thickness, zenith angles, and wind speed parameters.

    Args:
        path (str | Path): Path to directory containing Rayleigh LUT HDF4 files.

    Returns:
        dict[str, dict[str, Any]]: A nested dictionary structured as:
            - Key: Wavelength identifier string (e.g., '443', '561')
            - Value: Dictionary containing:
                - 'I', 'Q', 'U': Stokes vector component arrays
                - 'optical_thickness': Rayleigh optical thickness array ('taur')
                - 'theta_solar_zenith': Solar zenith angle array ('solz')
                - 'theta_viewing': Viewing zenith angle array ('senz')
                - 'sigma_wind': Wind speed parameter array ('sigma')

    Raises:
        FileNotFoundError: If the provided directory does not exist.
    """
    global rayleigh_lut

    # -------------------------------------------------------------------------
    # 1. Path Initialization & File Discovery
    # -------------------------------------------------------------------------
    path_obj = Path(path)
    if not path_obj.is_dir():
        raise FileNotFoundError(f"LUT directory not found: {path_obj}")

    rayleigh_lut = {}

    # Discover HDF4 lookup files
    lut_list = [f for f in path_obj.iterdir() if f.is_file() and '.hdf' in f.name.lower()]

    # -------------------------------------------------------------------------
    # 2. Extract Stokes Vector Components & Metadata Across Wavelengths
    # -------------------------------------------------------------------------
    for file_path in lut_list:
        file_name = file_path.name
        
        # Extract band wavelength tag (e.g., 'rayleigh_443_iqu.hdf' -> '443')
        wavelength = file_name.split('_iqu')[0].split('_')[-1]
        
        file_str = str(file_path)
        rayleigh_lut[wavelength] = {
            # Stokes vector components (dimensioned by wind, solar zenith, etc.)
            'I': read_hdf4_variable(file_str, 'i_ray'),
            'Q': read_hdf4_variable(file_str, 'q_ray'),
            'U': read_hdf4_variable(file_str, 'u_ray'),
            
            # Rayleigh optical thickness
            'optical_thickness': read_hdf4_variable(file_str, 'taur'),
            
            # Angle grids
            'theta_solar_zenith': read_hdf4_variable(file_str, 'solz'),
            'theta_viewing': read_hdf4_variable(file_str, 'senz'),
            
            # Surface wind parameters
            'sigma_wind': read_hdf4_variable(file_str, 'sigma'),
        }

    return rayleigh_lut


@Error_Handler
def loadSentinelLUTs(sensor: str) -> dict[str, dict[str, Any]]:
    """Loads Sentinel-2 Rayleigh Look-Up Tables (LUTs) from HDF4 files into a nested dictionary.

    Selects sensor-specific (S2A vs S2B/S2C) band mapping and directory paths, parses 
    wavelength lookup files, remaps band numbers to standardized wavelength tags, 
    and loads Stokes vector components (I, Q, U), optical thickness, viewing geometry, 
    and wind parameters.

    Args:
        sensor (str): Sentinel-2 sensor name or identifier string (e.g., 'MSI', 'S2A', 'S2B').

    Returns:
        dict[str, dict[str, Any]]: A nested dictionary structured as:
            - Key: Adjusted standard wavelength string (e.g., '443', '490', '560')
            - Value: Dictionary containing:
                - 'I', 'Q', 'U': Stokes vector component arrays
                - 'optical_thickness': Rayleigh optical thickness array ('taur')
                - 'theta_solar_zenith': Solar zenith angle array ('solz')
                - 'theta_viewing': Viewing zenith angle array ('senz')
                - 'sigma_wind': Wind speed parameter array ('sigma')

    Raises:
        FileNotFoundError: If the target LUT directory does not exist.
    """
    # -------------------------------------------------------------------------
    # 1. Define Sensor-Specific Directories and Band Mappings
    # -------------------------------------------------------------------------
    if 'a' in sensor.lower():
        lut_dir = Path("LUTs/s2a/rayleigh")
        bands = {
            '443': '01', '492': '02', '560': '03', '665': '04',
            '704': '05', '740': '06', '783': '07', '835': '08',
            '865': '8A', '945': '09', '1374': '10', '1613': '11', '2200': '12'
        }
    else:
        lut_dir = Path("LUTs/s2b/rayleigh")
        bands = {
            '442': '01', '492': '02', '559': '03', '665': '04',
            '704': '05', '739': '06', '780': '07', '835': '08',
            '864': '8A', '943': '09', '1377': '10', '1611': '11', '2184': '12'
        }

    if not lut_dir.is_dir():
        raise FileNotFoundError(f"Sentinel LUT directory not found at: {lut_dir}")

    # Standardized output wavelength keys mapped by band number
    adjusted_wavelengths = {
        '01': '443', '02': '490', '03': '560', '04': '665',
        '05': '705', '06': '740', '07': '783', '08': '833',
        '8A': '864', '09': '944', '10': '1375', '11': '1609', '12': '2200'
    }

    rayleigh_lut = {}

    # -------------------------------------------------------------------------
    # 2. Discover HDF4 Files in LUT Directory
    # -------------------------------------------------------------------------
    lut_list = [f for f in lut_dir.iterdir() if f.is_file() and '.hdf' in f.name.lower()]

    # -------------------------------------------------------------------------
    # 3. Read Stokes Vectors & Parameters Across Bands
    # -------------------------------------------------------------------------
    for file_path in lut_list:
        file_name = file_path.name

        # Extract central wavelength tag (e.g., 's2a_rayleigh_443_iqu.hdf' -> '443')
        raw_wavelength = file_name.split('_iqu')[0].split('_')[-1]
        band_num = bands[raw_wavelength]
        adjusted_wavelength = adjusted_wavelengths[band_num]

        file_str = str(file_path)
        rayleigh_lut[adjusted_wavelength] = {
            # Stokes vector components
            'I': read_hdf4_variable(file_str, 'i_ray'),
            'Q': read_hdf4_variable(file_str, 'q_ray'),
            'U': read_hdf4_variable(file_str, 'u_ray'),

            # Rayleigh optical thickness
            'optical_thickness': read_hdf4_variable(file_str, 'taur'),

            # Solar and viewing angle grids
            'theta_solar_zenith': read_hdf4_variable(file_str, 'solz'),
            'theta_viewing': read_hdf4_variable(file_str, 'senz'),

            # Wind speed parameter
            'sigma_wind': read_hdf4_variable(file_str, 'sigma'),
        }

    return rayleigh_lut


@Error_Handler
def read_landsat_images(
    scene_id: str,
    path: str | Path,
    fixed_resolution: int = 30,
) -> xr.Dataset:
    """Load Landsat OLI image bands and geometry into a unified xarray Dataset.

    Parameters
    ----------
    scene_id : str
        Landsat scene identifier.
    path : str or Path
        Directory containing Landsat image files.
    fixed_resolution : int, default=30
        Target spatial resolution in meters.

    Returns
    -------
    xr.Dataset
        Dataset where the spectral bands form the main 'reflectance' DataArray
        (dims: wavelength, y, x), geometry layers are data variables, and CRS /
        transforms are stored in global attributes.
    """
    # -------------------------------------------------------------------------
    # 1. Input Validation & Path Setup
    # -------------------------------------------------------------------------
    if not isinstance(scene_id, str):
        raise TypeError("scene_id must be a string.")
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path object.")
    if not isinstance(fixed_resolution, int):
        raise TypeError("fixed_resolution must be an integer.")

    path_obj = Path(path)
    all_files = [f.name for f in path_obj.iterdir() if f.is_file()]

    # -------------------------------------------------------------------------
    # 2. File Discovery
    # -------------------------------------------------------------------------
    tif_files = [
        f for f in all_files
        if f.endswith(".TIF")
        and any(tier in f for tier in ["_T2", "_T1", "_RT"])
        and scene_id in f
    ]
    img_names_list = [f for f in tif_files if "_B" in f]

    qa_list = [f for f in all_files if "QA_PIXEL" in f and scene_id in f]
    view_az_img = [f for f in all_files if "_VAA" in f and scene_id in f][0]
    view_zen_img = [f for f in all_files if "_VZA" in f and scene_id in f][0]
    sol_az_img = [f for f in all_files if "_SAA" in f and scene_id in f][0]
    sol_zen_img = [f for f in all_files if "_SZA" in f and scene_id in f][0]

    processed_wavelengths = [
        "443", "482", "561", "655", "865", "1609", "2201"
    ]
    assert len(img_names_list) > 0, "empty rhot list"

    # Helper function to read a raster into a 2D numpy array
    def read_raster(file_name: str) -> tuple[np.ndarray, str, rasterio.Affine]:
        with rasterio.open(path_obj / file_name) as src:
            return src.read(1), str(src.crs), src.transform

    # -------------------------------------------------------------------------
    # 3. Read Ancillary Geometry & QA
    # -------------------------------------------------------------------------
    v_az, ref_crs, ref_transform = read_raster(view_az_img)
    v_zen, _, _ = read_raster(view_zen_img)
    s_az, _, _ = read_raster(sol_az_img)
    s_zen, _, _ = read_raster(sol_zen_img)

    quality_arr = read_raster(qa_list[0])[0] if qa_list else None

    # -------------------------------------------------------------------------
    # 4. Read Spectral Bands & Stack into 3D Array
    # -------------------------------------------------------------------------
    band_arrays = []
    active_wavelengths = []

    for img_name in img_names_list:
        passband_ID = img_name.split("_B")[1].split(".")[0]
        wavelength = OLI_wavelengths[passband_ID]

        if wavelength not in processed_wavelengths:
            continue

        data, _, _ = read_raster(img_name)
        band_arrays.append(data)
        active_wavelengths.append(int(wavelength))

    # Sort spectral bands by wavelength
    sorted_pairs = sorted(zip(active_wavelengths, band_arrays), key=lambda x: x[0])
    sorted_wavelengths = [p[0] for p in sorted_pairs]
    # Stack 2D arrays into a 3D array with shape (wavelength, y, x)
    spectral_stack = np.stack([p[1] for p in sorted_pairs], axis=0)

    # Calculate spatial coordinates (pixel center positions)
    height, width = v_az.shape
    x_coords = ref_transform.c + (np.arange(width) + 0.5) * ref_transform.a
    y_coords = ref_transform.f + (np.arange(height) + 0.5) * ref_transform.e

    # -------------------------------------------------------------------------
    # 5. Assemble Final Dataset
    # -------------------------------------------------------------------------
    data_vars = {
        # Main variable holding all spectral bands
        "rhot": (("wavelength", "y", "x"), spectral_stack),
        # Geometry and QA data variables
        "viewing_azimuth": (("y", "x"), v_az),
        "viewing_zenith": (("y", "x"), v_zen),
        "solar_azimuth": (("y", "x"), s_az),
        "solar_zenith": (("y", "x"), s_zen),
    }

    if quality_arr is not None:
        data_vars["quality_flag"] = (("y", "x"), quality_arr)

    # Transformer for lat/lon conversions stored in attributes
    transformer = pyproj.Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "wavelength": sorted_wavelengths,
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "scene_id": scene_id,
            "crs": ref_crs,
            "transform": tuple(ref_transform),
            "pyproj_transformer": transformer,
            "fixed_resolution": fixed_resolution,
        },
    )

    return ds


@Error_Handler
def read_sentinel_images(
    scene_id: str,
    tile_id: str,
    path: str | Path,
    fixed_resolution: int = 30,
    show: bool = False,
) -> xr.Dataset:
    """Load Sentinel-2 MSI image bands and solar/viewing geometry into an xarray Dataset.

    Reads JPEG2000 (.jp2) surface band files and GeoTIFF geometry files, aligns 
    spectral arrays by wavelength, and packages spectral reflectance, per-band 
    viewing angles, and solar angles into a unified multidimensional dataset.

    Args:
        scene_id (str): Sentinel-2 scene identifier string.
        tile_id (str): MGRS tile identifier (e.g., 'T18TWL').
        path (str | Path): Path to directory containing Sentinel image band files.
        fixed_resolution (int, optional): Target spatial resolution in meters. 
            Defaults to 30.
        show (bool, optional): Reserved flag for visualization triggers. 
            Defaults to False.

    Returns:
        xr.Dataset: Dataset structured with:
            - Data Variable 'reflectance': 3D DataArray (wavelength, y, x)
            - Data Variable 'viewing_azimuth': 3D DataArray (wavelength, y, x)
            - Data Variable 'viewing_zenith': 3D DataArray (wavelength, y, x)
            - Data Variables 'solar_azimuth', 'solar_zenith': 2D DataArrays (y, x)
            - Coordinates: wavelength (int), x (float), y (float)
            - Attributes: crs, transform tuple, pyproj_transformer, scene_id, tile_id

    Raises:
        AssertionError: If no image band files are found or if viewing angle counts 
            do not match the target wavelength array count.
    """
    # -------------------------------------------------------------------------
    # 1. Path Initialization & Target Wavelength Definitions
    # -------------------------------------------------------------------------
    path_obj = Path(path)
    all_files = [f.name for f in path_obj.iterdir() if f.is_file()]

    processed_wavelengths = [
        "443", "490", "560", "665", "705", "740", "780", 
        "833", "864", "944", "1375", "1609", "2200"
    ]

    # Find JPEG2000 spectral band files
    gtif_files = [
        f for f in all_files 
        if f.endswith('.jp2') and '_B' in f and tile_id in f
    ]

    if len(gtif_files) == 0:
        print(f"File count for tile '{tile_id}': {len(gtif_files)}")
        print(f"Directory listing: {all_files}")
        raise AssertionError(f"No JP2 band files matching tile '{tile_id}' found in {path_obj}")

    # Helper function to read a raster file and retrieve spatial metadata
    def read_raster(file_name: str) -> tuple[np.ndarray, str, rasterio.Affine]:
        with rasterio.open(path_obj / file_name) as src:
            return src.read(1), str(src.crs), src.transform

    # -------------------------------------------------------------------------
    # 2. Load Solar Geometry Arrays (Scene-Wide)
    # -------------------------------------------------------------------------
    sol_az_img = [f for f in all_files if '_SAA' in f and scene_id in f][0]
    sol_zen_img = [f for f in all_files if '_SZA' in f and scene_id in f][0]

    s_az_data, ref_crs, ref_transform = read_raster(sol_az_img)
    s_zen_data, _, _ = read_raster(sol_zen_img)

    # -------------------------------------------------------------------------
    # 3. Read Spectral Bands (JP2 Files)
    # -------------------------------------------------------------------------
    band_arrays = {}
    
    for file_name in gtif_files:
        # Extract band number identifier (e.g., 'B02' -> '02')
        band_num = file_name.split('_')[-1].split('.')[0].replace('B', '')
        
        # Resolve to central wavelength using global dictionary lookup
        wavelength = str(MSI_wavelengths[band_num])

        if wavelength not in processed_wavelengths:
            continue

        data, _, _ = read_raster(file_name)
        band_arrays[wavelength] = data

    # -------------------------------------------------------------------------
    # 4. Read Per-Band Viewing Geometry Layers
    # -------------------------------------------------------------------------
    view_az_files = [f for f in all_files if '_VAA' in f and scene_id in f]
    view_zen_files = [f for f in all_files if '_VZA' in f and scene_id in f]

    v_az_arrays = {}
    v_zen_arrays = {}

    for file_az in view_az_files:
        wavelength_az = file_az.split('_')[-1].replace('nm.TIF', '')
        if wavelength_az in processed_wavelengths:
            data, _, _ = read_raster(file_az)
            v_az_arrays[wavelength_az] = data

    for file_zen in view_zen_files:
        wavelength_zen = file_zen.split('_')[-1].replace('nm.TIF', '')
        if wavelength_zen in processed_wavelengths:
            data, _, _ = read_raster(file_zen)
            v_zen_arrays[wavelength_zen] = data

    assert len(v_az_arrays) == len(processed_wavelengths), (
        f"Mismatched viewing azimuth angle image count ({len(v_az_arrays)} / {len(processed_wavelengths)})"
    )

    # -------------------------------------------------------------------------
    # 5. Sort & Stack Arrays Along Wavelength Dimension
    # -------------------------------------------------------------------------
    sorted_wavelength_ints = sorted([int(w) for w in processed_wavelengths])
    sorted_wavelength_strs = [str(w) for w in sorted_wavelength_ints]

    reflectance_stack = np.stack([band_arrays[w] for w in sorted_wavelength_strs], axis=0)
    v_az_stack = np.stack([v_az_arrays[w] for w in sorted_wavelength_strs], axis=0)
    v_zen_stack = np.stack([v_zen_arrays[w] for w in sorted_wavelength_strs], axis=0)

    # -------------------------------------------------------------------------
    # 6. Construct Spatial Coordinates & Assemble Dataset
    # -------------------------------------------------------------------------
    height, width = s_az_data.shape
    x_coords = ref_transform.c + (np.arange(width) + 0.5) * ref_transform.a
    y_coords = ref_transform.f + (np.arange(height) + 0.5) * ref_transform.e

    transformer = pyproj.Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)

    ds = xr.Dataset(
        data_vars={
            # Primary 3D Spectral Refectance Cube
            "rhot": (("wavelength", "y", "x"), reflectance_stack),
            # Per-Band 3D Viewing Geometry Cubes
            "viewing_azimuth": (("wavelength", "y", "x"), v_az_stack),
            "viewing_zenith": (("wavelength", "y", "x"), v_zen_stack),
            # Scene 2D Solar Geometry Layers
            "solar_azimuth": (("y", "x"), s_az_data),
            "solar_zenith": (("y", "x"), s_zen_data),
        },
        coords={
            "wavelength": sorted_wavelength_ints,
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "scene_id": scene_id,
            "tile_id": tile_id,
            "crs": ref_crs,
            "transform": tuple(ref_transform),
            "pyproj_transformer": transformer,
            "fixed_resolution": fixed_resolution,
        },
    )

    return ds


@Error_Handler
def load_landsat(
    scene_id: str,
    input_dir: str | Path = "data/",
    ancillary_dir: str | Path = "ancillary/",
    stepsize: int = 1,
    fixed_resolution: int = 30,
    no_gas_absorption: bool = True,
) -> xr.Dataset:
    """Loads Landsat scene data, metadata, ancillary metrics, and LUTs into an xarray Dataset.

    Integrates scene metadata, Earth-Sun distance, local atmospheric ancillary data,
    SRTM elevation, and Rayleigh Look-Up Tables (LUTs) into a structured multidimensional
    xarray Dataset.
    """
    input_path = Path(input_dir)
    # print(f"Operating in: {Path.cwd()}")

    # -------------------------------------------------------------------------
    # 1. Parse Date and Metadata
    # -------------------------------------------------------------------------
    # Example scene_id: LC08_L1TP_001070_20200202_20200211_01_T2
    date = dt.strptime(scene_id.split("_")[3], "%Y%m%d").strftime("%Y-%m-%d")

    meta = load_landsat_meta(input_path)
    hour = int(meta["IMAGE_ATTRIBUTES"]["SCENE_CENTER_TIME"].split(":")[0])

    # -------------------------------------------------------------------------
    # 2. Read Image Bands and Geometry Data via updated read_landsat_images
    # -------------------------------------------------------------------------
    img_ds = read_landsat_images(
        scene_id=scene_id,
        path=input_path,
        fixed_resolution=fixed_resolution,
    )

    width = img_ds.sizes["x"]
    height = img_ds.sizes["y"]
    sensor = "OLI"

    # Calculate Earth-Sun distance
    earth_sun_distance = get_earth_sun_distance(date)

    # -------------------------------------------------------------------------
    # 3. Search and Read Atmospheric Ancillary Data
    # -------------------------------------------------------------------------
    ancillary_path = ancillary_dir / f"{date}"
    file_list, use_gmao = search_local_ancillary(ancillary_path, date, hour)
    ancillary_data = read_ancillary(date, hour, sensor, file_list, use_gmao)

    # -------------------------------------------------------------------------
    # 4. Load Rayleigh Look-Up Tables (LUTs)
    # -------------------------------------------------------------------------
    lut_list = loadLandsatLUTs( (Path(__file__).resolve().parent /("LUTs/oli/rayleigh")))
    assert len(lut_list) > 0, "No Rayleigh LUT entries found."

    # -------------------------------------------------------------------------
    # 5. Extract Surface Elevation (SRTM Altitude)
    # -------------------------------------------------------------------------
    try:
        # Use pyproj transformer stored in attributes if available, or fall back to center coords
        if "pyproj_transformer" in img_ds.attrs:
            center_x = float(img_ds.x[width // 2])
            center_y = float(img_ds.y[height // 2])
            center_lon, center_lat = img_ds.attrs["pyproj_transformer"].transform(center_x, center_y)
        else:
            center_transform = img_ds.attrs.get("transform")
            center_lon, center_lat, _ = get_center(width, height, center_transform)

        assert center_lat != 0, "Invalid latitude center returned"
        altitude = ancillary_data["altitude"](center_lat, center_lon)
        assert altitude is not None, "Elevation lookup returned None"
    except Exception:
        print(f"SRTM mapping data failed for {scene_id}, defaulting to sea-level altitude (0m)")
        altitude = 0.0

    # -------------------------------------------------------------------------
    # 6. Merge Global Metadata into Dataset Attributes
    # -------------------------------------------------------------------------
    img_ds.attrs.update(
        {
            "sensor": sensor,
            "date": date,
            "hour": hour,
            "earth_sun_distance": earth_sun_distance,
            "altitude": altitude,
            "stepsize": stepsize,
            "no_gas_absorption": no_gas_absorption,
            "rad_offset": 0,
            "cloudless_mask": None,
            "outdir": "corrected_images/",
            "ancillary_data": ancillary_data,
            "metadata": meta,
            "rayleigh_lut": lut_list,
            "center_lat": center_lat,
            "center_lon": center_lon,
        }
    )

    return img_ds   


@Error_Handler
def mask_s2cloudless(
    gtifs: dict[str, Any],
    width: int,
    height: int,
    rad_offset: float,
    fixed_resolution: int
) -> np.ndarray:
    """Generates a binary cloudless mask for Sentinel-2 using s2cloudless.

    Extracts required spectral bands, applies radiometric offsets and scale factors,
    runs the `S2PixelCloudDetector`, inverts mask values (1 for clear, 0 for cloudy),
    and resamples the mask to the target spatial grid.

    Args:
        gtifs (dict[str, Any]): Dictionary mapping band wavelength strings to dataset objects.
        width (int): Target grid width in pixels.
        height (int): Target grid height in pixels.
        rad_offset (float): Radiometric offset value to apply to raw digital numbers.
        fixed_resolution (int): Target pixel resolution in meters.

    Returns:
        np.ndarray: 2D floating-point numpy array where 1.0 indicates clear/cloudless
            pixels and 0.0 indicates cloud coverage.
    """
    # Required spectral bands for S2PixelCloudDetector (10 bands)
    used_wavelengths = {'443', '490', '665', '705', '833', '864', '944', '1375', '1609', '2200'}
    images = []

    # -------------------------------------------------------------------------
    # 1. Read and Rescale Selected Wavelengths Sequentially
    # -------------------------------------------------------------------------
    sorted_wavelengths = sorted(gtifs.keys(), key=lambda x: int(x))
    for wavelength in sorted_wavelengths:
        if wavelength not in used_wavelengths:
            continue

        gtif = gtifs[wavelength]
        
        # Support rasterio-style read(1) or GDAL-style ReadAsArray
        if hasattr(gtif, 'ReadAsArray'):
            temp = gtif.ReadAsArray(buf_xsize=width, buf_ysize=height)
        elif hasattr(gtif, 'read'):
            temp = gtif.read(1, out_shape=(height, width))
        else:
            temp = np.array(gtif)

        # Apply offset and DN to TOA Reflectance scaling
        img = (temp.astype(np.float32) + rad_offset) / 10000.0
        images.append(img)
        del temp

    print(f"Loaded {len(images)} full bands for s2cloudless mask generation.")

    # Stack along channels: shape -> (height, width, bands)
    bands = np.dstack(images)
    del images

    # -------------------------------------------------------------------------
    # 2. Run s2cloudless Pixel Detector
    # -------------------------------------------------------------------------
    threshold = 0.4
    cloud_detector = S2PixelCloudDetector(
        threshold=threshold, 
        average_over=4, 
        dilation_size=2, 
        all_bands=False
    )

    # Output mask: 1 = Cloud, 0 = Non-cloud
    raw_cloud_mask = cloud_detector.get_cloud_masks(bands[np.newaxis, ...])
    cloud_mask = np.squeeze(raw_cloud_mask).astype(np.float64)

    del bands
    del cloud_detector

    # -------------------------------------------------------------------------
    # 3. Invert Mask Logic (1 = Clear/Cloudless, 0 = Cloud)
    # -------------------------------------------------------------------------
    cloud_mask[cloud_mask == 1] = 3
    cloud_mask[cloud_mask == 0] = 1
    cloud_mask[cloud_mask == 3] = 0

    # -------------------------------------------------------------------------
    # 4. Target Dimensions and Spatial Resampling
    # -------------------------------------------------------------------------
    ref_gtif = next(iter(gtifs.values()))
    
    if hasattr(ref_gtif, 'GetGeoTransform'):
        ref_scale = ref_gtif.GetGeoTransform()[1] / fixed_resolution
        target_rows = int(ref_gtif.RasterYSize * ref_scale)
        target_cols = int(ref_gtif.RasterXSize * ref_scale)
    else:
        # rasterio or tuple fallback
        pixel_width = getattr(ref_gtif, 'transform', (None, 30))[1]
        ref_scale = abs(pixel_width) / fixed_resolution
        target_rows = int(ref_gtif.shape[0] * ref_scale)
        target_cols = int(ref_gtif.shape[1] * ref_scale)

    print(f"Dimensions of unscaled cloud mask: {cloud_mask.shape}")

    if cloud_mask.shape[0] != target_rows or cloud_mask.shape[1] != target_cols:
        zoom_y = target_rows / cloud_mask.shape[0]
        zoom_x = target_cols / cloud_mask.shape[1]

        # Bilinear scaling
        cloud_mask = zoom(cloud_mask, [zoom_y, zoom_x], order=1)

        # Threshold to eliminate interpolation artifacts
        cloud_mask = (cloud_mask > 0.5).astype(np.float64)

    print(f"Dimensions of scaled cloud mask: {cloud_mask.shape}")

    return cloud_mask


@Error_Handler
def load_msi_chunk_for_s2cloudless(
    gtifs: dict[str, Any],
    numX: int,
    ystart: int,
    ystep: int,
    rad_offset: float,
    fixed_resolution: int
) -> list[np.ndarray]:
    """Loads a spatial image chunk across required s2cloudless bands.

    Computes target resampling grid dimensions from a reference dataset and
    slices the corresponding region across all required bands.

    Args:
        gtifs (dict[str, Any]): Dictionary mapping band wavelength strings to dataset objects.
        numX (int): Number of horizontal pixels (columns) to read.
        ystart (int): Vertical starting index (row offset).
        ystep (int): Number of vertical pixels (rows) to read.
        rad_offset (float): Radiometric offset value.
        fixed_resolution (int): Target resolution in meters.

    Returns:
        list[np.ndarray]: List of 2D numpy arrays corresponding to sliced band data.
    """
    used_wavelengths = {'443', '490', '665', '705', '833', '864', '944', '1375', '1609', '2200'}

    # -------------------------------------------------------------------------
    # 1. Derive Resampled Output Grid Dimensions
    # -------------------------------------------------------------------------
    ref_gtif = next(iter(gtifs.values()))
    
    if hasattr(ref_gtif, 'GetGeoTransform'):
        ref_scale = ref_gtif.GetGeoTransform()[1] / fixed_resolution
        out_rows = int(ref_gtif.RasterYSize * ref_scale)
        out_cols = int(ref_gtif.RasterXSize * ref_scale)
    else:
        pixel_width = getattr(ref_gtif, 'transform', (None, 30))[0]
        ref_scale = abs(pixel_width) / fixed_resolution
        out_rows = int(ref_gtif.shape[0] * ref_scale)
        out_cols = int(ref_gtif.shape[1] * ref_scale)

    # -------------------------------------------------------------------------
    # 2. Extract and Resample Chunks for S2 Cloudless Wavelengths
    # -------------------------------------------------------------------------
    rhot_images = {}
    for wavelength, gtif in gtifs.items():
        if wavelength not in used_wavelengths:
            continue
        rhot_images[wavelength] = read_resample_sentinel_range(
            gtif, rad_offset, out_rows, out_cols, ystart, numX, ystep
        )

    return sorted_list_from_dict(rhot_images)


def read_resample_sentinel_range(
    gtif: Any,
    rad_offset: float,
    out_rows: int,
    out_cols: int,
    ystart: int,
    numX: int,
    ystep: int
) -> np.ndarray:
    """Resamples a band to a target grid size and extracts a specific 2D bounding window.

    Args:
        gtif (Any): Band dataset object.
        rad_offset (float): Radiometric offset applied to raw digital numbers.
        out_rows (int): Resampled full-raster row height.
        out_cols (int): Resampled full-raster column width.
        ystart (int): Row offset for vertical cropping window.
        numX (int): Column length for horizontal cropping window.
        ystep (int): Row length for vertical cropping window.

    Returns:
        np.ndarray: 2D resampled and window-cropped reflectance array.
    """
    # -------------------------------------------------------------------------
    # 1. Resample Band to Shared Grid and Read Array
    # -------------------------------------------------------------------------
    if hasattr(gtif, 'ReadAsArray'):
        temp = gtif.ReadAsArray(buf_xsize=out_cols, buf_ysize=out_rows)
    elif hasattr(gtif, 'read'):
        temp = gtif.read(1, out_shape=(out_rows, out_cols))
    else:
        temp = np.array(gtif)

    # -------------------------------------------------------------------------
    # 2. Window Crop and Conversion to TOA Reflectance
    # -------------------------------------------------------------------------
    img = (temp[ystart : ystart + ystep, 0 : numX].astype(np.float32) + rad_offset) / 10000.0

    return img


@Error_Handler
def load_sentinel(
    scene_id: str,
    input_dir: str | Path = 'data/',
    stepsize: int = 1,
    fixed_resolution: int = 30,
    no_gas_absorption: bool = True,
) -> xr.Dataset:
    """Loads Sentinel-2 MSI scene data, metadata, ancillary metrics, cloud mask, and LUTs into an xarray Dataset.

    Parses SAFE directory structure and XML metadata, extracts radiometric offset,
    computes solar/sensor angles, loads local atmospheric ancillary data and SRTM
    altitude, generates an s2cloudless mask, and bundles everything into a structured
    xarray Dataset.

    Args:
        scene_id (str): Sentinel-2 scene/granule identifier.
        input_dir (str | Path, optional): Directory containing S2 SAFE datasets. Defaults to 'data/'.
        stepsize (int, optional): Processing step size parameter. Defaults to 1.
        fixed_resolution (int, optional): Target spatial resolution in meters. Defaults to 30.
        no_gas_absorption (bool, optional): Flag to disable gaseous absorption correction. Defaults to True.

    Returns:
        xr.Dataset: Unified dataset containing:
            - Data Variables:
                - `reflectance` (wavelength, y, x): 3D spectral band image data
                - `cloudless_mask` (y, x): Binary cloudless mask raster
                - `solar_azimuth` (y, x): Solar azimuth angle raster
                - `solar_zenith` (y, x): Solar zenith angle raster
                - `viewing_azimuth` (wavelength, y, x): Band-specific viewing azimuth angles
                - `viewing_zenith` (wavelength, y, x): Band-specific viewing zenith angles
            - Coordinates:
                - `wavelength`: Wavelength identifier strings
                - `x`, `y`: Spatial coordinate grids
            - Attributes:
                - Metadata, ancillary data dictionary, Earth-Sun distance, SRTM altitude,
                  radiometric offset, Rayleigh LUTs, projection, transform, and processing flags.
    """
    input_path = Path(input_dir)

    # -------------------------------------------------------------------------
    # 1. Parse Date, Timestamp, and Path Structure (.SAFE)
    # -------------------------------------------------------------------------
    # Example scene_id: S2A_MSIL1C_20200202T102121_N0208_R065_T31UFU_20200202T122000
    timestamp_str = scene_id.split('_')[2]
    date = dt.strptime(timestamp_str.split('T')[0], '%Y%m%d').strftime("%Y-%m-%d")
    hour = int(timestamp_str.split('T')[1][:2])

    # Find matching .SAFE directory
    safe_dirs = [f for f in input_path.iterdir() if f.is_dir() and '.SAFE' in f.name and scene_id in f.name]
    if not safe_dirs:
        raise FileNotFoundError(f"No matching .SAFE directory found for scene: {scene_id}")
    
    subfolder = safe_dirs[0]
    granule_path = subfolder / "GRANULE"
    pseudo_id = list(granule_path.iterdir())[0].name
    image_path = granule_path / pseudo_id / "IMG_DATA"

    # Load XML metadata
    meta = load_sentinel_meta(subfolder)
    sensor = meta['General_Info']['Product_Info']['PRODUCT_URI'].split('_')[0][1:]

    # -------------------------------------------------------------------------
    # 2. Radiometric Offset & Image Reading
    # -------------------------------------------------------------------------
    try:
        rad_offsets = meta['General_Info']['Product_Image_Characteristics']['Radiometric_Offset_List']['RADIO_ADD_OFFSET']
        rad_offset = float(rad_offsets[0]['#text'])
        assert rad_offset == float(rad_offsets[3]['#text'])
    except (KeyError, IndexError, AssertionError):
        rad_offset = 0.0
    
    print(f"Radiometric offset: {rad_offset}")

    tile_id = scene_id.split('_')[5]

    (
        gtifs, 
        transforms, 
        projections, 
        viewing_azimuth_gtifs, 
        viewing_zenith_gtifs, 
        solar_azimuth_gtif, 
        solar_zenith_gtif
    ) = read_sentinel_images(scene_id, tile_id, image_path, fixed_resolution)

    # -------------------------------------------------------------------------
    # 3. Earth-Sun Distance & Spatial Grid Scaling
    # -------------------------------------------------------------------------
    earth_sun_distance = get_earth_sun_distance(date)

    ref_gtif = gtifs['443']
    ct = transforms['443']

    scale_factor = ct.px_w / fixed_resolution
    print(f"Scale factor: {scale_factor}")

    width = int(ref_gtif.shape[1] * scale_factor) if hasattr(ref_gtif, 'shape') else int(ref_gtif.RasterXSize * scale_factor)
    height = int(ref_gtif.shape[0] * scale_factor) if hasattr(ref_gtif, 'shape') else int(ref_gtif.RasterYSize * scale_factor)

    # -------------------------------------------------------------------------
    # 4. Search and Read Local Atmospheric Ancillary Data
    # -------------------------------------------------------------------------
    ancillary_path = Path(f'ancillary/{date}')
    file_list, use_gmao = search_local_ancillary(ancillary_path, date, hour)
    ancillary_data = read_ancillary(date, hour, sensor, file_list, use_gmao)

    # -------------------------------------------------------------------------
    # 5. Load Rayleigh Look-Up Tables (LUTs)
    # -------------------------------------------------------------------------
    lut_list = loadSentinelLUTs(sensor)
    assert len(lut_list) > 0, "No Rayleigh LUT entries found."

    # -------------------------------------------------------------------------
    # 6. Extract Elevation (SRTM Altitude) & Cloud Mask
    # -------------------------------------------------------------------------
    try:
        center_lon, center_lat, _ = get_center(width, height, transforms)
        assert center_lat != 0, "Invalid latitude center returned"
        center_altitude = ancillary_data['altitude_tf'](center_lat, center_lon)
        assert altitude is not None, "Elevation lookup returned None"
    except Exception:
        print(f"SRTM mapping data failed for {scene_id}, defaulting to sea-level altitude (0m)")
        altitude = 0.0

    cloudless_mask = mask_s2cloudless(gtifs, width, height, rad_offset, fixed_resolution)

    # -------------------------------------------------------------------------
    # 7. Stack Bands, Angles, and Build Coordinate Grids
    # -------------------------------------------------------------------------
    wavelength_keys = list(gtifs.keys())

    def to_array(obj: Any) -> np.ndarray:
        return obj.read(1) if hasattr(obj, 'read') else np.array(obj)

    # Convert band dictionary to 3D array (wavelength, y, x)
    image_arrays = [to_array(gtifs[w]) for w in wavelength_keys]
    reflectance_stack = np.stack(image_arrays, axis=0)

    # Convert band-specific viewing angle dicts to 3D arrays (wavelength, y, x)
    viewing_azimuth_stack = np.stack([to_array(viewing_azimuth_gtifs[w]) for w in wavelength_keys], axis=0)
    viewing_zenith_stack = np.stack([to_array(viewing_zenith_gtifs[w]) for w in wavelength_keys], axis=0)

    # Build affine pixel center coordinates
    x_coords = ct.xoffset + (np.arange(width) + 0.5) * ct.px_w
    y_coords = ct.yoffset + (np.arange(height) + 0.5) * ct.px_h

    # -------------------------------------------------------------------------
    # 8. Construct and Return xarray.Dataset
    # -------------------------------------------------------------------------
    ds = xr.Dataset(
        data_vars={
            # 3D Spectral Reflectance Cube (wavelength, y, x)
            "reflectance": (("wavelength", "y", "x"), reflectance_stack),
            
            # 3D Viewing Geometry Layers (wavelength, y, x)
            "viewing_azimuth": (("wavelength", "y", "x"), viewing_azimuth_stack),
            "viewing_zenith": (("wavelength", "y", "x"), viewing_zenith_stack),
            
            # 2D Cloud Mask and Solar Angles (y, x)
            "cloudless_mask": (("y", "x"), to_array(cloudless_mask) if cloudless_mask is not None else np.ones((height, width), dtype=bool)),
            "solar_azimuth": (("y", "x"), to_array(solar_azimuth_gtif)),
            "solar_zenith": (("y", "x"), to_array(solar_zenith_gtif)),
        },
        coords={
            "wavelength": wavelength_keys,
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "scene_id": scene_id,
            "sensor": 'MSI',
            "date": date,
            "hour": hour,
            "earth_sun_distance": earth_sun_distance,
            "altitude": center_altitude,
            "rad_offset": rad_offset,
            "stepsize": stepsize,
            "fixed_resolution": fixed_resolution,
            "no_gas_absorption": no_gas_absorption,
            "outdir": "corrected_images/",
            "projection": projections[0] if isinstance(projections, list) else projections,
            "transform": ct,
            "ancillary_data": ancillary_data,
            "metadata": meta,
            "rayleigh_lut": lut_list,
        },
    )

    return ds     