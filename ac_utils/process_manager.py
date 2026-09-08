# -*- coding: utf-8 -*-

"""
File Name:              process_manager.py
Description:            This code-file contains the list functions needed to download the data/retrieve the ancilliaries needed for the MDN-AC model.

Date Created:           August 25th, 2026
Author:                 William Wainwright, 
Email:                  william.wainwright@ssaihq.com/william.wainwright@nasa.gov
"""

from datetime import datetime
from pathlib import Path
import requests
from io import StringIO
from lxml import etree
import requests
import xarray as xr
import numpy as np

from .load_rayleigh_data import select_closest_file
from .aq_error import Error_Handler


class SessionWithHeaderRedirection(requests.Session):
    """
    This class helps handle the interaction with NASA's Earthdata login system, ensuring that the Authorization header is 
    preserved when redirected to or from the NASA auth host.
    """
    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    # Overrides from the library to keep headers when redirected to or from the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
                    redirect_parsed.hostname != self.AUTH_HOST and \
                    original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return


def determine_sensor(scene_ID: str):
    """
    Determines the sensor type based on the scene ID.

    Args:
        scene_ID (str): Identifier for the satellite scene.

    Returns:
        str: The sensor type ('OLI', 'MSI', or 'Unknown').
    """
    if "LC08" in scene_ID or "LC09" in scene_ID:
        return 'OLI'
    elif 'S2A' in scene_ID or "S2B" in scene_ID or "S2C" in scene_ID:
        return 'MSI'
    return "Unknown"


def extract_date_hour(scene_id: str, sensor: str) -> tuple[str, str]:
    """Extracts and formats the date string and target UTC hour from a satellite scene identifier.

    Parses naming conventions for Landsat (OLI) and Sentinel-2 (MSI) scene IDs to 
    derive a standardized date string and acquisition hour.

    Args:
        scene_id (str): Standardized scene identifier (e.g., Landsat LC08_... or 
            Sentinel-2 S2A_...).
        sensor (str): Sensor type identifier. Accepted values are 'OLI' or 'MSI'.

    Returns:
        tuple[str, str]: A tuple containing:
            - date_formatted (str): Date formatted as 'YYYY-MM-DD'.
            - hour (str): Two-digit UTC hour string (e.g., '16').

    Raises:
        ValueError: If `sensor` is not 'OLI' or 'MSI'.
    """
    # -------------------------------------------------------------------------
    # 1. Parse Landsat (OLI) Scene Naming Structure
    # Example: LC08_L1TP_014032_20230514... -> Date: 20230514
    # -------------------------------------------------------------------------
    if sensor == 'OLI':
        date_str = scene_id.split('_')[3]
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        
        # Landsat product IDs omit exact acquisition time.
        # Defaulting to nominal CONUS overpass hour (~16:00 UTC).
        hour = "16"

    # -------------------------------------------------------------------------
    # 2. Parse Sentinel-2 (MSI) Scene Naming Structure
    # Example: S2A_MSIL1C_20230514T153021... -> Datetime: 20230514T153021
    # -------------------------------------------------------------------------
    elif sensor == 'MSI':
        datetime_str = scene_id.split('_')[2]
        date_str, time_str = datetime_str.split('T')
        
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        hour = time_str[:2]

    else:
        raise ValueError(f"Unknown Sensor '{sensor}'. Expected 'OLI' or 'MSI'.")

    return date_formatted, hour


def get_links(session: requests.Session, url: str) -> tuple[list[str], list[int]]:
    """Parses an OBPG HTML directory page to extract file download URLs and file sizes.

    Fetches the HTML directory listing from NASA's Ocean Biology Processing Group 
    (OBPG) server and parses table elements to retrieve direct file download links 
    along with their expected file sizes in bytes.

    Args:
        session (requests.Session): Authenticated HTTP session object used to make 
            the request.
        url (str): Target directory URL containing the file listings.

    Returns:
        tuple[list[str], list[int]]: A tuple containing:
            - refs (list[str]): List of absolute or relative file download links containing 
              'getfile'.
            - sizes (list[int]): List of expected file sizes in bytes parsed from table cells.
    """
    print(f"Getting links from: {url}")
    
    # -------------------------------------------------------------------------
    # 1. Fetch Directory Page
    # -------------------------------------------------------------------------
    page = session.get(url)
    
    if page.status_code != 200:
        print(f"Failed to access {url} (Status: {page.status_code})")
        return [], []
        
    # -------------------------------------------------------------------------
    # 2. Parse HTML Tree
    # -------------------------------------------------------------------------
    html = page.content.decode("utf-8")
    tree = etree.parse(StringIO(html), parser=etree.HTMLParser())
    
    # -------------------------------------------------------------------------
    # 3. Extract File Sizes & Download Links
    # -------------------------------------------------------------------------
    # Parse numerical byte values from table cells
    sizes = [s.text for s in tree.xpath("//td") if s.text is not None]
    sizes = [int(s) for s in sizes if s.isdigit()]
    
    # Parse target href attributes containing the download endpoint
    refs = tree.xpath("//a")
    refs = [link.get('href', '') for link in refs]
    refs = [link for link in refs if 'getfile' in link]
    
    return refs, sizes


def precheck_links(links: list[str]) -> str | None:
    """Evaluates retrieved links to determine the highest priority GMAO model prefix present.

    Checks a collection of remote URL links against NASA GMAO (Global Modeling and 
    Assimilation Office) ancillary prefixes in hierarchical order of preference 
    (MERRA-2 > IT > FP-IT/FP).

    Args:
        links (list[str]): List of remote URLs or file links retrieved from the server.

    Returns:
        str | None: The highest priority matching GMAO prefix string if found 
            (e.g., 'gmao_merra2'), or None if no valid GMAO prefixes are present.
    """
    # -------------------------------------------------------------------------
    # 1. Define Model Priority & Initialize Presence Tracker
    # Priority order: gmao_merra2 > gmao_it > gmao_fp
    # -------------------------------------------------------------------------
    gmao_prefixes = ['gmao_merra2', 'gmao_it', 'gmao_fp']
    prefix_present = {p: False for p in gmao_prefixes}

    # -------------------------------------------------------------------------
    # 2. Scan Retrieved Links for GMAO Prefixes
    # -------------------------------------------------------------------------
    for link in links:
        link_lower = link.lower()
        for p in gmao_prefixes:
            if p in link_lower:
                prefix_present[p] = True

    # -------------------------------------------------------------------------
    # 3. Select Highest-Priority Available Prefix
    # -------------------------------------------------------------------------
    for p in gmao_prefixes:
        if prefix_present[p]:
            return p

    print("No GMAO ancillaries found for this date, unable to continue processing")
    return None


def check_criteria(link: str, hour: str | int, gmao_prefix: str | None) -> bool:
    """Evaluates whether an ancillary file link matches required processing criteria.

    Filters remote links based on file types (excluding aerosol data) and checks if 
    the timestamp in the filename falls within valid temporal thresholds relative 
    to the target acquisition hour (1-hour window for GEOS-CF, 3-hour window for GMAO MET/PROFILE).

    Args:
        link (str): Remote URL or filename string to evaluate.
        hour (str | int): Target UTC hour of the scene acquisition (0–23).
        gmao_prefix (str | None): Active prioritized GMAO model prefix 
            (e.g., 'gmao_merra2', 'gmao_it', 'gmao_fp') or None.

    Returns:
        bool: True if the file satisfies all filtering and temporal criteria; 
            False otherwise.
    """
    link_lower = link.lower()

    # -------------------------------------------------------------------------
    # 1. Filter Out Excluded Dataset Types
    # -------------------------------------------------------------------------
    # Ignore aerosol files completely
    if '.aer.' in link_lower:
        return False

    target_hour = int(hour)

    try:
        # Extract filename and parse the two-digit UTC hour from the timestamp token
        # Expected token structure example: ...20230514T150000... -> '15'
        file_name = Path(link).name
        time_token = file_name.split('.')[1]
        file_hour = int(time_token.split('T')[1][:2])

        # -------------------------------------------------------------------------
        # 2. Check Hourly Datasets (GMAO GEOS-CF: 1-Hour Tolerance)
        # -------------------------------------------------------------------------
        if 'gmao_geos-cf.' in link_lower:
            if abs(file_hour - target_hour) < 1:
                return True

        # -------------------------------------------------------------------------
        # 3. Check 3-Hourly Datasets (GMAO MET/PROFILE: 3-Hour Tolerance)
        # -------------------------------------------------------------------------
        if gmao_prefix and (gmao_prefix in link_lower):
            if '.met.' in link_lower or '.profile.' in link_lower:
                if abs(file_hour - target_hour) < 3:
                    return True

    except (IndexError, ValueError):
        # Handle malformed filename structures or failed integer conversions
        return False

    return False


def download_retrieve_ancillary(
    scene_id: str,
    date_str: str,
    hour: str,
    EARTHDATA_USERNAME: str,
    EARTHDATA_PASSWORD: str,
    anc_dir: str | Path = './ancillary',
    replace: bool = False,
    max_delta: int = 6
) -> None:
    """Downloads required meteorological NetCDF/HDF ancillary files to the local directory.

    Queries the NASA Ocean Biology Processing Group (OBPG) server for ancillary data 
    matching a specified date and hour, validates remote vs. local file sizes, and 
    streams missing or incomplete datasets to disk.

    Args:
        scene_id (str): Identifier for the satellite scene.
        date_str (str): Date of interest formatted as 'YYYY-MM-DD'.
        hour (str): Target UTC hour for ancillary data filtering.
        EARTHDATA_USERNAME (str): NASA Earthdata Login username.
        EARTHDATA_PASSWORD (str): NASA Earthdata Login password.
        anc_dir (str | Path, optional): Base destination directory for output files. 
            Defaults to './ancillary'.
        replace (bool, optional): If True, forces redownload and replacement of existing 
            files regardless of size validation. Defaults to False.
        max_delta (int, optional): Maximum allowable hour difference for selecting

    Raises:
        ValueError: If `date_str` does not match expected 'YYYY-MM-DD' formatting.
        AssertionError: If retrieved links do not match the parsed file sizes.
    """
    
    BASE_URL = 'https://oceandata.sci.gsfc.nasa.gov/directdataaccess/Ancillary/GLOBAL'

    try:
        d = datetime.strptime(date_str, r"%Y-%m-%d")
    except ValueError:
        raise ValueError("Incorrect Date Format. Expected YYYY-MM-DD")
        
    year = datetime.strftime(d, r'%Y')
    day = datetime.strftime(d, r'%d')
    month = datetime.strftime(d, r'%b')

    anc_dir = Path(anc_dir) / date_str
    anc_dir.mkdir(parents=True, exist_ok=True)
    
    url = f"{BASE_URL}/{year}/{day}-{month}-{year}/"

    session = SessionWithHeaderRedirection(EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
    links, expected_sizes = get_links(session, url)
    
    if not links:
        print("No ancillary links found for this date.")
        return False
        
    assert len(links) == len(expected_sizes), "Mismatch between links and parsed sizes."
    print(f"{len(links)} total links retrieved from directory.")

    # Create a link -> expected_size lookup dictionary for fast size verification
    link_size_map = dict(zip(links, expected_sizes))

    # Select GEOS-CF (Ozone) closest to target hour
    geos_cf_link = select_closest_file(links, lambda f: 'gmao_geos-cf.' in f, hour, max_delta=max_delta)

    # Select Surface MET closest to target hour
    met_link = select_closest_file(links, lambda f: '.met.' in f and not 'gmao_geos-cf.' in f, hour, max_delta=max_delta)

    # Select PROFILE closest to target hour
    profile_link = select_closest_file(links, lambda f: '.profile.' in f and not 'gmao_geos-cf.' in f, hour, max_delta=max_delta)

    # Collect selected links
    target_links = [l for l in [geos_cf_link, met_link, profile_link] if l is not None]

    # Enforce full 3-file complete set requirement
    if len(target_links) < 3:
        missing = []
        if not geos_cf_link: 
            missing.append("GEOS-CF")
        if not met_link: 
            missing.append("MET")
        if not profile_link: 
            missing.append("PROFILE")
        print(f"Cannot complete download. Server missing required file products: {', '.join(missing)}")
        return False

    print("Target remote links resolved by closest time delta:")
    for link in target_links:
        print(f" - {link.split('/')[-1]}")

    # Process and download each resolved target file
    for link in target_links:
        file_name = link.split('/')[-1]
        out_path = Path(anc_dir) / file_name #os.path.join(anc_dir, file_name)
        expected_size = link_size_map[link]

        success = False
        while not success:
            try:
                # Check if file already exists locally
                if Path(out_path).is_file():
                    if not replace:
                        if Path(out_path).stat().st_size == expected_size:
                            print(f"Valid file {file_name} already exists. Skipping.")
                            success = True
                            continue
                        else:
                            print(f"Incomplete file {file_name} found. Redownloading...")
                    else:
                        os.remove(out_path)
                        print(f"Removed existing {file_name} for replacement.")
                        
                # Download stream
                response = session.get(link, stream=True)
                response.raise_for_status()
                
                print(f"Downloading: {file_name}")
                with open(out_path, 'wb') as fd:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        fd.write(chunk)
                
                # Size verification check
                actual_size = Path(out_path).stat().st_size
                if actual_size != expected_size:
                    print(f"Size mismatch for {file_name}! Expected {expected_size}, got {actual_size}. Retrying...")
                    continue
                    
                print(f"Successfully downloaded {file_name}")
                success = True
                
            except requests.exceptions.RequestException as e:
                print(f"HTTP Error downloading {file_name}: {e}")
                break

    print("Ancillary download process complete.")
    return True


@Error_Handler
def process_image_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Apply sensor-specific conversions, scaling, and cosine corrections.

    Processes raw geometry angle variables and scales reflectance data
    based on the sensor metadata stored within the dataset attributes.
    Supports Landsat (OLI) and Sentinel-2 (MSI) workflows.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing scene reflectance, viewing and solar
        geometry variables (`viewing_azimuth`, `viewing_zenith`,
        `solar_azimuth`, `solar_zenith`), and required dataset attributes
        (e.g., `sensor`, `rad_offset`).

    Returns
    -------
    xarray.Dataset
        Dataset with normalized angle geometries (in degrees) and solar zenith
        cosine-corrected top-of-atmosphere (TOA) apparent reflectance (`rhot`).

    Notes
    -----
    - For **OLI (Landsat)**:
      Scales integer angle files by dividing by 100. Converts reflectance
      digital numbers (DN) using standard Collection 2 multiplicative
      (:math:`2 \\times 10^{-5}`) and additive (-0.1) factors, then divides
      by :math:`\\cos(\\theta_s)` (solar zenith angle in radians).
    - For **MSI (Sentinel-2)**:
      Applies baseline radiometric offset (`rad_offset`) if defined in attributes.
    """
    sensor = ds.attrs.get("sensor")

    # 1. Process Geometry Angles
    if sensor == "OLI":
        angle_vars = [ "viewing_azimuth", "viewing_zenith", "solar_azimuth", "solar_zenith", ]
        for var in angle_vars:
            if var in ds.data_vars:
                # Convert raw Landsat integer angles (scaled by 100) to degrees and enforce float32
                if np.issubdtype(ds[var].dtype, np.integer):
                    ds[var] = (ds[var] / 100.0).astype(np.float32)
                else:
                    # Enforce float32 if already floating point (e.g., float64)
                    ds[var] = ds[var].astype(np.float32)

    # 2. Process Reflectance (rhot)
    if sensor == "OLI":
        reflectance_mult = 2e-5
        reflectance_add = -0.1

        # Calculate TOA Reflectance
        toa_reflectance = (ds["rhot"].astype(np.float32) * reflectance_mult) + reflectance_add

        # Solar zenith cosine correction
        # xarray automatically broadcasts 2D cos_sz over 3D reflectance (wavelength, y, x)
        cos_sz = np.cos(np.deg2rad(ds["solar_zenith"].astype(np.float32)))
        ds["rhot"] = toa_reflectance / cos_sz

    elif sensor == "MSI":
        rad_offset = ds.attrs.get("rad_offset", 0)
        quantification_value = ds.attrs.get("quantification_value", 10000.0)

        # Correct for BOA/TOA offset and scale DN to 0-1 reflectance range
        ds["rhot"] = ((ds["rhot"].astype(np.float32) + rad_offset) / quantification_value).astype(np.float32)

    return ds