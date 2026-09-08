# -*- coding: utf-8 -*-
"""Sentinel-2 Viewing and Solar Geometry Angle Extractor and Generator.

Processes Sentinel-2 L1C SAFE metadata to generate solar azimuth, solar zenith,
viewing azimuth, and viewing zenith raster layers using rasterio and scipy.
"""

from pathlib import Path
import sys
from typing import Any
import numpy as np
from scipy.ndimage import zoom
import rasterio
from rasterio.transform import Affine
import xml.etree.ElementTree as ET
import xmltodict

from .driver_S2_SAFE import sentinel2_driver

# -------------------------------------------------------------------------
# Band Wavelength Map (Passband ID -> Central Wavelength Tag in nm)
# -------------------------------------------------------------------------
MSI_WAVELENGTHS: dict[str, str] = {
    '01': '443',
    '02': '490',
    '03': '560',
    '04': '665',
    '05': '705',
    '06': '740',
    '07': '780',
    '08': '833',
    '8A': '864',
    '09': '944',
    '10': '1375',
    '11': '1609',
    '12': '2200',
}


def main(scene_id: str, input_dir: str | Path = 'data/') -> None:
    """Main execution pipeline for Sentinel-2 angle extraction and output writing.

    Args:
        scene_id (str): Sentinel-2 scene identifier string.
        input_dir (str | Path, optional): Directory containing SAFE subfolders. Defaults to 'data/'.

    Raises:
        FileNotFoundError: If matching SAFE directory or granule subfolder cannot be found.
        AssertionError: If output array dimensions mismatch expected wavelength counts.
    """
    print("Starting S2 angle generation")
    input_path = Path(input_dir)

    # -------------------------------------------------------------------------
    # 1. Parse Directory Paths and Metadata Locations
    # -------------------------------------------------------------------------
    safe_dirs = [f for f in input_path.iterdir() if f.is_dir() and '.SAFE' in f.name and scene_id in f.name]
    if not safe_dirs:
        raise FileNotFoundError(f"No .SAFE directory found matching scene: {scene_id}")

    safe_path = safe_dirs[0]
    print(f"S2 path: {safe_path}")

    granule_path = safe_path / "GRANULE"
    pseudo_id = list(granule_path.iterdir())[0].name
    mtd_path = granule_path / pseudo_id
    image_path = mtd_path / "IMG_DATA"

    wavelengths = ['443', '490', '560', '665', '705', '740', '780', '833', '864', '944', '1375', '1609', '2200']

    date_str = scene_id.split('_')[2].split('T')[0]
    year = int(date_str[:4])

    # -------------------------------------------------------------------------
    # 2. Extract or Compute Solar / Viewing Geometry Grids
    # -------------------------------------------------------------------------
    if year < 2017:
        print("Using pre-2017 angle generation")
        height, width = check_dimensions(image_path)

        viewing_azimuth_image, viewing_zenith_image, solar_azimuth_image, solar_zenith_image = get_sentinel_angles(mtd_path)

        for wl in wavelengths:
            viewing_azimuth_image[wl] = resize_array(viewing_azimuth_image[wl], height, width)
            viewing_zenith_image[wl] = resize_array(viewing_zenith_image[wl], height, width)

        solar_azimuth_image = resize_array(np.array(solar_azimuth_image, dtype=np.float32), height, width)
        solar_zenith_image = resize_array(np.array(solar_zenith_image, dtype=np.float32), height, width)

    else:
        # Dynamic driver import for legacy SAFE handling
        l1c = sentinel2_driver(str(safe_path))
        l1c.load_product()

        viewing_azimuth_image = l1c.prod.vaa.values
        viewing_zenith_image = l1c.prod.vza.values
        solar_azimuth_image = l1c.prod.saa.values
        solar_zenith_image = l1c.prod.sza.values

    print("Angles generated, beginning saving")
    assert len(wavelengths) == len(viewing_azimuth_image), "Wavelength count mismatch with generated VAA array count."

    outdir = image_path

    # Extract transform and CRS reference from an existing image band
    ref_image = list(image_path.glob("*.jp2")) or list(image_path.glob("*.tif")) or list(image_path.glob("*.TIF"))
    if ref_image:
        with rasterio.open(ref_image[0]) as ref_ds:
            crs_ref = ref_ds.crs
            transform_ref = ref_ds.transform
    else:
        crs_ref = None
        transform_ref = Affine.identity()

    # -------------------------------------------------------------------------
    # 3. Write Band-Specific Viewing Angles (VAA and VZA) to GeoTIFF
    # -------------------------------------------------------------------------
    for wl, view_az_raw, view_zen_raw in zip(wavelengths, viewing_azimuth_image, viewing_zenith_image):
        view_az = np.copy(view_az_raw).astype(np.float32)
        view_az[(view_az < 0) | (view_az > 36000)] = np.nan

        vaa_file = outdir / f"{scene_id}_VAA_{wl}nm.TIF"
        write_geotiff(vaa_file, view_az, crs=crs_ref, transform=transform_ref)

        view_zen = np.copy(view_zen_raw).astype(np.float32)
        view_zen[(view_zen < 0) | (view_zen > 36000)] = np.nan

        vza_file = outdir / f"{scene_id}_VZA_{wl}nm.TIF"
        write_geotiff(vza_file, view_zen, crs=crs_ref, transform=transform_ref)

    # -------------------------------------------------------------------------
    # 4. Write Solar Angles (SAA and SZA) to GeoTIFF
    # -------------------------------------------------------------------------
    saa_file = outdir / f"{scene_id}_SAA.TIF"
    write_geotiff(saa_file, np.array(solar_azimuth_image, dtype=np.float32), crs=crs_ref, transform=transform_ref)

    sza_file = outdir / f"{scene_id}_SZA.TIF"
    write_geotiff(sza_file, np.array(solar_zenith_image, dtype=np.float32), crs=crs_ref, transform=transform_ref)

    print("All angle images generated and saved")


def write_geotiff(
    filename: str | Path,
    data: np.ndarray,
    crs: Any = None,
    transform: Affine = Affine.identity()
) -> None:
    """Writes a 2D single-band float32 numpy array to a GeoTIFF raster file via rasterio.

    Args:
        filename (str | Path): Output GeoTIFF destination path.
        data (np.ndarray): 2D floating-point array to write.
        crs (Any, optional): Coordinate reference system. Defaults to None.
        transform (Affine, optional): Affine spatial transform. Defaults to Affine.identity().
    """
    height, width = data.shape
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': rasterio.float32,
        'crs': crs,
        'transform': transform,
        'nodata': np.nan,
    }

    with rasterio.open(str(filename), 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def resize_array(var_array: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resizes a 2D array to target pixel row height and column width using zoom.

    Args:
        var_array (np.ndarray): Input 2D numpy array.
        height (int): Target row length.
        width (int): Target column length.

    Returns:
        np.ndarray: Resampled 2D array.
    """
    zoom_y = height / var_array.shape[0]
    zoom_x = width / var_array.shape[1]
    return zoom(var_array, [zoom_y, zoom_x])


def check_dimensions(image_path: str | Path) -> tuple[int, int]:
    """Inspects the first available raster in a path to determine spatial row height and column width.

    Args:
        image_path (str | Path): Directory path containing spatial band files.

    Returns:
        tuple[int, int]: Spatial dimensions formatted as (height, width).

    Raises:
        FileNotFoundError: If no readable image file exists in the directory.
    """
    img_dir = Path(image_path)
    image_files = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jp2', '.tif', '.tiff']]

    if not image_files:
        raise FileNotFoundError(f"No valid raster images (.jp2, .tif) found in: {image_path}")

    with rasterio.open(image_files[0]) as dataset:
        height = dataset.height
        width = dataset.width

    return height, width


def get_sentinel_angles(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[list[str]], list[list[str]]]:
    """Parses Tile MTD XML metadata and processes viewing/solar angle arrays.

    Args:
        path (str | Path): Directory path containing MTD_TL.xml and processed_angle_maps.

    Returns:
        tuple containing:
            - dict[str, np.ndarray]: Viewing Azimuth Angle (VAA) arrays keyed by wavelength.
            - dict[str, np.ndarray]: Viewing Zenith Angle (VZA) arrays keyed by wavelength.
            - list[list[str]]: 2D grid matrix of Solar Azimuth values.
            - list[list[str]]: 2D grid matrix of Solar Zenith values.

    Raises:
        FileNotFoundError: If the MTD XML file is missing.
        AssertionError: If an angle image file does not contain exactly 2 raster bands.
    """
    import s2a_angle_bands_mod as sentinel_angles

    mtd_dir = Path(path)
    mtd_files = [f for f in mtd_dir.iterdir() if 'MTD' in f.name and f.suffix == '.xml']
    if not mtd_files:
        raise FileNotFoundError(f"No MTD XML metadata file found in path: {path}")

    mtd_file = mtd_files[0]
    meta = load_sentinel_meta(mtd_file)

    # -------------------------------------------------------------------------
    # 1. Parse Solar Zenith and Azimuth Grids from XML
    # -------------------------------------------------------------------------
    solz_grid = meta['Geometric_Info']['Tile_Angles']['Sun_Angles_Grid']['Zenith']['Values_List']['VALUES']
    solar_zenith_grid = [solz.split(' ') for solz in solz_grid]

    sola_grid = meta['Geometric_Info']['Tile_Angles']['Sun_Angles_Grid']['Azimuth']['Values_List']['VALUES']
    solar_azimuth_grid = [sola.split(' ') for sola in sola_grid]

    # Resample sensor angles using module helper
    gsd, subsamp = sentinel_angles.main(str(mtd_file))

    # -------------------------------------------------------------------------
    # 2. Extract Band Angle Rasters from processed_angle_maps
    # -------------------------------------------------------------------------
    angle_folder = mtd_dir / "processed_angle_maps"
    image_files = [f for f in angle_folder.iterdir() if f.suffix == '.img']

    view_az_images: dict[str, np.ndarray] = {}
    view_zen_images: dict[str, np.ndarray] = {}

    for file_path in image_files:
        passband_id = file_path.stem.split('_')[-1].replace('B', '')

        with rasterio.open(file_path) as dataset:
            assert dataset.count == 2, f"{dataset.count} rasters found in {file_path.name}, expected 2"

            # Band 1: Azimuth, Band 2: Zenith (scaled by 100)
            azimuth_array = dataset.read(1) / 100.0
            zenith_array = dataset.read(2) / 100.0

            wavelength_key = MSI_WAVELENGTHS[passband_id]
            view_az_images[wavelength_key] = azimuth_array
            view_zen_images[wavelength_key] = zenith_array

    return view_az_images, view_zen_images, solar_azimuth_grid, solar_zenith_grid


def load_sentinel_meta(path: str | Path) -> dict[str, Any]:
    """Parses Sentinel-2 XML metadata file into a clean dictionary structure.

    Args:
        path (str | Path): File path to XML metadata.

    Returns:
        dict[str, Any]: Parsed metadata dictionary.
    """
    root = ET.parse(str(path)).getroot()
    xmlstr = ET.tostring(root, encoding='utf-8', method='xml')
    meta = dict(xmltodict.parse(xmlstr))

    # Prune top-level tile namespace key if present
    try:
        meta = meta['ns0:Level-1C_Tile_ID']
        meta = {k.split(':')[1]: v for (k, v) in meta.items()}
    except KeyError:
        print("Unable to prune upper tree level (ns0:Level-1C_Tile_ID not found)")

    return meta


if __name__ == '__main__':
    args = sys.argv[1:]
    if args:
        scene_id_arg = args[0]
        main(scene_id_arg)