# -*- coding: utf-8 -*-
"""
File Name:              user_utilities_ac.py
Description:            This code file will be used write functions that simplify the application of the AC code for the endusers. It 
                        will be used to abstract detail sections like argument setting etc completely away from the end users

Date Created:           August 31st, 2026
Author:                 Arun M Saranathan
Email:                  arun.saranathan@ssaihq.com/
                        fnu.arunmuralidharansaranathan@nasa.gov
"""
from __future__ import annotations
import numpy as np
import xarray as xr
import gc
import tensorflow as tf
from dask.diagnostics import ProgressBar
import pandas as pd


from .parameters import get_args
from .user_utilities import map_cube_mdn_chunk
from .utilities import get_mdn_preds_uncertainties


WAVELENGTHS_IN = {
    "OLI":[443, 482, 561, 655, 865],
    "MSI":[443, 490, 560, 665, 705, 740, 864]
}

WAVELENGTHS_OUT = {
    "OLI":[443, 482, 561, 655],
    "MSI":[443, 490, 560, 665, 705, 740]
}
OLI_WAVELENGTHS_OUT = [443, 482, 561, 655]
used_wavelengths_oli = ['443','482','561','655','865','1609','2201']
used_wavelengths_msi = ['443','490','560','665','705','740','780','833','864','944','1375','1609','2200']

POSITIONS = {
    "OLI":[used_wavelengths_oli.index(str(a)) for a in WAVELENGTHS_IN["OLI"]],
    "MSI":[used_wavelengths_msi.index(str(a)) for a in WAVELENGTHS_IN["MSI"]]
}

# Ancillary data used as inputs
ANCILLARY = ['senz', 'solz', 'relaz', 'scattang', 'water_vapor']


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


def subAngles(
    viewing_azimuth: xr.DataArray,
    viewing_zenith: xr.DataArray,
    solar_azimuth: xr.DataArray,
    solar_zenith: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate relative azimuth and scattering angles from solar and viewing geometry.

    Parameters
    ----------
    viewing_azimuth : xarray.DataArray
        Sensor viewing azimuth angles (degrees).
    viewing_zenith : xarray.DataArray
        Sensor viewing zenith angles (degrees).
    solar_azimuth : xarray.DataArray
        Solar azimuth angles (degrees).
    solar_zenith : xarray.DataArray
        Solar zenith angles (degrees).

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray]
        - relative_azimuth : Relative azimuth angle between sun and sensor (degrees).
        - scattering_angle : Solar radiation scattering angle (degrees).
    """
    # Relative azimuth angle calculation
    relative_azimuth = np.abs(solar_azimuth - viewing_azimuth - 180)

    # Convert angular inputs to radians for trigonometric functions
    solz_rad = np.radians(solar_zenith)
    senz_rad = np.radians(viewing_zenith)
    relaz_rad = np.radians(relative_azimuth)

    # Vectorized cosine of scattering angle computation
    cos_theta = -np.cos(solz_rad) * np.cos(senz_rad) - np.sin(solz_rad) * np.sin(
        senz_rad
    ) * np.cos(relaz_rad)

    # Clip values to avoid numerical domain errors in arccos near [-1.0, 1.0]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Convert scattering angle back to degrees
    scattering_angle = np.degrees(np.arccos(cos_theta))

    # Name data arrays for clean downstream assignment in xarray Datasets
    relative_azimuth.name = "relative_azimuth"
    scattering_angle.name = "scattering_angle"

    return relative_azimuth, scattering_angle


def get_default_ac_pipeline_kwargs(sensor):
    """
    Returns a dictionary of default keyword arguments for the atmospheric correction pipeline based on the sensor type.

    Parameters:
    - sensor (str): The sensor type, either 'OLI' or 'MSI'.

    Returns:
    - dict: A dictionary containing default keyword arguments for the specified sensor.
    """
    if sensor not in WAVELENGTHS_IN.keys():
        raise ValueError(f"Unsupported sensor: {sensor}. Supported sensors are: {list(WAVELENGTHS_IN.keys())}")

    if sensor == 'OLI':
        kwargs = {
            'verbose'   : False,
            'no_load'   : False,
            'sensor'    : 'OLI',
            'model_lbl' : 'oli_2024v5_B',
            'model_uid'  : '8c3a54b6e8cefe789cea4596bdb34328522ee2303da19aebf8081535f510a6c9',
            'model_loc' : r'F:\\Scratch\\',                                     ## this needs to be updated to realtive location
            'n_rounds'  : 10,
            'plot_loss' : False,
            'n_iter'    : 5000,
            'l2'        : 1e-3,
            'n_hidden'  : 50,
            'n_layers'  : 5,
            'batch'     : 2048,
            'lr'        : 1e-3,
            'silent'    : True
            }

    elif sensor == 'MSI':
        kwargs = {
            'verbose'   : False,
            'no_load'   : False,
            'sensor'    : 'MSI',
            'model_lbl' : 'msi_2024v5_A',
            'model_uid'  : '7ed555f7e7ecf61d18684dde626f3a99a3bd3882eb7cdae7a8abe8b2f2c16d0b',
            'model_loc' : r'F:\\Scratch\\',                                    ## this needs to be updated to realtive location                                         
            'n_rounds'  : 10,
            'plot_loss' : False,
            'n_iter'    : 5000,
            'l2'        : 1e-3,
            'n_hidden'  : 50,
            'n_layers'  : 5,
            'batch'     : 2048,
            'lr'        : 1e-3,
            'silent'    : True
            }
    return kwargs



def acmap_cube_mdn_chunk(
    img_chunk: np.ndarray,          # 1st positional: from main object (img_data block) -> shape (variable, y, x)
    img_mask: np.ndarray,           # 2nd positional: from args=[mask_arg] block         -> shape (y, x)
    # Keyword arguments passed via kwargs={...}
    args: dict=None,
    n_outputs: int = 1,
    wvl_bands: Union[List[float], np.ndarray] = None,
    op_mode: str = "select",
    return_uncert: bool = True,
    scaler_mode: str = "invert",
    land_mask: bool = False,
    landmask_threshold: float = 0.0,
) -> xr.Dataset:
    """
    Map a chunk of an image cube using MDN and return an xarray Dataset.

    Parameters
    ----------
    img_chunk : xr.DataArray
        An individual localized chunk slice of the input satellite image.
    args : Dict
        Configuration dictionary containing pipeline running options.
    target_products : str, default 'chl'
        A comma-separated string indicating target water constituents to calculate.
    n_outputs : int, default 1
        Calculated output array depth (matching the split-length of target_products).
    wvl_bands : Union[List[float], np.ndarray], default None
        Wavelength arrays configured for the target sensor.
    op_mode : str, default 'select'
        Operation mode parameter ('select' or 'ensemble').
    return_uncert : bool, default True
        If True, calculates uncertainty ranges dynamically.
    scaler_mode : str, default 'invert'
        Scaler mode config passed to the get_spectral_preds pipeline helper.
    land_mask : bool, default False
        If True, calculates a spatial land mask inside this chunk block.
    landmask_threshold : float, default 0.0
        Wavelength threshold utilized by the land masking algorithm.

    Returns
    -------
    xr.Dataset
        An xr.Dataset block representing processed output variables for this chunk area.
    """
    # 1. Transpose chunk and extract NumPy structures safely
    img_chunk_t = img_chunk.transpose("y", "x", "variable")
    img_chunk_np = np.asarray(img_chunk_t.data)  # Safely coerces the Dask chunk to a local NumPy array

    # 2. Extract metadata parameters safely
    n_rounds = (args.get("n_rounds", args["n_rounds"])
        if isinstance(args, dict)
        else args.n_rounds
    )
    n_models = 1 if op_mode == "select" else n_rounds
    no_data_val = (args.get("no_data_val", args.get("no_data", -9999.0))
        if isinstance(args, dict)
        else getattr(args, "no_data_val", getattr(args, "no_data", -9999.0))
    )

    # -------------------------------------------------------------------------
    # In-place Array Cleaning on img_chunk_np
    # -------------------------------------------------------------------------
    # Drop +/- Inf values up front
    img_chunk_np[np.isinf(img_chunk_np)] = np.nan

    # Replace negative values with no_data_val directly on img_chunk_np
    img_chunk_np[img_chunk_np < 0] = no_data_val

    # 3. Initialize blank output structures using the correct 'no_data' value
    img_preds = no_data_val * np.ones(
        (n_models, img_chunk_np.shape[0], img_chunk_np.shape[1], n_outputs),
        dtype=np.float32,
    )

    if return_uncert:
        img_uncert_lb = no_data_val * np.ones(( n_models, img_chunk_np.shape[0], img_chunk_np.shape[1], n_outputs,), dtype=np.float32,)
        img_uncert_ub = no_data_val * np.ones(( n_models, img_chunk_np.shape[0], img_chunk_np.shape[1], n_outputs,), dtype=np.float32,)

    #  Create water mask
    img_mask = 1. - np.asarray(img_mask.data, dtype=np.float32)  # Safely coerce the Dask chunk to a local NumPy array

    bool_mask = img_mask == 0
    water_pixels = np.where(bool_mask)
    water_spectra = img_chunk_np[bool_mask]

    # First filter: Remove spectra with majority invalid/don't-care values
    maj_neg = ((water_spectra == no_data_val) | (water_spectra < 1e-6)).sum(axis=1) > 5
    water_spectra = water_spectra[~maj_neg]
    water_pixels = tuple(p[~maj_neg] for p in water_pixels)

    # Second filter: Prepare spectra and drop any remaining NaNs
    water_final = np.ma.masked_invalid(water_spectra).reshape((-1, water_spectra.shape[-1]))
    valid_mask = ~np.any(water_final.mask, axis=1)

    water_final = water_final[valid_mask]
    water_pixels = tuple(p[valid_mask] for p in water_pixels)

    if isinstance(water_final, np.ma.MaskedArray):
        water_final = water_final.filled(no_data_val)

    # Also since the MDN expects positive values replace negatives with 1.e-6
    water_final[water_final < 1.e-6] = 1.e-6

    # Build dynamic output dict base
    data_vars = {"pred_rrs": (("model", "y", "x", "output"), img_preds)}

    # If we have valid water pixels, process them through the MDN model
    if water_final.size > 0:
        sensor_str = args["sensor"] if isinstance(args, dict) else args.sensor

        # get the predictions for this chunk
        preds, uncert, _ = get_mdn_preds_uncertainties(test_x=water_final, args=args, op_mode=op_mode, scaler_mode=scaler_mode, uncert_mode="limits", progress_vis=False)
        
        # Re-assign the predictions back into the spatial grid coordinate slices
        img_preds[:, water_pixels[0], water_pixels[1], :] = preds['pred'][:,:, 0:len(wvl_bands)]
        data_vars["pred_rrs"] = (("model", "y", "x", "output"), img_preds)

        if return_uncert:
            img_uncert_lb[:, water_pixels[0], water_pixels[1], :] = uncert["low_lim"][:,:, 0:len(wvl_bands)]
            img_uncert_ub[:, water_pixels[0], water_pixels[1], :] = uncert["high_lim"][:,:, 0:len(wvl_bands)]

    if return_uncert:
        data_vars["uncertainty_low"] = (
            ("model", "y", "x", "output"),
            img_uncert_lb,
        )
        data_vars["uncertainty_high"] = (
            ("model", "y", "x", "output"),
            img_uncert_ub,
        )

    # =================================================================
    # CRITICAL FIX: DO NOT DEFINE SPATIAL COORDINATES HERE AT ALL
    # =================================================================
    ds_chunk = xr.Dataset(
        data_vars=data_vars,
        coords={
            "model": np.arange(n_models),
            "output": np.arange(n_outputs),
        },
    )

    # Explicitly clear temporary arrays before returning
    del img_chunk_np, water_spectra, water_final
    gc.collect()
    tf.keras.backend.clear_session()

    return ds_chunk



def acmap_cube_mdn(
    ds_rrc: xr.Dataset,
    sensor: str,
    op_mode: str = "select",
    return_uncert: bool = True,
    uncert_mode: str = "limits",
    scaler_mode: str = "invert",
    land_mask: bool = True,
    landmask_threshold: float = 0.0,
    progress_vis: bool = True,
) -> xr.Dataset:
    """
    Applies the MDN atmospheric correction model to a Rayleigh-corrected reflectance dataset.

    Parameters:
    - ds_rrc (xr.Dataset): Input dataset containing Rayleigh-corrected reflectance.
    - return_uncert (bool): Whether to return uncertainty estimates. Default is True.
    - uncert_mode (str): Mode for uncertainty estimation. Options are "limits" or "std". Default is "limits".
    - scaler_mode (str): Mode for scaling the output. Options are "invert" or "standard". Default is "invert".
    - land_mask (bool): Whether to apply a land mask. Default is False.
    - landmask_threshold (float): Threshold for land masking. Default is 0.0.
    - progress_vis (bool): Whether to visualize progress. Default is True.

    Returns:
    - xr.Dataset: Dataset containing the corrected reflectance and optional uncertainty estimates.
    """

    assert sensor in WAVELENGTHS_IN.keys(), f"Unsupported sensor: {sensor}. Supported sensors are: {list(WAVELENGTHS_IN.keys())}"

    #  Fetch pipeline configurations
    kwargs = get_default_ac_pipeline_kwargs(sensor=sensor)
    args = get_args(**kwargs)

    #  Extract configuration metadata
    wvl_in = WAVELENGTHS_IN[sensor]
    wvl_out = WAVELENGTHS_OUT[sensor]
    n_rounds = args.get("n_rounds", args["n_rounds"]) if isinstance(args, dict) else args.n_rounds
    n_outputs = len(wvl_out)
    n_models = 1 if op_mode == "select" else n_rounds
    no_data_val = args.get("no_data", args["no_data"]) if isinstance(args, dict) else args.no_data

    # Pass the dataset DataArrays into the function
    rel_az, scat_ang = subAngles(ds_rrc["viewing_azimuth"], ds_rrc["viewing_zenith"], ds_rrc["solar_azimuth"],
                                     ds_rrc["solar_zenith"])

    # Assign the resulting DataArrays back to your Dataset
    ds_rrc["relative_azimuth"] = rel_az
    ds_rrc["scattering_angle"] = scat_ang

    # get the target input bands
    target_vars = [f"rho_rc_{b}" for b in wvl_in]  + ["viewing_zenith", "solar_zenith", "relative_azimuth", "scattering_angle", "water_vapor"]

    # Select and mask all target variables in a single vectorized step
    if land_mask:
        masked_ds = ds_rrc[target_vars].where(ds_rrc["water_mask"] == 1)
    else:
        masked_ds = ds_rrc[target_vars]

    # Convert the Dataset subset directly into a 3D DataArray (variable, y, x)
    img_data = masked_ds.to_array(dim="variable").astype("float32")
    
    # Extract existing 1D chunk sizes for y and x, and keep 'variable' unchunked (-1)
    y_chunk_size = img_data.chunksizes["y"][0] if img_data.chunks else "auto"
    x_chunk_size = img_data.chunksizes["x"][0] if img_data.chunks else "auto"

    img_data = img_data.chunk({
        "y": y_chunk_size, 
        "x": x_chunk_size, 
        "variable": -1
    })


    # Create dummy arrays for building the template
    dummy_preds = np.empty((n_models, len(img_data.y), len(img_data.x), n_outputs), dtype=np.float32) 
    template_vars = {
        "pred_rrs": (("model", "y", "x", "output"), dummy_preds)
    }


    if return_uncert:
        dummy_uncert = np.empty((n_models, len(img_data.y), len(img_data.x), n_outputs), dtype=np.float32)
        template_vars["uncertainty_low"] = (("model", "y", "x", "output"), dummy_uncert)
        template_vars["uncertainty_high"] = (("model", "y", "x", "output"), dummy_uncert)


    # 4. Define the initial full template dataset
    template_ds = xr.Dataset(
        data_vars=template_vars,
        coords={
            "model": np.arange(n_models),
            "y": img_data.y,
            "x": img_data.x,
            "output": np.arange(n_outputs),
        }
    )

    # =================================================================
    # CRITICAL FIX: PRESERVE, STRIP & CHUNK COORDINATES
    # =================================================================
    # Capture the global coordinates to safely bind back on later
    global_coords = {
        "y": img_data.coords["y"],
        "x": img_data.coords["x"]
    }
    if "spatial_ref" in ds_rrc.variables:
        global_coords["spatial_ref"] = ds_rrc["spatial_ref"]
    if "latitude" in ds_rrc.variables:
        global_coords["latitude"] = ds_rrc["latitude"]
    if "longitude" in ds_rrc.variables:
        global_coords["longitude"] = ds_rrc["longitude"]

    # Strip coordinates from template so xr.map_blocks doesn't perform strict coordinate validation
    coords_to_drop = ["y", "x", "spatial_ref", "latitude", "longitude"]
    clean_template = template_ds.drop_vars(coords_to_drop, errors="ignore")

    # Chunk the template matching the input image chunks so it has Dask arrays
    clean_template = clean_template.chunk({
        "y": img_data.chunksizes["y"],
        "x": img_data.chunksizes["x"]
    })

    # 5. Run mapping across dask-chunk blocks lazily
    # 1. Determine the mask to pass
    mask_arg = ds_rrc["water_mask"] if land_mask else xr.DataArray(np.ones_like(img_data[0, :, :]), dims=("y", "x"))

    # 2. Call map_blocks cleanly
    lazy_result_ds = xr.map_blocks(
        acmap_cube_mdn_chunk,             # func: target function
        img_data,                         # obj: primary xarray object
        args=[mask_arg],                  # args: positional arguments passed to func AFTER img_data
        kwargs={
            "args": args,
            #"target_products": products,
            "n_outputs": n_outputs,
            "wvl_bands": wvl_out,
            "op_mode": op_mode,
            "land_mask": False,
            "landmask_threshold": landmask_threshold,
            "scaler_mode": scaler_mode,
            "return_uncert": return_uncert
        },
        template=clean_template,
    )

    # Re-assign the true global coordinates back to the lazy dataset before computing
    final_lazy_ds = lazy_result_ds.assign_coords(global_coords)

    # Execute computation with visual feedback
    if progress_vis:
        print("Computing MDN predictions across chunks...")
        with ProgressBar():
            final_ds = final_lazy_ds.compute(scheduler="single-threaded")
    else:
        final_ds = final_lazy_ds.compute(scheduler="single-threaded")

    # EXTRACT SPATIAL EXTENT AS A TUPLE
    lon_min = float(ds_rrc["longitude"].min())
    lon_max = float(ds_rrc["longitude"].max())
    lat_min = float(ds_rrc["latitude"].min())
    lat_max = float(ds_rrc["latitude"].max())

     # ADD METADATA ATTRIBUTES HERE
    final_ds.attrs.update({
        "title": "MDN Satellite Product Predictions",
        "sensor": sensor,
        "wavelength": wvl_out,
        "target_products": [b"Rrs_{b}" for b in wvl_out],
        "operation_mode": op_mode,
        "land_mask_applied": str(land_mask),
        "land_mask_threshold": landmask_threshold,
        "scaler_mode": scaler_mode,
        "no_data_value": no_data_val,
        "extent": [lon_min, lon_max, lat_min, lat_max],
        "history": f"Created on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} using map_cube_mdn"
    })

    return final_ds
