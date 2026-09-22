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
    relative_azimuth = np.abs((viewing_azimuth - solar_azimuth) % 360 - 180) #np.abs(solar_azimuth - viewing_azimuth - 180) % 360

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
    img_chunk: xr.DataArray,
    img_mask: xr.DataArray,
    args: dict = None,
    n_outputs: int = 1,
    wvl_bands: Union[List[float], np.ndarray] = None,
    op_mode: str = "select",
    return_uncert: bool = True,
    scaler_mode: str = "invert",
    land_mask: bool = False,
    landmask_threshold: float = 0.0,
) -> xr.Dataset:
    """Apply MDN atmospheric correction to a localized image chunk block.

    Parameters
    ----------
    img_chunk : xr.DataArray
        A 3D localized chunk slice of stacked spectral and auxiliary features
        with dimensions ``('variable', 'y', 'x')``.
    img_mask : xr.DataArray
        A 2D spatial mask chunk with dimensions ``('y', 'x')``, where non-zero
        or True values represent valid water pixels.
    args : dict, optional
        Configuration dictionary containing pipeline parameters and model weights.
    n_outputs : int, default=1
        Number of output target spectral bands (matching length of `wvl_bands`).
    wvl_bands : Union[List[float], np.ndarray], optional
        Array or list of target output wavelengths.
    op_mode : str, default='select'
        Model execution mode. Options are ``'select'`` (returns single best-performing
        model predictions) or ``'ensemble'`` (returns predictions across all rounds).
    return_uncert : bool, default=True
        If True, returns upper and lower uncertainty bounds.
    scaler_mode : str, default='invert'
        Scaling inversion mode passed to the prediction pipeline.
    land_mask : bool, default=False
        Flag indicating whether land masking was enforced during setup.
    landmask_threshold : float, default=0.0
        Threshold employed for spatial masking operations.

    Returns
    -------
    xr.Dataset
        An xarray Dataset block containing remote sensing reflectance (``Rrs``)
        and optional uncertainty bounds (``uncertainty_low``, ``uncertainty_high``)
        with dimensions matching the template structure:
        - If ``op_mode == 'select'``: ``('wavelength', 'y', 'x')``
        - If ``op_mode == 'ensemble'``: ``('model', 'wavelength', 'y', 'x')``

    Notes
    -----
    This function operates on localized chunk data within an ``xr.map_blocks``
    pipeline. Spatial coordinates are omitted from the returned chunk Dataset
    to avoid coordinate validation overhead in Dask block processing.
    """
    # 1. Transpose chunk to (y, x, variable) for pixel extraction
    img_chunk_t = img_chunk.transpose("y", "x", "variable")
    img_chunk_np = np.asarray(img_chunk_t.data, dtype=np.float32)
    ny, nx, _ = img_chunk_np.shape

    # 2. Extract configuration metadata
    n_rounds = args.get("n_rounds", args["n_rounds"]) if isinstance(args, dict) else args.n_rounds
    n_models = 1 if op_mode == "select" else n_rounds
    no_data_val = (
        args.get("no_data_val", args.get("no_data", -9999.0))
        if isinstance(args, dict)
        else getattr(args, "no_data_val", getattr(args, "no_data", -9999.0))
    )

    # Clean input arrays in-place
    img_chunk_np[np.isinf(img_chunk_np)] = np.nan
    img_chunk_np[img_chunk_np < 0] = no_data_val

    # 3. Initialize output arrays matching template layout
    # Select mode: (wavelength, y, x) | Ensemble mode: (model, wavelength, y, x)
    if op_mode == "select":
        out_shape = (n_outputs, ny, nx)
        var_dims = ("wavelength", "y", "x")
    else:
        out_shape = (n_models, n_outputs, ny, nx)
        var_dims = ("model", "wavelength", "y", "x")

    img_preds = np.full(out_shape, no_data_val, dtype=np.float32)

    if return_uncert:
        img_uncert_lb = np.full(out_shape, no_data_val, dtype=np.float32)
        img_uncert_ub = np.full(out_shape, no_data_val, dtype=np.float32)

    # 4. Extract valid water spectra (where mask == 1 or True)
    mask_np = np.asarray(img_mask.data, dtype=bool)
    water_pixels = np.where(mask_np)
    water_spectra = img_chunk_np[mask_np]

    if water_spectra.size > 0:
        # Filter invalid/low-signal spectra
        maj_neg = ((water_spectra == no_data_val) | (water_spectra < 1e-6)).sum(axis=1) > 5
        water_spectra = water_spectra[~maj_neg]
        water_pixels = tuple(p[~maj_neg] for p in water_pixels)

        # Drop invalid NaN/masked elements
        water_final = np.ma.masked_invalid(water_spectra).reshape((-1, water_spectra.shape[-1]))
        valid_mask = ~np.any(water_final.mask, axis=1)

        water_final = water_final[valid_mask]
        water_pixels = tuple(p[valid_mask] for p in water_pixels)

        if isinstance(water_final, np.ma.MaskedArray):
            water_final = water_final.filled(no_data_val)

        water_final[water_final < 1e-6] = 1e-6

        # 5. Run inference if water pixels remain
        if water_final.size > 0:
            preds, uncert, _ = get_mdn_preds_uncertainties(
                test_x=water_final,
                args=args,
                op_mode=op_mode,
                scaler_mode=scaler_mode,
                uncert_mode="limits",
                progress_vis=False,
            )

            # Assign predictions back into spatial grid matching out_shape axis layout
            p_val = preds["pred"][:, :, :n_outputs]

            if op_mode == "select":
                # Reshape from (1, n_samples, n_outputs) -> (n_outputs, n_samples)
                p_val = np.squeeze(p_val, axis=0).T
                img_preds[:, water_pixels[0], water_pixels[1]] = p_val

                if return_uncert:
                    u_low = np.squeeze(uncert["low_lim"][:, :, :n_outputs], axis=0).T
                    u_high = np.squeeze(uncert["high_lim"][:, :, :n_outputs], axis=0).T
                    img_uncert_lb[:, water_pixels[0], water_pixels[1]] = u_low
                    img_uncert_ub[:, water_pixels[0], water_pixels[1]] = u_high
            else:
                # Ensemble mode: (n_models, n_outputs, n_samples)
                p_val = np.swapaxes(p_val, 1, 2)
                img_preds[:, :, water_pixels[0], water_pixels[1]] = p_val

                if return_uncert:
                    u_low = np.swapaxes(uncert["low_lim"][:, :, :n_outputs], 1, 2)
                    u_high = np.swapaxes(uncert["high_lim"][:, :, :n_outputs], 1, 2)
                    img_uncert_lb[:, :, water_pixels[0], water_pixels[1]] = u_low
                    img_uncert_ub[:, :, water_pixels[0], water_pixels[1]] = u_high

    # 6. Assemble output Dataset with consistent coordinate metadata
    data_vars = {"Rrs": (var_dims, img_preds)}

    if return_uncert:
        data_vars["uncertainty_low"] = (var_dims, img_uncert_lb)
        data_vars["uncertainty_high"] = (var_dims, img_uncert_ub)

    coords = {"wavelength": wvl_bands}
    if op_mode != "select":
        coords["model"] = np.arange(n_models)

    ds_chunk = xr.Dataset(data_vars=data_vars, coords=coords)

    # Cleanup memory
    del img_chunk_np, water_spectra
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
    """Applies the MDN atmospheric correction model to a Rayleigh-corrected reflectance dataset.

    Parameters:
    - ds_rrc (xr.Dataset): Input dataset containing Rayleigh-corrected reflectance.
    - sensor (str): Sensor name key (e.g., 'MSI', 'OLCI', 'OCI').
    - op_mode (str): Operational mode ('select' or 'ensemble'). Default is 'select'.
    - return_uncert (bool): Whether to return uncertainty estimates. Default is True.
    - uncert_mode (str): Mode for uncertainty estimation. Options are "limits" or "std". Default is "limits".
    - scaler_mode (str): Mode for scaling the output. Options are "invert" or "standard". Default is "invert".
    - land_mask (bool): Whether to apply a land mask. Default is True.
    - landmask_threshold (float): Threshold for land masking. Default is 0.0.
    - progress_vis (bool): Whether to visualize progress. Default is True.

    Returns:
    - xr.Dataset: Dataset containing corrected remote sensing reflectance (Rrs) as 3D DataArray ('wavelength', 'y', 'x')
                  and optional uncertainty arrays.
    """
    assert sensor in WAVELENGTHS_IN.keys(), (
        f"Unsupported sensor: {sensor}. Supported sensors are: {list(WAVELENGTHS_IN.keys())}"
    )

    # Fetch pipeline configurations
    kwargs = get_default_ac_pipeline_kwargs(sensor=sensor)
    args = get_args(**kwargs)

    # Handle sensor-specific viewing geometry aggregation
    if sensor == "MSI":
        ds_rrc = ds_rrc.assign(
            viewing_azimuth=ds_rrc["viewing_azimuth"].median(dim="wavelength"),
            viewing_zenith=ds_rrc["viewing_zenith"].median(dim="wavelength"),
        )

    # Extract configuration metadata
    wvl_in = WAVELENGTHS_IN[sensor]
    wvl_out = WAVELENGTHS_OUT[sensor]
    n_rounds = args.get("n_rounds", args["n_rounds"]) if isinstance(args, dict) else args.n_rounds
    n_outputs = len(wvl_out)
    n_models = 1 if op_mode == "select" else n_rounds
    no_data_val = args.get("no_data", args["no_data"]) if isinstance(args, dict) else args.no_data

    # Compute relative azimuth and scattering angles
    rel_az, scat_ang = subAngles(
        ds_rrc["viewing_azimuth"],
        ds_rrc["viewing_zenith"],
        ds_rrc["solar_azimuth"],
        ds_rrc["solar_zenith"],
    )
    ds_rrc["relative_azimuth"] = rel_az
    ds_rrc["scattering_angle"] = scat_ang

    # Define 2D auxiliary variables (non-wavelength dependent)
    aux_vars = ["viewing_zenith", "solar_zenith", "relative_azimuth", "scattering_angle", "water_vapor"]

    # Extract target bands and convert float wavelengths to clean string labels
    rho_rc_sub = ds_rrc["rho_rc"].sel(wavelength=wvl_in, method="nearest")
    rho_rc_da = rho_rc_sub.assign_coords(
        wavelength=[f"rho_rc_{w:.1f}" for w in rho_rc_sub.wavelength.values]
    ).rename({"wavelength": "variable"})

    # Extract auxiliary variables along 'variable' dimension
    aux_da = ds_rrc[aux_vars].to_array(dim="variable")

    # Stack spectral bands and aux variables into a unified 3D DataArray
    img_data = xr.concat([rho_rc_da, aux_da], dim="variable")

    # Apply land/water masking
    if land_mask:
        img_data = img_data.where(ds_rrc["water_mask"] == 1)

    # Cast data type
    img_data = img_data.astype("float32")

    # Preserve spatial chunking for y and x, keeping 'variable' unchunked
    y_chunk_size = img_data.chunksizes["y"][0] if img_data.chunks else "auto"
    x_chunk_size = img_data.chunksizes["x"][0] if img_data.chunks else "auto"

    img_data = img_data.chunk({
        "y": y_chunk_size, 
        "x": x_chunk_size, 
        "variable": -1
    })

    # =================================================================
    # CONSTRUCT TEMPLATE WITH ('wavelength', 'y', 'x') DIMENSIONS
    # =================================================================
    ny, nx = len(img_data.y), len(img_data.x)

    if op_mode == "select":
        # 3D output dimensions: (wavelength, y, x)
        var_dims = ("wavelength", "y", "x")
        dummy_shape = (n_outputs, ny, nx)
        template_coords = {
            "wavelength": wvl_out,
            "y": img_data.y,
            "x": img_data.x,
        }
    else:
        # 4D output dimensions for ensembling: (model, wavelength, y, x)
        var_dims = ("model", "wavelength", "y", "x")
        dummy_shape = (n_models, n_outputs, ny, nx)
        template_coords = {
            "model": np.arange(n_models),
            "wavelength": wvl_out,
            "y": img_data.y,
            "x": img_data.x,
        }

    template_vars = {
        "Rrs": (var_dims, np.empty(dummy_shape, dtype=np.float32))
    }

    if return_uncert:
        template_vars["uncertainty_low"] = (var_dims, np.empty(dummy_shape, dtype=np.float32))
        template_vars["uncertainty_high"] = (var_dims, np.empty(dummy_shape, dtype=np.float32))

    template_ds = xr.Dataset(data_vars=template_vars, coords=template_coords)

    # Preserve spatial coordinates to reattach after block mapping
    global_coords = {
        "wavelength": wvl_out,
        "y": img_data.coords["y"],
        "x": img_data.coords["x"],
    }
    for coord_key in ["spatial_ref", "latitude", "longitude"]:
        if coord_key in ds_rrc.variables:
            global_coords[coord_key] = ds_rrc[coord_key]

    # Strip coordinates from template so map_blocks avoids strict coordinate alignment overhead
    clean_template = template_ds.drop_vars(["y", "x", "spatial_ref", "latitude", "longitude"], errors="ignore")

    # Match chunk structure of input image
    clean_template = clean_template.chunk({
        "y": img_data.chunksizes["y"],
        "x": img_data.chunksizes["x"],
    })

    # Determine mask argument for block execution
    mask_arg = ds_rrc["water_mask"] if land_mask else xr.ones_like(img_data.isel(variable=0), dtype=bool)

    # Map chunk processing function across Dask blocks
    lazy_result_ds = xr.map_blocks(
        acmap_cube_mdn_chunk,
        img_data,
        args=[mask_arg],
        kwargs={
            "args": args,
            "n_outputs": n_outputs,
            "wvl_bands": wvl_out,
            "op_mode": op_mode,
            "land_mask": False,
            "landmask_threshold": landmask_threshold,
            "scaler_mode": scaler_mode,
            "return_uncert": return_uncert,
        },
        template=clean_template,
    )

    # Re-assign coordinate metadata
    final_lazy_ds = lazy_result_ds.assign_coords(global_coords)

    # Explicitly bind spatial dimensions for rioxarray
    final_lazy_ds = final_lazy_ds.rio.set_spatial_dims(x_dim="x", y_dim="y")

    # 3. Preserve CRS and Geotransform from input dataset
    if hasattr(ds_rrc, "rio") and ds_rrc.rio.crs:
        final_lazy_ds = final_lazy_ds.rio.write_crs(ds_rrc.rio.crs)
        if ds_subsampled.rio.transform():
            final_lazy_ds = final_lazy_ds.rio.write_transform(ds_rrc.rio.transform())

    # Compute graph execution
    if progress_vis:
        print("Computing MDN predictions across chunks...")
        with ProgressBar():
            final_ds = final_lazy_ds.compute(scheduler="single-threaded")
    else:
        final_ds = final_lazy_ds.compute(scheduler="single-threaded")

    # Calculate spatial extent metadata
    lon_min = float(ds_rrc["longitude"].min())
    lon_max = float(ds_rrc["longitude"].max())
    lat_min = float(ds_rrc["latitude"].min())
    lat_max = float(ds_rrc["latitude"].max())

    # Update Dataset metadata attributes
    final_ds.attrs.update({
        "title": "MDN Satellite Remote Sensing Reflectance (Rrs)",
        "sensor": sensor,
        "wavelengths": wvl_out,
        "target_products": [f"Rrs_{b}" for b in wvl_out],
        "operation_mode": op_mode,
        "land_mask_applied": str(land_mask),
        "land_mask_threshold": landmask_threshold,
        "scaler_mode": scaler_mode,
        "no_data_value": no_data_val,
        "extent": [lon_min, lon_max, lat_min, lat_max],
        "history": f"Created on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} using acmap_cube_mdn",
    })

    return final_ds