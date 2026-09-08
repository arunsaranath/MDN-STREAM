# -*- coding: utf-8 -*-
import glob
import os
import xml.etree.ElementTree as ET

import geopandas as gpd
import numpy as np
import pandas as pd
import xmltodict
import xarray as xr
from numba import jit

import eoreader as eo
from affine import Affine
from eoreader.reader import Reader
from pyproj import CRS
from rasterio.features import rasterize
import scipy.odr as odr

# import cartopy.crs as ccrs

opj = os.path.join
BAND_NAMES = np.array([
    'B01',
    'B02',
    'B03',
    'B04',
    'B05',
    'B06',
    'B07',
    'B08',
    'B8A',
    'B09',
    'B10',
    'B11',
    'B12',
])
BAND_NAMES_EOREADER = np.array([
    'CA',
    'BLUE',
    'GREEN',
    'RED',
    'VRE_1',
    'VRE_2',
    'VRE_3',
    'NIR',
    'NARROW_NIR',
    'WV',
    'SWIR_CIRRUS',
    'SWIR_1',
    'SWIR_2',
])

BAND_ID = [b.replace('B', '') for b in BAND_NAMES]
NATIVE_RESOLUTION = [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20]
WAVELENGTH = np.array(
    [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]
)
BAND_WIDTH = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180]

INFO = (
    pd.DataFrame({
        'bandId': range(13),
        'ESA': BAND_NAMES,
        'EOREADER': BAND_NAMES_EOREADER,
        'Wavelength (nm)': WAVELENGTH,
        'Band width (nm)': BAND_WIDTH,
        'Resolution (m)': NATIVE_RESOLUTION,
    })
    .set_index('bandId')
    .T
)


class sentinel2_driver:

  def __init__(
      self,
      imageSAFE,
      band_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      band_tbp_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
      resolution=20,
      verbose=False,
      **kwargs,
  ):

    abspath = os.path.abspath(imageSAFE)
    dirroot, basename = os.path.split(abspath)
    self.verbose = verbose
    self.band_idx = band_idx
    self.band_tbp_idx = band_tbp_idx
    self.resolution = resolution
    self.INFO = INFO[band_idx]

    # -----------------------------------------------------
    # define prod and geom where data will be loaded
    # -----------------------------------------------------
    self.prod = xr.Dataset()
    self.geom = xr.Dataset()

    # --------------------------------
    # define interpolation parameters
    # --------------------------------
    # tile for 10m resolution: width,height = 10980,10980
    # tile for 20m resolution: width,height = 5490,5490
    # tile for 60m resolution: width,height = 1830,1830
    if resolution == 10:
      self.width, self.height = 10980, 10980
    elif resolution == 20:
      self.width, self.height = 5490, 5490
    elif resolution == 30:
      self.width, self.height = 3660, 3660
    elif resolution == 60:
      self.width, self.height = 1830, 1830

    self.xml_granule = glob.glob(
        opj(imageSAFE, 'GRANULE', '*', 'MTD_TL.xml')
    )[0]
    self.xml_file = glob.glob(opj(imageSAFE, 'MTD*.xml'))[0]

    # Native XML parsing replacing GDAL GetMetadata()
    with open(self.xml_file, 'r', encoding='utf-8') as f:
      xml_content = f.read()

    parsed_xml = xmltodict.parse(xml_content)
    self.metadata2 = parsed_xml
    self.metadata = self._flatten_xml_dict(parsed_xml)

    __ = []
    # Locate Solar Irradiance List dynamically regardless of L1C wrapper tags
    product_char = self._get_nested_key(
        parsed_xml, 'Product_Image_Characteristics'
    )
    solar_list = product_char['Reflectance_Conversion']['Solar_Irradiance_List'][
        'SOLAR_IRRADIANCE'
    ]

    for _ in solar_list:
      __.append([int(_['@bandId']), float(_['#text'])])
    self.solar_irradiance = np.array(__)

    # Spectral Response Functions
    SRFs = []
    wl_hr = np.arange(400, 2350)
    spectral_list = product_char['Spectral_Information_List'][
        'Spectral_Information'
    ]

    for _ in spectral_list:
      bandID = int(_['@bandId'])
      if not self.band_idx.__contains__(bandID):
        continue

      wl_min, wl_max = float(_['Wavelength']['MIN']['#text']), float(
          _['Wavelength']['MAX']['#text']
      )
      step = float(_['Spectral_Response']['STEP']['#text'])
      wl = np.arange(wl_min, wl_max + step, step)
      SRF = np.asarray(
          _['Spectral_Response']['VALUES'].split(), dtype=np.float32
      )
      SRFs.append(
          xr.DataArray(SRF, coords=dict(wl_hr=wl), name='SRF')
          .interp(wl_hr=wl_hr)
          .assign_coords(dict(wl=WAVELENGTH[bandID]))
      )
    self.SRFs = xr.concat(SRFs, dim='wl')
    self.SRFs.attrs['description'] = 'Spectral response function of each band'

    # Open instance of eoreader
    reader = Reader()

    # Open the product
    reader = reader.open(imageSAFE, remove_tmp=True, **kwargs)
    self.reader = reader
    self.processing_baseline = reader._processing_baseline
    self.datetime = reader.datetime

    # save geographic data
    self.extent = reader.extent()
    self.bounds = self.extent.bounds
    minx, miny, maxx, maxy = self.bounds.values[0]
    self.crs = self.reader.crs()
    self.epsg = self.extent.crs.to_epsg()
    self.transform = Affine(resolution, 0.0, minx, 0.0, -resolution, maxy)

    # -------------------------
    # interpolation
    # -------------------------
    self.indexing = 'xy'
    self.new_x = np.linspace(minx, maxx, self.width)
    self.new_y = np.linspace(miny, maxy, self.height)[::-1]

    # ---------------------------------
    # getting appropriate version of mask driver
    # ---------------------------------
    if self.processing_baseline < 4:
      self._open_mask = reader._open_mask_lt_4_0
      self._open_mask_failsafe = reader._open_mask_gt_4_0
    else:
      self._open_mask = reader._open_mask_gt_4_0

  @staticmethod
  def _get_nested_key(d, target_key):
    """Recursively search for a key in nested dictionaries."""
    if isinstance(d, dict):
      for k, v in d.items():
        clean_k = k.split(':')[-1]
        if clean_k == target_key:
          return v
        res = sentinel2_driver._get_nested_key(v, target_key)
        if res is not None:
          return res
    elif isinstance(d, list):
      for item in d:
        res = sentinel2_driver._get_nested_key(item, target_key)
        if res is not None:
          return res
    return None

  @staticmethod
  def _flatten_xml_dict(d, parent_key='', sep='_'):
    """Helper to mimic GDAL metadata dictionary format from XML."""
    items = []
    if isinstance(d, dict):
      for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        clean_key = new_key.split(':')[-1]
        items.extend(
            sentinel2_driver._flatten_xml_dict(v, clean_key, sep=sep).items()
        )
    elif isinstance(d, list):
      for i, item in enumerate(d):
        items.extend(
            sentinel2_driver._flatten_xml_dict(
                item, f'{parent_key}_{i}', sep=sep
            ).items()
        )
    else:
      items.append((parent_key, str(d)))
    return dict(items)

  def load_product(self, add_time=False, **kwargs):

    self.load_bands(add_time=False, **kwargs)
    self.load_geom()
    self.prod = xr.merge([self.prod, self.geom])

    # add native metadata
    for item in self.metadata:
      self.prod.attrs[item] = self.metadata[item]

    # Extract satellite metadata safely
    product_uri = self._get_nested_key(self.metadata2, 'PRODUCT_URI')
    if product_uri:
      self.prod.attrs['satellite'] = product_uri.split('_')[0]
    else:
      self.prod.attrs['satellite'] = 'SENTINEL-2'

    self.prod.attrs['solar_irradiance'] = self.solar_irradiance[:, 1]
    self.prod.attrs['solar_irradiance_unit'] = 'W/m²/µm'

    start_time = self._get_nested_key(
        self.metadata2, 'DATATAKE_1_DATATAKE_SENSING_START'
    )
    if start_time:
      self.prod.attrs['DATATAKE_1_DATATAKE_SENSING_START'] = start_time
      self.prod.attrs['acquisition_date'] = start_time

  def load_bands(self, add_time=False, **kwargs):

    # ----------------------------------
    # getting bands
    # ----------------------------------
    bands = self.reader.stack(
        list(BAND_NAMES_EOREADER[self.band_idx]),
        resolution=self.resolution,
        **kwargs,
    )
    if 'z' in bands.coords:
      bands = bands.rename({'z': 'bands'})

    # ----------------------------------
    # setting up coordinates and dimensions
    # ----------------------------------
    temp = bands.assign_coords(
        wl=('bands', self.INFO.loc['Wavelength (nm)'])
    ).swap_dims({'bands': 'wl'})

    for coordinate in ['band', 'bands', 'variable']:
      try:
        temp = temp.drop({coordinate})
      except Exception:
        pass
    self.prod = temp
    self.prod = self.prod.assign_coords(
        bandID=('wl', self.INFO.loc['ESA'].values)
    )
    self.prod = self.prod.to_dataset(name='bands', promote_attrs=True)
    self.prod.attrs['wl_to_process'] = WAVELENGTH[self.band_tbp_idx]

    # add spectral response function
    self.prod = xr.merge(
        [self.prod, self.SRFs.sel(wl=self.prod.wl.values)]
    ).drop_vars('bandID')

    # compute central wavelengths
    wl_true = []
    for wl_, srf in self.prod.SRF.groupby('wl'):
      srf = srf.dropna('wl_hr')
      wl_true.append(
          (srf.wl_hr * srf).integrate('wl_hr') / srf.integrate('wl_hr')
      )
    wl_true = xr.concat(wl_true, dim='wl')
    wl_true.name = 'wl_true'
    self.prod = xr.merge([self.prod, wl_true])

    # add time
    if add_time:
      self.prod = self.prod.assign_coords(time=self.datetime).expand_dims(
          'time'
      )

  @staticmethod
  def parse_angular_grid_node(node):
    '''Internal parsing function for angular grids'''
    values = []
    for c in node.find('Values_List'):
      values.append(np.array([float(t) for t in c.text.split()]))
    values_array = np.stack(values)
    return values_array

  @staticmethod
  def set_crs(arr, crs):
    arr.rio.set_crs(crs, inplace=True)
    arr.rio.write_crs(inplace=True)

  def get_raw_angles(self):

    minx, miny, maxx, maxy = self.bounds.values[0]
    with open(self.xml_granule) as xml_file:
      tree = ET.parse(xml_file)
      root = tree.getroot()

    raw_sza = self.parse_angular_grid_node(
        root.find('.//Tile_Angles/Sun_Angles_Grid/Zenith')
    )
    raw_sazi = self.parse_angular_grid_node(
        root.find('.//Tile_Angles/Sun_Angles_Grid/Azimuth')
    )

    Nx, Ny = raw_sza.shape

    xang = np.linspace(minx, maxx, Nx)
    yang = np.linspace(miny, maxy, Ny)[::-1]

    raw_sun_ang = xr.Dataset(
        data_vars=dict(
            sza=(['y', 'x'], raw_sza), sazi=(['y', 'x'], raw_sazi)
        ),
        coords=dict(x=xang, y=yang),
    )
    self.set_crs(raw_sun_ang, self.crs)
    self.raw_sun_ang = raw_sun_ang

    # ---------------------------------
    # getting viewing geometry datacube
    # ---------------------------------
    bandIds, detectorIds = [], []
    for angleID in root.findall(
        './/Tile_Angles/Viewing_Incidence_Angles_Grids'
    ):
      bandIds.append(int(angleID.attrib['bandId']))
      detectorIds.append(int(angleID.attrib['detectorId']))
    Nband, Ndetector = np.max(bandIds) + 1, np.max(detectorIds) + 1

    vza = np.full((Nband, Ndetector, Nx, Ny), np.nan, dtype=float)
    vazi = np.full((Nband, Ndetector, Nx, Ny), np.nan, dtype=float)

    for angleID in root.findall(
        './/Tile_Angles/Viewing_Incidence_Angles_Grids'
    ):
      iband = int(angleID.attrib['bandId'])
      idetector = int(angleID.attrib['detectorId'])
      vza[iband, idetector] = self.parse_angular_grid_node(
          angleID.find('Zenith')
      )
      vazi[iband, idetector] = self.parse_angular_grid_node(
          angleID.find('Azimuth')
      )

    raw_view_ang = xr.Dataset(
        data_vars=dict(
            vza=(['bandId', 'detectorId', 'y', 'x'], vza),
            vazi=(['bandId', 'detectorId', 'y', 'x'], vazi),
        ),
        coords=dict(
            bandId=range(Nband), detectorId=range(Ndetector), x=xang, y=yang
        ),
    )
    self.set_crs(raw_view_ang, self.crs)

    raw_view_ang = raw_view_ang.dropna('detectorId', how='all')
    self.raw_view_ang = raw_view_ang
    self.detector_num = len(raw_view_ang.detectorId)

    return

  def load_geom(self, method='linear'):

    self.get_raw_angles()
    self.get_all_band_angles(method=method)

  @staticmethod
  def linfit(beta, x):
    return beta[0] * x[0] + beta[1] * x[1] + beta[2]

  @staticmethod
  @jit(nopython=True)
  def lin2D(arr, x, y, mask, betas, detector_offset=0, scale_factor=100):

    Nx, Ny = mask.shape

    for ii in range(Nx):
      for jj in range(Ny):
        detect = mask[ii, jj]
        if detect == 0:
          continue
        beta = betas[detect - detector_offset]
        val = beta[0] * x[jj] + beta[1] * y[ii] + beta[2]
        arr[ii, jj] = val * scale_factor

  def data_fitting(self, x0, y0, arr):

    xgrid, ygrid = np.meshgrid(x0, y0, indexing=self.indexing)

    values = arr.values.flatten()
    x_ = xgrid.flatten()
    y_ = ygrid.flatten()

    idx = ~np.isnan(values)
    values = values[idx]
    points = np.empty((2, len(values)))
    points[0] = x_[idx]
    points[1] = y_[idx]

    mean = np.nanmean(values)
    linear = odr.Model(self.linfit)
    data = odr.Data(points, values)
    beta0 = [0, 0, mean]

    fit = odr.ODR(data, linear, beta0=beta0)
    resfit = fit.run()

    if self.verbose:
      resfit.pprint()

    return resfit.beta

  def get_detector_mask(
      self, bandId=0, resolution=20, detector_mask_name='DETFOO'
  ):

    if self.processing_baseline < 4:
      try:
        mask_df = self._open_mask(detector_mask_name, BAND_ID[bandId])
      except Exception:
        mask_df = self._open_mask_failsafe(
            detector_mask_name, BAND_ID[bandId]
        )
      detector_num = mask_df.gml_id.str.split('-', expand=True).values[:, 2]
      poly_shp = [
          [geom, int(value)]
          for geom, value in zip(mask_df.geometry, detector_num)
      ]

      mask = rasterize(
          shapes=poly_shp,
          out_shape=(self.height, self.width),
          transform=self.transform,
      )
    else:
      mask = self._open_mask(
          detector_mask_name,
          BAND_ID[bandId],
          resolution=resolution,
          pixel_size=resolution,
      ).astype(np.int8)
      mask = mask.squeeze()
    return np.array(mask)

  def get_band_angle_as_numpy(
      self,
      xarr,
      bandId=0,
      resolution=20,
      detector_mask_name='DETFOO',
      compress=False,
  ):

    detector_offset = xarr.detectorId.values.min()
    mask = self.get_detector_mask(bandId=bandId, resolution=resolution)

    x, y = self.new_x, self.new_y
    betas = np.full((self.detector_num, 3), np.nan)
    xarr_ = xarr.sel(bandId=bandId)
    for id in range(self.detector_num):
      arr = (
          xarr_.isel(detectorId=id).dropna('y', how='all').dropna('x', how='all')
      )
      x0, y0 = arr.x.values, arr.y.values
      betas[id, :] = self.data_fitting(x0, y0, arr)

    if compress:
      new_arr = np.full((self.width, self.height), np.nan, dtype=np.int16)
      self.lin2D(
          new_arr,
          x,
          y,
          mask,
          betas,
          detector_offset=detector_offset,
          scale_factor=100,
      )
    else:
      new_arr = np.full((self.width, self.height), np.nan, dtype=np.float32)
      self.lin2D(
          new_arr,
          x,
          y,
          mask,
          betas,
          detector_offset=detector_offset,
          scale_factor=1,
      )

    del mask
    return new_arr

  @staticmethod
  def scat_angle(sza, vza, azi):
    sza = np.radians(sza)
    vza = np.radians(vza)
    azi = np.radians(azi)
    ang = -np.cos(sza) * np.cos(vza) - np.sin(sza) * np.sin(vza) * np.cos(azi)
    ang = np.arccos(ang)
    return np.degrees(ang)

  def get_all_band_angles(self, method='linear'):

    new_x, new_y = self.new_x, self.new_y
    band_idx = self.band_idx

    new_sun_ang = self.raw_sun_ang.interp(x=new_x, y=new_y, method=method)

    raw_vza = self.raw_view_ang.vza
    raw_vazi = self.raw_view_ang.vazi

    new_vza, new_vazi = [], []
    for ibandId, bandId in enumerate(band_idx):
      if self.verbose:
        print('Band number ' + str(bandId) + ' is being loaded')
      new_vza.append(
          self.get_band_angle_as_numpy(
              raw_vza, bandId=bandId, resolution=self.resolution
          )
      )
      new_vazi.append(
          self.get_band_angle_as_numpy(
              raw_vazi, bandId=bandId, resolution=self.resolution
          )
      )
    raa = (np.array(new_vazi) - new_sun_ang.sazi.values) % 360

    self.geom['vza'] = xr.DataArray(np.array(new_vza), dims=['wl', 'y', 'x'])
    self.geom['raa'] = xr.DataArray(raa, dims=['wl', 'y', 'x'])
    self.geom['sza'] = xr.DataArray(new_sun_ang.sza.values, dims=['y', 'x'])
    self.geom['saa'] = xr.DataArray(new_sun_ang.sazi.values, dims=['y', 'x'])
    self.geom['vaa'] = xr.DataArray(np.array(new_vazi), dims=['wl', 'y', 'x'])

    del new_vza, new_vazi, raa, new_sun_ang

    return