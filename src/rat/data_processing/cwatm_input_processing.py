"""
CWatM Input Processing for RAT.

This module provides functions for preparing meteorological forcing data
in the format required by CWatM from RAT's combined NetCDF data.

CWatM expects meteorological data in specific formats and with specific
variable names. This module handles the conversion from RAT's internal
data format to CWatM's expected input format.
"""

import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple, Union

from logging import getLogger
from rat.utils.logging import LOG_NAME, LOG_LEVEL, NOTIFICATION
from rat.utils.utils import create_directory

log = getLogger(LOG_NAME)
log.setLevel(LOG_LEVEL)


# Variable mapping from RAT/MetSim to CWatM
RAT_TO_CWATM_VARS = {
    # RAT/MetSim variable -> CWatM variable
    'prec': 'pr',  # Precipitation (mm/day)
    'Prec': 'pr',
    't_min': 'tasmin',  # Minimum temperature (K or C)
    't_max': 'tasmax',  # Maximum temperature (K or C)
    'wind': 'sfcWind',  # Wind speed (m/s)
    'shortwave': 'rsds',  # Shortwave radiation (W/m2)
    'longwave': 'rlds',  # Longwave radiation (W/m2)
    'vapor_pressure': 'huss',  # Specific humidity (kg/kg) - may need conversion
    'rel_humid': 'hurs',  # Relative humidity (%)
    'air_temp': 'tas',  # Average air temperature (K)
}

# CWatM expected units
CWATM_UNITS = {
    'pr': 'kg m-2 s-1',  # Precipitation flux
    'tas': 'K',  # Temperature in Kelvin
    'tasmin': 'K',
    'tasmax': 'K',
    'sfcWind': 'm s-1',
    'rsds': 'W m-2',
    'rlds': 'W m-2',
    'huss': '1',  # Dimensionless
    'hurs': '%',
}


def convert_metsim_to_cwatm(metsim_output_path: str,
                            output_path: str,
                            start_date: Optional[datetime.datetime] = None,
                            end_date: Optional[datetime.datetime] = None) -> str:
    """
    Convert MetSim output to CWatM input format.

    MetSim produces 6-hourly forcing data that needs to be aggregated to
    daily values for CWatM.

    Parameters
    ----------
    metsim_output_path : str
        Path to MetSim output NetCDF file(s)
    output_path : str
        Path where CWatM input will be saved
    start_date : datetime, optional
        Start date for subsetting data
    end_date : datetime, optional
        End date for subsetting data

    Returns
    -------
    str
        Path to the created CWatM forcing file
    """
    log.log(NOTIFICATION, "Converting MetSim output to CWatM input format")

    # Open MetSim output
    ds = xr.open_mfdataset(metsim_output_path, engine='netcdf4')

    # Subset by date if specified
    if start_date:
        ds = ds.sel(time=slice(start_date, end_date))

    # Aggregate to daily if sub-daily
    time_diff = pd.to_timedelta(ds.time.diff('time').values[0])
    if time_diff < pd.Timedelta('1D'):
        log.debug("Aggregating sub-daily MetSim output to daily for CWatM")
        ds_daily = aggregate_to_daily(ds)
    else:
        ds_daily = ds

    # Convert variables to CWatM names and units
    ds_cwatm = convert_variables_to_cwatm(ds_daily)

    # Set CRS and attributes
    ds_cwatm = set_cwatm_attributes(ds_cwatm)

    # Save
    output_file = os.path.join(output_path, 'cwatm_forcing.nc')
    ds_cwatm.to_netcdf(output_file)

    log.log(NOTIFICATION, f"CWatM forcing data saved to: {output_file}")
    ds.close()

    return output_file


def aggregate_to_daily(ds: xr.Dataset) -> xr.Dataset:
    """
    Aggregate sub-daily data to daily values.

    Different variables require different aggregation methods:
    - Precipitation: sum
    - Temperature: mean (or min/max)
    - Radiation: mean
    - Wind: mean
    - Humidity: mean

    Parameters
    ----------
    ds : xr.Dataset
        Sub-daily dataset

    Returns
    -------
    xr.Dataset
        Daily aggregated dataset
    """
    log.debug("Aggregating to daily values")

    # Define aggregation methods for different variables
    sum_vars = ['prec', 'Prec', 'precipitation', 'precip']
    min_vars = ['t_min', 'tmin', 'tasmin']
    max_vars = ['t_max', 'tmax', 'tasmax']
    # All other variables use mean

    # Separate variables by aggregation type
    ds_sum = ds[[v for v in ds.data_vars if any(sv in v.lower() for sv in sum_vars) and v in ds.data_vars]]
    ds_min = ds[[v for v in ds.data_vars if any(mv in v.lower() for mv in min_vars) and v in ds.data_vars]]
    ds_max = ds[[v for v in ds.data_vars if any(mv in v.lower() for mv in max_vars) and v in ds.data_vars]]

    mean_var_names = [v for v in ds.data_vars
                      if v not in ds_sum.data_vars
                      and v not in ds_min.data_vars
                      and v not in ds_max.data_vars]
    ds_mean = ds[mean_var_names]

    # Resample
    daily_datasets = []

    if len(ds_sum.data_vars) > 0:
        daily_datasets.append(ds_sum.resample(time='1D').sum())
    if len(ds_min.data_vars) > 0:
        daily_datasets.append(ds_min.resample(time='1D').min())
    if len(ds_max.data_vars) > 0:
        daily_datasets.append(ds_max.resample(time='1D').max())
    if len(ds_mean.data_vars) > 0:
        daily_datasets.append(ds_mean.resample(time='1D').mean())

    # Merge all
    if daily_datasets:
        ds_daily = xr.merge(daily_datasets)
    else:
        ds_daily = ds.resample(time='1D').mean()

    return ds_daily


def convert_variables_to_cwatm(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert variable names and units to CWatM conventions.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with RAT/MetSim variable names

    Returns
    -------
    xr.Dataset
        Dataset with CWatM variable names and units
    """
    log.debug("Converting variable names and units for CWatM")

    rename_dict = {}
    for rat_var, cwatm_var in RAT_TO_CWATM_VARS.items():
        if rat_var in ds.data_vars:
            rename_dict[rat_var] = cwatm_var

    ds_renamed = ds.rename(rename_dict)

    # Convert units where necessary
    for var in ds_renamed.data_vars:
        if var == 'pr':
            # Convert mm/day to kg m-2 s-1
            # 1 mm/day = 1 kg/m2/day = 1/86400 kg/m2/s
            if 'mm' in str(ds_renamed[var].attrs.get('units', '')).lower():
                ds_renamed[var] = ds_renamed[var] / 86400.0
                ds_renamed[var].attrs['units'] = CWATM_UNITS['pr']

        elif var in ['tas', 'tasmin', 'tasmax']:
            # Convert Celsius to Kelvin if needed
            if ds_renamed[var].values.mean() < 100:  # Likely Celsius
                log.debug(f"Converting {var} from Celsius to Kelvin")
                ds_renamed[var] = ds_renamed[var] + 273.15
            ds_renamed[var].attrs['units'] = CWATM_UNITS[var]

    return ds_renamed


def set_cwatm_attributes(ds: xr.Dataset) -> xr.Dataset:
    """
    Set NetCDF attributes required by CWatM.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset

    Returns
    -------
    xr.Dataset
        Dataset with CWatM-required attributes
    """
    # Global attributes
    ds.attrs['Conventions'] = 'CF-1.6'
    ds.attrs['institution'] = 'RAT - Reservoir Assessment Tool'
    ds.attrs['source'] = 'RAT meteorological forcing for CWatM'
    ds.attrs['history'] = f'Created by RAT on {datetime.datetime.now().isoformat()}'

    # Coordinate attributes
    if 'lat' in ds.coords:
        ds['lat'].attrs['units'] = 'degrees_north'
        ds['lat'].attrs['long_name'] = 'latitude'
        ds['lat'].attrs['standard_name'] = 'latitude'

    if 'lon' in ds.coords:
        ds['lon'].attrs['units'] = 'degrees_east'
        ds['lon'].attrs['long_name'] = 'longitude'
        ds['lon'].attrs['standard_name'] = 'longitude'

    if 'time' in ds.coords:
        ds['time'].attrs['standard_name'] = 'time'
        ds['time'].attrs['long_name'] = 'time'

    return ds


def prepare_cwatm_forcing_from_combined(combined_nc_path: str,
                                        output_dir: str,
                                        start_date: datetime.datetime,
                                        end_date: datetime.datetime,
                                        domain_nc_path: Optional[str] = None) -> str:
    """
    Prepare CWatM forcing data directly from RAT's combined NetCDF file.

    This function can be used when not running MetSim, directly converting
    RAT's combined meteorological data to CWatM format.

    Parameters
    ----------
    combined_nc_path : str
        Path to RAT's combined NetCDF file
    output_dir : str
        Directory where CWatM forcing will be saved
    start_date : datetime
        Start date for forcing data
    end_date : datetime
        End date for forcing data
    domain_nc_path : str, optional
        Path to domain/mask NetCDF file for spatial subsetting

    Returns
    -------
    str
        Path to the created CWatM forcing file
    """
    log.log(NOTIFICATION, "Preparing CWatM forcing from combined NetCDF")

    create_directory(output_dir, True)

    # Open combined data
    ds = xr.open_dataset(combined_nc_path)

    # Subset by time
    ds = ds.sel(time=slice(start_date, end_date))

    # Apply domain mask if provided
    if domain_nc_path and os.path.exists(domain_nc_path):
        domain = xr.open_dataset(domain_nc_path)
        if 'mask' in domain.data_vars:
            ds = ds.where(domain['mask'] > 0)
        domain.close()

    # Convert to CWatM format
    ds_cwatm = convert_variables_to_cwatm(ds)
    ds_cwatm = set_cwatm_attributes(ds_cwatm)

    # Save
    output_file = os.path.join(output_dir, 'cwatm_forcing.nc')
    ds_cwatm.to_netcdf(output_file)

    log.log(NOTIFICATION, f"CWatM forcing saved to: {output_file}")
    ds.close()

    return output_file


def generate_cwatm_state_and_inputs(forcings_startdate: datetime.datetime,
                                    forcings_enddate: datetime.datetime,
                                    combined_datapath: str,
                                    out_dir: str) -> Tuple[str, str]:
    """
    Generate CWatM state and input files from RAT's combined data.

    This is analogous to MetSim's generate_state_and_inputs function,
    preparing data in CWatM's expected format.

    Parameters
    ----------
    forcings_startdate : datetime
        Start date for forcing data
    forcings_enddate : datetime
        End date for forcing data
    combined_datapath : str
        Path to RAT's combined NetCDF file
    out_dir : str
        Output directory

    Returns
    -------
    tuple
        (state_file_path, input_file_path)
    """
    log.log(NOTIFICATION, "Generating CWatM state and input files")

    create_directory(out_dir, True)

    # For CWatM, state and input can be the same forcing file
    # as CWatM handles its own internal state
    forcing_file = prepare_cwatm_forcing_from_combined(
        combined_nc_path=combined_datapath,
        output_dir=out_dir,
        start_date=forcings_startdate,
        end_date=forcings_enddate
    )

    # CWatM state file path (will be created by CWatM if init_load is True)
    state_file = os.path.join(out_dir, 'cwatm_state.nc')

    return state_file, forcing_file


class CWatMForcingProcessor:
    """
    Class for processing meteorological forcing data for CWatM.

    This class provides methods for converting various input formats
    to CWatM's expected forcing format.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the forcing processor.

        Parameters
        ----------
        output_dir : str
            Directory where processed forcing files will be saved
        """
        self.output_dir = output_dir
        create_directory(output_dir, True)

    def process_metsim_output(self, metsim_path: str,
                               start_date: datetime.datetime,
                               end_date: datetime.datetime) -> str:
        """
        Process MetSim output for CWatM.

        Parameters
        ----------
        metsim_path : str
            Path to MetSim output
        start_date : datetime
            Start date
        end_date : datetime
            End date

        Returns
        -------
        str
            Path to processed forcing file
        """
        return convert_metsim_to_cwatm(
            metsim_output_path=metsim_path,
            output_path=self.output_dir,
            start_date=start_date,
            end_date=end_date
        )

    def process_combined_nc(self, combined_path: str,
                            start_date: datetime.datetime,
                            end_date: datetime.datetime) -> str:
        """
        Process RAT combined NetCDF for CWatM.

        Parameters
        ----------
        combined_path : str
            Path to combined NetCDF
        start_date : datetime
            Start date
        end_date : datetime
            End date

        Returns
        -------
        str
            Path to processed forcing file
        """
        return prepare_cwatm_forcing_from_combined(
            combined_nc_path=combined_path,
            output_dir=self.output_dir,
            start_date=start_date,
            end_date=end_date
        )
