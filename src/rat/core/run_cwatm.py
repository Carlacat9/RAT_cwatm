"""
CWatM (Community Water Model) integration for RAT.

This module provides the CWatMRunner class that implements the HydrologicalModel
interface for running CWatM as the hydrological model in RAT's pipeline.

CWatM is a community-driven global hydrological model developed by IIASA that
explicitly models human water management including reservoirs, irrigation,
and water withdrawals.
"""

import os
import csv
import math
import time
import datetime
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Optional, Dict, Any
from functools import partial

from logging import getLogger
from rat.utils.logging import LOG_NAME, LOG_LEVEL, NOTIFICATION
from rat.utils.utils import create_directory
from rat.utils.run_command import run_command
from rat.core.hydro_model import HydrologicalModel, ModelFactory

log = getLogger(LOG_NAME)
log.setLevel(LOG_LEVEL)


class CWatMRunner(HydrologicalModel):
    """
    CWatM hydrological model runner for RAT.

    This class implements the HydrologicalModel interface for CWatM,
    allowing RAT to use CWatM for inflow estimation instead of VIC.

    CWatM provides several advantages over VIC for reservoir-dominated basins:
    - Explicit reservoir operation modeling
    - Human water management (irrigation, groundwater extraction)
    - Global coverage with consistent parameterization

    Attributes
    ----------
    MODEL_NAME : str
        Name identifier for the model
    MODEL_VERSION : str
        CWatM version being used
    """

    MODEL_NAME = "cwatm"
    MODEL_VERSION = "1.0"

    # CWatM-specific variable mapping to RAT standard names
    CWATM_VAR_MAPPING = {
        'runoff': 'runoff',  # CWatM runoff variable
        'baseflow': 'baseflow',  # CWatM baseflow variable
        'discharge': 'discharge',  # CWatM discharge (routed)
        'evaporation': 'totalET',  # Total evapotranspiration
        'net_radiation': 'Rn',  # Net radiation
        'air_temperature': 'Tavg',  # Average temperature
        'precipitation': 'Precipitation',  # Precipitation
    }

    def __init__(self, config: Dict[str, Any], basin_name: str) -> None:
        """
        Initialize the CWatM runner.

        Parameters
        ----------
        config : dict
            RAT configuration dictionary containing CWATM section
        basin_name : str
            Name of the basin being processed
        """
        super().__init__(config, basin_name)

        # Get CWatM-specific configuration
        cwatm_config = config.get('CWATM', {})

        self.cwatm_env = cwatm_config.get('cwatm_env', None)
        self.cwatm_path = cwatm_config.get('cwatm_path', None)
        self.settings_file = cwatm_config.get('cwatm_settings_file', None)
        self.conda_hook = cwatm_config.get('conda_hook', None)

        # CWatM parameter files
        self.meteo_netcdf_dir = cwatm_config.get('meteo_netcdf_dir', None)
        self.mask_map = cwatm_config.get('mask_map', None)
        self.init_load = cwatm_config.get('init_load', False)

        # Output paths (will be set during setup)
        self.output_dir: Optional[str] = None
        self.cwatm_output_file: Optional[str] = None
        self.forcing_path: Optional[str] = None

        # Routing input directory
        self.rout_input_dir: Optional[str] = None

        # State management
        self.init_state_file: Optional[str] = None
        self.save_state: bool = True

    def setup(self, start_date: datetime.datetime, end_date: datetime.datetime,
              forcing_path: str, output_path: str, **kwargs) -> None:
        """
        Set up CWatM for a simulation run.

        Parameters
        ----------
        start_date : datetime.datetime
            Start date of simulation
        end_date : datetime.datetime
            End date of simulation
        forcing_path : str
            Path to meteorological forcing data (NetCDF format)
        output_path : str
            Path where CWatM outputs will be saved
        **kwargs : dict
            Additional parameters:
            - init_state: Path to initial state file
            - save_state: Whether to save state at end of run
            - settings_file: Override settings file path
        """
        self.start_date = start_date
        self.end_date = end_date
        self.forcing_path = forcing_path
        self.output_dir = output_path
        self.model_output_path = Path(output_path)

        # Handle optional parameters
        self.init_state_file = kwargs.get('init_state', None)
        self.save_state = kwargs.get('save_state', True)

        if 'settings_file' in kwargs:
            self.settings_file = kwargs['settings_file']

        # Create output directory
        create_directory(output_path, True)

        # Set expected output file path
        self.cwatm_output_file = os.path.join(
            output_path,
            f"cwatm_output_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc"
        )

        log.log(NOTIFICATION, f"CWatM setup complete for {self.basin_name}")
        log.debug(f"Start date: {start_date}")
        log.debug(f"End date: {end_date}")
        log.debug(f"Forcing path: {forcing_path}")
        log.debug(f"Output path: {output_path}")

    def run(self, num_processors: int = 1) -> int:
        """
        Execute CWatM model.

        Parameters
        ----------
        num_processors : int
            Number of processors for parallel execution

        Returns
        -------
        int
            Return code (0 for success, non-zero for failure)
        """
        log.log(NOTIFICATION, f"Running CWatM Model for basin: {self.basin_name}")

        if self.cwatm_path is None:
            log.error("CWatM path not configured")
            return 1

        if self.settings_file is None:
            log.error("CWatM settings file not configured")
            return 1

        # Build the CWatM command
        if self.cwatm_env:
            # Running with conda environment
            if self.conda_hook:
                cmd = (
                    f"source {self.conda_hook} && "
                    f"conda activate {self.cwatm_env} && "
                    f"python {self.cwatm_path} {self.settings_file}"
                )
            else:
                cmd = (
                    f'eval "$(conda shell.bash hook)" && '
                    f"conda activate {self.cwatm_env} && "
                    f"python {self.cwatm_path} {self.settings_file}"
                )
        else:
            # Running without conda (assumes CWatM is in PATH)
            cmd = f"python {self.cwatm_path} {self.settings_file}"

        log.debug(f"Running command: {cmd}")

        try:
            ret_code = run_command(cmd, shell=True, executable="/bin/bash")
            if ret_code != 0:
                log.error(f"CWatM execution failed with return code: {ret_code}")
            else:
                log.log(NOTIFICATION, "CWatM execution completed successfully")
            return ret_code
        except Exception as e:
            log.exception(f"Error running CWatM: {e}")
            return 1

    def get_runoff_output(self) -> str:
        """
        Get the path to the CWatM runoff output file.

        Returns
        -------
        str
            Path to the NetCDF file containing runoff and discharge data
        """
        if self.cwatm_output_file and os.path.exists(self.cwatm_output_file):
            return self.cwatm_output_file

        # Try to find the output file in the output directory
        if self.output_dir:
            # CWatM output naming convention
            potential_files = [
                os.path.join(self.output_dir, 'discharge_dailyTot_output.nc'),
                os.path.join(self.output_dir, 'runoff_dailyTot_output.nc'),
                os.path.join(self.output_dir, 'cwatm_output.nc'),
            ]
            for f in potential_files:
                if os.path.exists(f):
                    return f

        raise FileNotFoundError("CWatM runoff output file not found")

    def get_evaporation_variables(self) -> str:
        """
        Get the path to the file containing evaporation-related variables.

        CWatM outputs these variables in the main output file or in
        separate files depending on configuration.

        Returns
        -------
        str
            Path to the NetCDF file containing evaporation variables
        """
        # For CWatM, evaporation variables may be in different files
        # Check main output first
        if self.cwatm_output_file and os.path.exists(self.cwatm_output_file):
            return self.cwatm_output_file

        # Check for specific evaporation output file
        if self.output_dir:
            evap_file = os.path.join(self.output_dir, 'totalET_dailyTot_output.nc')
            if os.path.exists(evap_file):
                return evap_file

        return self.get_runoff_output()

    def prepare_routing_input(self, rout_input_dir: str,
                               rout_input_state_file: Optional[str] = None,
                               use_rout_state: bool = False) -> None:
        """
        Prepare input files for RAT's routing model from CWatM outputs.

        This method converts CWatM's gridded runoff output to the format
        expected by RAT's routing model (fluxes_*.csv files per grid cell).

        Parameters
        ----------
        rout_input_dir : str
            Directory where routing input files will be saved
        rout_input_state_file : str, optional
            Path to routing state file for continuous runs
        use_rout_state : bool
            Whether to use existing routing state
        """
        self.rout_input_dir = rout_input_dir
        log.log(NOTIFICATION, "Preparing routing input from CWatM outputs")

        create_directory(rout_input_dir, True)

        # Open CWatM output
        cwatm_output = self.get_runoff_output()
        log.debug(f"Reading CWatM output from: {cwatm_output}")

        ds = xr.open_dataset(cwatm_output)

        # Get runoff and baseflow (or discharge if already routed)
        # Variable names may vary based on CWatM configuration
        runoff_var = self._find_variable(ds, ['runoff', 'Runoff', 'SurfaceRunoff'])
        baseflow_var = self._find_variable(ds, ['baseflow', 'Baseflow', 'GWRunoff'])

        if runoff_var is None and baseflow_var is None:
            # Try using total discharge if runoff components not available
            discharge_var = self._find_variable(ds, ['discharge', 'Discharge', 'dis'])
            if discharge_var is not None:
                log.warning("Using total discharge instead of runoff components")
                runoff_var = discharge_var
                baseflow_var = None

        if runoff_var is None:
            raise ValueError("Could not find runoff variable in CWatM output")

        # Handle routing state file merging if needed
        if use_rout_state and rout_input_state_file and os.path.isfile(rout_input_state_file):
            log.debug(f"Merging with routing state file: {rout_input_state_file}")
            ds = self._merge_with_state(ds, rout_input_state_file)

        # Create routing input files
        self._create_routing_files(ds, runoff_var, baseflow_var, rout_input_dir)

        ds.close()
        log.log(NOTIFICATION, "Finished preparing routing input")

    def _find_variable(self, ds: xr.Dataset, var_names: list) -> Optional[str]:
        """Find a variable in the dataset from a list of possible names."""
        for var in var_names:
            if var in ds.data_vars:
                return var
        return None

    def _merge_with_state(self, ds: xr.Dataset, state_file: str) -> xr.Dataset:
        """Merge current output with previous state file."""
        try:
            state_ds = xr.open_dataset(state_file)
            # Get the time overlap
            first_new_time = ds.time[0]

            # Keep only state data before the new data
            state_ds_trimmed = state_ds.sel(time=slice(None, first_new_time - np.timedelta64(1, 'D')))

            # Concatenate
            merged = xr.concat([state_ds_trimmed, ds], dim='time')
            state_ds.close()
            return merged
        except Exception as e:
            log.warning(f"Could not merge with state file: {e}")
            return ds

    def _create_routing_files(self, ds: xr.Dataset, runoff_var: str,
                               baseflow_var: Optional[str], output_dir: str) -> None:
        """
        Create routing input files (fluxes_lat_lon format) from CWatM output.

        This mimics the VIC disagg_results method output format.
        """
        log.debug("Creating routing input files")

        # Prepare data subset
        if baseflow_var:
            data_vars = [runoff_var, baseflow_var]
        else:
            data_vars = [runoff_var]

        fluxes = ds[data_vars]

        # Get coordinate names (CWatM may use different names)
        lat_coord = self._find_coord(fluxes, ['lat', 'latitude', 'y'])
        lon_coord = self._find_coord(fluxes, ['lon', 'longitude', 'x'])

        if lat_coord is None or lon_coord is None:
            raise ValueError("Could not find lat/lon coordinates in CWatM output")

        # Rename to standard names for processing
        fluxes = fluxes.rename({lat_coord: 'lat', lon_coord: 'lon'})

        # Create consistent variable names for routing
        rename_dict = {runoff_var: 'OUT_RUNOFF'}
        if baseflow_var:
            rename_dict[baseflow_var] = 'OUT_BASEFLOW'
        fluxes = fluxes.rename(rename_dict)

        # If no baseflow, create zero baseflow
        if not baseflow_var:
            fluxes['OUT_BASEFLOW'] = xr.zeros_like(fluxes['OUT_RUNOFF'])

        # Add dummy precipitation and evaporation for routing format compatibility
        fluxes['OUT_PREC'] = xr.zeros_like(fluxes['OUT_RUNOFF'])
        fluxes['OUT_EVAP'] = xr.zeros_like(fluxes['OUT_RUNOFF'])

        # VIC Routing truncates lat/lon values
        lats_vicfmt = (np.floor(np.abs(fluxes.lat.values) * 100) / 100) * np.sign(fluxes.lat.values)
        lons_vicfmt = (np.floor(np.abs(fluxes.lon.values) * 100) / 100) * np.sign(fluxes.lon.values)

        total_files = 0
        s = time.time()

        for lat_idx, lat_val in enumerate(fluxes.lat):
            for lon_idx in range(len(fluxes.lon)):
                # Check if there's valid data at this grid cell
                sample_val = fluxes['OUT_RUNOFF'].isel(time=0, lat=lat_idx, lon=lon_idx).values
                if not math.isnan(sample_val):
                    fname = os.path.join(
                        output_dir,
                        f"fluxes_{lats_vicfmt[lat_idx]:.2f}_{lons_vicfmt[lon_idx]:.2f}"
                    )

                    # Extract time series for this grid cell
                    da = fluxes.isel(lat=lat_idx, lon=lon_idx).to_dataframe().reset_index()

                    # Write in VIC routing format
                    da.to_csv(
                        fname,
                        sep=' ',
                        header=False,
                        index=False,
                        float_format="%.5f",
                        quotechar="",
                        quoting=csv.QUOTE_NONE,
                        date_format="%Y %m %d",
                        escapechar="\t"
                    )
                    total_files += 1

            if lat_idx % 10 == 0:
                log.debug(f"Processed latitude band {lat_idx}/{len(fluxes.lat)}")

        log.log(NOTIFICATION, f"Created {total_files} routing input files in {time.time()-s:.2f}s")

    def _find_coord(self, ds: xr.Dataset, coord_names: list) -> Optional[str]:
        """Find a coordinate in the dataset from a list of possible names."""
        for coord in coord_names:
            if coord in ds.coords or coord in ds.dims:
                return coord
        return None

    def save_state(self, state_path: str, state_date: datetime.datetime) -> str:
        """
        Save CWatM model state for restart.

        Parameters
        ----------
        state_path : str
            Directory where state file will be saved
        state_date : datetime.datetime
            Date for which to save the state

        Returns
        -------
        str
            Path to the saved state file
        """
        create_directory(state_path, True)

        # CWatM state files are typically saved automatically
        # This method copies/moves them to the specified location
        state_file = os.path.join(
            state_path,
            f"cwatm_state_{state_date.strftime('%Y%m%d')}.nc"
        )

        log.debug(f"Saving CWatM state to: {state_file}")

        # Check if CWatM already saved a state file
        if self.output_dir:
            cwatm_state_pattern = os.path.join(self.output_dir, '*state*.nc')
            import glob
            state_files = glob.glob(cwatm_state_pattern)
            if state_files:
                import shutil
                shutil.copy2(state_files[-1], state_file)
                log.debug(f"Copied CWatM state from {state_files[-1]}")

        return state_file

    def load_state(self, state_path: str) -> None:
        """
        Load a previously saved CWatM state.

        Parameters
        ----------
        state_path : str
            Path to the state file to load
        """
        if os.path.exists(state_path):
            self.init_state_file = state_path
            self.init_load = True
            log.debug(f"CWatM will load state from: {state_path}")
        else:
            log.warning(f"State file not found: {state_path}")

    def generate_routing_input_state(self, ndays: int, rout_input_state_file: str,
                                      save_path: str, use_rout_state: bool) -> None:
        """
        Generate routing input state file from CWatM outputs.

        This method creates a state file containing the last N days of
        CWatM output for use in continuous runs.

        Parameters
        ----------
        ndays : int
            Number of days to include in state file
        rout_input_state_file : str
            Path to existing routing state file (if any)
        save_path : str
            Path where new state file will be saved
        use_rout_state : bool
            Whether to use existing routing state
        """
        log.debug(f"Generating routing input state file: {save_path}")

        cwatm_output = self.get_runoff_output()

        if os.path.isfile(rout_input_state_file) and use_rout_state:
            log.debug(f"Merging with existing state file: {rout_input_state_file}")

            new_output = xr.open_dataset(cwatm_output)
            first_existing_time = new_output.time[0]
            new_output.close()

            def _remove_coinciding_days(ds, cutoff_time, ndays):
                file_name = ds.encoding.get("source", "")
                file_stem = Path(file_name).stem if file_name else ""
                if 'ro_init' in file_stem or 'state' in file_stem.lower():
                    return ds.sel(
                        time=slice(
                            cutoff_time - np.timedelta64(ndays, 'D'),
                            cutoff_time - np.timedelta64(1, 'D')
                        )
                    )
                return ds

            remove_func = partial(_remove_coinciding_days,
                                  cutoff_time=first_existing_time, ndays=ndays)

            try:
                merged = xr.open_mfdataset(
                    [rout_input_state_file, cwatm_output],
                    preprocess=remove_func
                )
                merged.to_netcdf(save_path)
                merged.close()
                log.debug(f"Saved merged routing state to: {save_path}")
            except Exception as e:
                log.warning(f"Could not merge state files: {e}")
                # Fall back to just using recent output
                new_output = xr.open_dataset(cwatm_output)
                recent = new_output.isel(time=slice(-ndays, None))
                recent.to_netcdf(save_path)
                new_output.close()
        else:
            # No existing state, create new one from recent output
            output = xr.open_dataset(cwatm_output)
            recent = output.isel(time=slice(-ndays, None))
            recent.to_netcdf(save_path)
            output.close()
            log.debug(f"Saved new routing state to: {save_path}")

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Get mapping from CWatM variable names to RAT standard names.

        Returns
        -------
        dict
            Dictionary mapping RAT standard names to CWatM-specific names
        """
        return {
            'runoff': 'runoff',
            'baseflow': 'baseflow',
            'evaporation': 'totalET',
            'net_radiation': 'Rn',
            'vapor_pressure': 'VP',  # May need estimation
            'wind_speed': 'wind',  # May need from forcing
            'air_temperature': 'Tavg',
            'precipitation': 'Precipitation'
        }


# Register CWatM with the ModelFactory
ModelFactory.register('cwatm', CWatMRunner)


class CWatMParameterManager:
    """
    Manager for CWatM parameter and settings files.

    This class handles reading, modifying, and writing CWatM settings files
    to configure model runs within RAT's pipeline.
    """

    def __init__(self, template_settings_file: str, output_settings_file: str):
        """
        Initialize the parameter manager.

        Parameters
        ----------
        template_settings_file : str
            Path to template CWatM settings file
        output_settings_file : str
            Path where modified settings file will be saved
        """
        self.template_path = template_settings_file
        self.output_path = output_settings_file
        self.settings = {}

    def load_template(self) -> Dict[str, Any]:
        """Load settings from template file."""
        # CWatM uses INI-style settings files
        import configparser
        config = configparser.ConfigParser()
        config.read(self.template_path)

        self.settings = {section: dict(config[section])
                         for section in config.sections()}
        return self.settings

    def update_dates(self, start_date: datetime.datetime,
                     end_date: datetime.datetime) -> None:
        """Update simulation dates in settings."""
        date_format = "%Y-%m-%d"
        if 'TIME-RELATED_CONSTANTS' in self.settings:
            self.settings['TIME-RELATED_CONSTANTS']['StepStart'] = start_date.strftime(date_format)
            self.settings['TIME-RELATED_CONSTANTS']['StepEnd'] = end_date.strftime(date_format)

    def update_paths(self, **paths) -> None:
        """Update file paths in settings."""
        if 'FILE_PATHS' in self.settings:
            for key, value in paths.items():
                if key in self.settings['FILE_PATHS']:
                    self.settings['FILE_PATHS'][key] = str(value)

    def save(self) -> str:
        """Save settings to output file."""
        import configparser
        config = configparser.ConfigParser()

        for section, params in self.settings.items():
            config[section] = params

        with open(self.output_path, 'w') as f:
            config.write(f)

        log.debug(f"Saved CWatM settings to: {self.output_path}")
        return self.output_path
