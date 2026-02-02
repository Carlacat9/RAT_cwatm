"""
CWatM Parameter File Reader and Manager for RAT.

This module provides the CWatMParameterFile class that manages CWatM settings
files, similar to how VICParameterFile manages VIC parameter files.

CWatM uses INI-style settings files with multiple sections for different
model components.
"""

import os
import datetime
import configparser
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

from logging import getLogger
from rat.utils.logging import LOG_NAME, NOTIFICATION, LOG_LEVEL
from rat.utils.utils import create_directory

log = getLogger(LOG_NAME)
log.setLevel(LOG_LEVEL)


class CWatMParameterFile:
    """
    Context manager for CWatM parameter/settings file management.

    This class handles reading, modifying, and writing CWatM settings files
    for use in RAT's pipeline. It follows the same pattern as VICParameterFile
    for consistency.

    CWatM settings files are INI-format with sections:
    - FILE_PATHS: Input/output file paths
    - NETCDF_ATTRIBUTES: NetCDF metadata
    - TIME-RELATED_CONSTANTS: Simulation dates and time stepping
    - INOUT: Input/output configurations
    - MASK_OUTLET: Basin mask settings
    - GROUNDWATER: Groundwater module settings
    - WATERDEMAND: Water demand settings
    - RESERVOIR: Reservoir module settings (important for RAT!)
    - LAKES: Lake module settings
    - OUTPUT: Output variable selections

    Attributes
    ----------
    config : dict
        RAT configuration dictionary
    basin_name : str
        Name of the basin being processed
    settings : dict
        CWatM settings dictionary
    """

    # Default CWatM settings template
    DEFAULT_SETTINGS = {
        'FILE_PATHS': {
            'PathRoot': '',
            'PathOut': '',
            'PathMaps': '',
            'PathMeteo': '',
        },
        'NETCDF_ATTRIBUTES': {
            'institution': 'RAT - Reservoir Assessment Tool',
            'title': 'CWatM output for RAT',
            'source': 'CWatM model v1.0',
            'Conventions': 'CF-1.6',
        },
        'TIME-RELATED_CONSTANTS': {
            'StepStart': '',
            'StepEnd': '',
            'SpinUp': '0',
            'timestepsInit': '1',
        },
        'INOUT': {
            'MaskMap': '',
            'Ldd': '',
            'Outlets': '',
        },
        'OPTIONS': {
            'calc_evaporation': 'True',
            'includeWaterDemand': 'False',
            'includeWaterBodies': 'True',
            'includeRoutingKinematic': 'True',
            'includeGroundwater': 'True',
            'includeLakes': 'True',
            'includeReservoirs': 'True',  # Critical for RAT
        },
        'RESERVOIR': {
            'useResAndLakes': 'True',
            'ReservoirSimple': 'True',
            'ResAndLakesArea': '',
            'ReservoirID': '',
        },
        'OUTPUT': {
            'OUT_Dir': '',
            'OUT_TSS_Daily': 'discharge',
            'OUT_Map_Daily': 'runoff,baseflow,discharge,totalET',
        }
    }

    def __init__(self, config: Dict[str, Any], basin_name: str,
                 startdate: Optional[datetime.datetime] = None,
                 enddate: Optional[datetime.datetime] = None,
                 cwatm_output_path: Optional[str] = None,
                 forcing_path: Optional[str] = None,
                 init_state: Optional[str] = None,
                 init_state_dir: Optional[str] = None,
                 save_init_state: bool = True,
                 init_state_save_date: Optional[datetime.datetime] = None,
                 init_state_out_dir: Optional[str] = None,
                 runname: Optional[str] = None):
        """
        Initialize CWatM parameter file manager.

        Parameters
        ----------
        config : dict
            RAT configuration dictionary
        basin_name : str
            Name of the basin being processed
        startdate : datetime, optional
            Simulation start date
        enddate : datetime, optional
            Simulation end date
        cwatm_output_path : str, optional
            Path for CWatM outputs
        forcing_path : str, optional
            Path to meteorological forcing data
        init_state : str, optional
            Path to initial state file or date
        init_state_dir : str, optional
            Directory containing initial state files
        save_init_state : bool
            Whether to save state at end of simulation
        init_state_save_date : datetime, optional
            Date for which to save state
        init_state_out_dir : str, optional
            Directory where state will be saved
        runname : str, optional
            Unique identifier for this run
        """
        self.config = config
        self.basin_name = basin_name
        self.settings = self._init_default_settings()

        # Get CWatM configuration section
        self.cwatm_config = config.get('CWATM', {})

        # Template settings file
        self.init_settings_file = self.cwatm_config.get('cwatm_settings_file', None)
        self.workspace = self.cwatm_config.get('cwatm_workspace', None)

        # Paths
        self.cwatm_settings_path: Optional[str] = None
        self.cwatm_output_path = cwatm_output_path
        self.forcing_path = forcing_path

        # Dates
        self.startdate = startdate
        self.enddate = enddate
        self.cwatm_startdate: Optional[datetime.datetime] = None
        self.cwatm_enddate: Optional[datetime.datetime] = None

        # State management
        self.init_state = init_state
        self.init_state_dir = init_state_dir or 'cwatm_init_states'
        self.save_init_state = save_init_state
        self.init_state_save_date = init_state_save_date or config['BASIN'].get('end')
        self.init_state_out_dir = init_state_out_dir or 'cwatm_init_states'

        # Run identifier
        if runname is None:
            self.runname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.runname = str(runname)

        # Set up workspace
        self._setup_workspace()

        # Load template if available
        if self.init_settings_file and os.path.exists(self.init_settings_file):
            self._load_from_settings_file()

        # Apply configuration
        self._load_from_config()

    def _init_default_settings(self) -> Dict[str, Dict[str, str]]:
        """Initialize with default settings."""
        return {section: dict(params)
                for section, params in self.DEFAULT_SETTINGS.items()}

    def _setup_workspace(self) -> None:
        """Set up workspace directory for CWatM files."""
        if self.workspace:
            self.workspace = create_directory(
                os.path.join(self.workspace, f'run_{self.runname}'), True
            )
        else:
            self.workspace = create_directory(
                os.path.join(
                    self.config['GLOBAL']['data_dir'],
                    self.config['BASIN']['region_name'],
                    'basins', self.basin_name, 'cwatm', 'cwatm_workspace',
                    f'run_{self.runname}'
                ), True
            )

        self.cwatm_settings_path = os.path.join(self.workspace, 'cwatm_settings.ini')

    def _load_from_settings_file(self) -> None:
        """Load settings from existing CWatM settings file."""
        log.debug(f"Loading CWatM settings from: {self.init_settings_file}")

        parser = configparser.ConfigParser()
        parser.read(self.init_settings_file)

        for section in parser.sections():
            if section not in self.settings:
                self.settings[section] = {}
            for key, value in parser[section].items():
                self.settings[section][key] = value

    def _load_from_config(self) -> None:
        """Load/override settings from RAT configuration."""
        config = self.config

        # Update dates
        if self.startdate:
            self.settings['TIME-RELATED_CONSTANTS']['StepStart'] = \
                self.startdate.strftime('%Y-%m-%d')
            self.cwatm_startdate = self.startdate

        if self.enddate:
            self.settings['TIME-RELATED_CONSTANTS']['StepEnd'] = \
                self.enddate.strftime('%Y-%m-%d')
            self.cwatm_enddate = self.enddate

        # Update output path
        if self.cwatm_output_path:
            self.settings['FILE_PATHS']['PathOut'] = self.cwatm_output_path
            self.settings['OUTPUT']['OUT_Dir'] = self.cwatm_output_path

        # Update forcing path
        if self.forcing_path:
            self.settings['FILE_PATHS']['PathMeteo'] = self.forcing_path

        # Update mask/domain from CWATM config section
        if self.cwatm_config.get('mask_map'):
            self.settings['INOUT']['MaskMap'] = self.cwatm_config['mask_map']

        if self.cwatm_config.get('ldd_map'):
            self.settings['INOUT']['Ldd'] = self.cwatm_config['ldd_map']

        # Update reservoir settings if available
        if self.cwatm_config.get('reservoir_file'):
            self.settings['RESERVOIR']['ReservoirID'] = \
                self.cwatm_config['reservoir_file']

        # Handle state files
        if self.init_state:
            self._configure_init_state()

        if self.save_init_state:
            self._configure_save_state()

        # Override with any CWATM PARAMETERS section
        if config.get('CWATM PARAMETERS'):
            for key, value in config['CWATM PARAMETERS'].items():
                # Find the section containing this parameter
                for section in self.settings:
                    if key in self.settings[section]:
                        self.settings[section][key] = str(value)
                        break
                else:
                    # If not found in any section, add to OPTIONS
                    if 'OPTIONS' not in self.settings:
                        self.settings['OPTIONS'] = {}
                    self.settings['OPTIONS'][key] = str(value)

    def _configure_init_state(self) -> None:
        """Configure initial state loading."""
        if isinstance(self.init_state, datetime.date):
            init_state_date_str = self.init_state.strftime('%Y%m%d')
            state_path = os.path.join(
                self.config['GLOBAL']['data_dir'],
                self.config['BASIN']['region_name'],
                'basins', self.basin_name, 'cwatm',
                self.init_state_dir,
                f'cwatm_state_{init_state_date_str}.nc'
            )
        else:
            state_path = str(Path(self.init_state).resolve())

        if 'INITIAL_CONDITIONS' not in self.settings:
            self.settings['INITIAL_CONDITIONS'] = {}

        self.settings['INITIAL_CONDITIONS']['load_initial'] = 'True'
        self.settings['INITIAL_CONDITIONS']['initLoad'] = state_path

    def _configure_save_state(self) -> None:
        """Configure state saving at end of simulation."""
        if 'INITIAL_CONDITIONS' not in self.settings:
            self.settings['INITIAL_CONDITIONS'] = {}

        save_dir = create_directory(
            os.path.join(
                self.config['GLOBAL']['data_dir'],
                self.config['BASIN']['region_name'],
                'basins', self.basin_name, 'cwatm',
                self.init_state_out_dir
            ), False
        )

        self.settings['INITIAL_CONDITIONS']['save_initial'] = 'True'
        self.settings['INITIAL_CONDITIONS']['initSave'] = save_dir

    def _write(self) -> None:
        """Write settings to INI file."""
        parser = configparser.ConfigParser()

        for section, params in self.settings.items():
            parser[section] = params

        with open(self.cwatm_settings_path, 'w') as f:
            # Write header comment
            f.write(f'# CWatM Settings File for RAT\n')
            f.write(f'# Generated by CWatMParameterFile on '
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %X")}\n')
            f.write(f'# Basin: {self.basin_name}\n\n')

        # Append using configparser
        with open(self.cwatm_settings_path, 'a') as f:
            parser.write(f)

    def get_output_file(self) -> str:
        """
        Get the expected primary output file path.

        Returns
        -------
        str
            Path to the main CWatM output file
        """
        out_dir = self.settings['OUTPUT'].get('OUT_Dir',
                                               self.settings['FILE_PATHS'].get('PathOut', ''))
        start_str = self.cwatm_startdate.strftime('%Y%m%d') if self.cwatm_startdate else ''
        end_str = self.cwatm_enddate.strftime('%Y%m%d') if self.cwatm_enddate else ''

        return os.path.join(out_dir, f'discharge_dailyTot_{start_str}_{end_str}.nc')

    def __enter__(self):
        """Context manager entry."""
        # Save config record
        config_record_path = os.path.join(self.workspace, 'config_record.yml')
        yaml.dump(self.config, open(config_record_path, 'w'))

        # Copy template settings file if exists
        if self.init_settings_file and os.path.exists(self.init_settings_file):
            import shutil
            template_copy = os.path.join(self.workspace, 'template_settings.ini')
            shutil.copy2(self.init_settings_file, template_copy)

        # Write the modified settings
        self._write()

        log.debug(f"CWatM settings written to: {self.cwatm_settings_path}")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Context manager exit."""
        pass

    def update_setting(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific setting.

        Parameters
        ----------
        section : str
            Settings section name
        key : str
            Parameter name
        value : Any
            New value (will be converted to string)
        """
        if section not in self.settings:
            self.settings[section] = {}
        self.settings[section][key] = str(value)

    def get_setting(self, section: str, key: str,
                    default: Optional[str] = None) -> Optional[str]:
        """
        Get a specific setting value.

        Parameters
        ----------
        section : str
            Settings section name
        key : str
            Parameter name
        default : str, optional
            Default value if not found

        Returns
        -------
        str or None
            The setting value or default
        """
        return self.settings.get(section, {}).get(key, default)
