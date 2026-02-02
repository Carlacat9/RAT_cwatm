"""
Abstract base class for hydrological models in RAT.

This module provides the interface that all hydrological models (VIC, CWatM, etc.)
must implement to work with RAT's reservoir assessment pipeline.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import datetime


class HydrologicalModel(ABC):
    """
    Abstract base class for hydrological models in RAT.

    All hydrological models (VIC, CWatM, etc.) should inherit from this class
    and implement the required methods to integrate with RAT's pipeline.

    The hydrological model is responsible for:
    1. Generating runoff/discharge from meteorological forcings
    2. Providing outputs compatible with RAT's routing model
    3. Providing variables needed for evaporation calculation
    """

    # Class-level constants for model identification
    MODEL_NAME: str = "base"
    MODEL_VERSION: str = "1.0"

    # Required output variables for evaporation calculation
    EVAP_REQUIRED_VARS = ['OUT_R_NET', 'OUT_VP', 'OUT_WIND', 'OUT_AIR_TEMP']

    # Required output variables for routing
    ROUTING_REQUIRED_VARS = ['OUT_RUNOFF', 'OUT_BASEFLOW']

    def __init__(self, config: Dict[str, Any], basin_name: str) -> None:
        """
        Initialize the hydrological model.

        Parameters
        ----------
        config : dict
            RAT configuration dictionary
        basin_name : str
            Name of the basin being processed
        """
        self.config = config
        self.basin_name = basin_name
        self.model_output_path: Optional[Path] = None
        self.start_date: Optional[datetime.datetime] = None
        self.end_date: Optional[datetime.datetime] = None

    @abstractmethod
    def setup(self, start_date: datetime.datetime, end_date: datetime.datetime,
              forcing_path: str, output_path: str, **kwargs) -> None:
        """
        Set up the model for a simulation run.

        Parameters
        ----------
        start_date : datetime.datetime
            Start date of simulation
        end_date : datetime.datetime
            End date of simulation
        forcing_path : str
            Path to meteorological forcing data
        output_path : str
            Path where model outputs will be saved
        **kwargs : dict
            Additional model-specific parameters
        """
        pass

    @abstractmethod
    def run(self, num_processors: int = 1) -> int:
        """
        Execute the hydrological model.

        Parameters
        ----------
        num_processors : int
            Number of processors to use for parallel execution

        Returns
        -------
        int
            Return code (0 for success, non-zero for failure)
        """
        pass

    @abstractmethod
    def get_runoff_output(self) -> str:
        """
        Get the path to the runoff output file.

        Returns
        -------
        str
            Path to the NetCDF file containing runoff data
        """
        pass

    @abstractmethod
    def get_evaporation_variables(self) -> str:
        """
        Get the path to the file containing evaporation-related variables.

        Returns
        -------
        str
            Path to the NetCDF file containing variables needed for
            evaporation calculation (R_NET, VP, WIND, AIR_TEMP)
        """
        pass

    @abstractmethod
    def prepare_routing_input(self, rout_input_dir: str,
                               rout_input_state_file: Optional[str] = None,
                               use_rout_state: bool = False) -> None:
        """
        Prepare input files for the routing model.

        This method should convert model outputs to the format expected
        by RAT's routing model (fluxes_*.csv files).

        Parameters
        ----------
        rout_input_dir : str
            Directory where routing input files will be saved
        rout_input_state_file : str, optional
            Path to routing state file for continuous runs
        use_rout_state : bool
            Whether to use existing routing state
        """
        pass

    @abstractmethod
    def save_state(self, state_path: str, state_date: datetime.datetime) -> str:
        """
        Save model state for restart capability.

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
        pass

    @abstractmethod
    def load_state(self, state_path: str) -> None:
        """
        Load a previously saved model state.

        Parameters
        ----------
        state_path : str
            Path to the state file to load
        """
        pass

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Get mapping from model-specific variable names to RAT standard names.

        Override this method if the model uses different variable names.

        Returns
        -------
        dict
            Dictionary mapping RAT standard names to model-specific names
        """
        return {
            'runoff': 'OUT_RUNOFF',
            'baseflow': 'OUT_BASEFLOW',
            'evaporation': 'OUT_EVAP',
            'net_radiation': 'OUT_R_NET',
            'vapor_pressure': 'OUT_VP',
            'wind_speed': 'OUT_WIND',
            'air_temperature': 'OUT_AIR_TEMP',
            'precipitation': 'OUT_PREC'
        }

    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the model.

        Returns
        -------
        dict
            Dictionary containing model name, version, and other metadata
        """
        return {
            'name': self.MODEL_NAME,
            'version': self.MODEL_VERSION,
            'basin': self.basin_name
        }

    def validate_outputs(self) -> Tuple[bool, str]:
        """
        Validate that all required outputs exist and are readable.

        Returns
        -------
        tuple
            (success: bool, message: str)
        """
        try:
            runoff_path = self.get_runoff_output()
            evap_path = self.get_evaporation_variables()

            if not Path(runoff_path).exists():
                return False, f"Runoff output file not found: {runoff_path}"

            if not Path(evap_path).exists():
                return False, f"Evaporation variables file not found: {evap_path}"

            return True, "All outputs validated successfully"
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class ModelFactory:
    """
    Factory class for creating hydrological model instances.
    """

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """
        Register a new model class.

        Parameters
        ----------
        name : str
            Name to register the model under (e.g., 'vic', 'cwatm')
        model_class : type
            The model class to register
        """
        cls._models[name.lower()] = model_class

    @classmethod
    def create(cls, model_name: str, config: Dict[str, Any],
               basin_name: str) -> HydrologicalModel:
        """
        Create a model instance.

        Parameters
        ----------
        model_name : str
            Name of the model to create ('vic', 'cwatm', etc.)
        config : dict
            RAT configuration dictionary
        basin_name : str
            Name of the basin

        Returns
        -------
        HydrologicalModel
            Instance of the requested model

        Raises
        ------
        ValueError
            If the requested model is not registered
        """
        model_name = model_name.lower()
        if model_name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {available}"
            )
        return cls._models[model_name](config, basin_name)

    @classmethod
    def available_models(cls) -> list:
        """
        Get list of available model names.

        Returns
        -------
        list
            List of registered model names
        """
        return list(cls._models.keys())
