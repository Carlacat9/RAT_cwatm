# RAT-CWatM Integration Guide

## Overview

This document describes how to use the Community Water Model (CWatM) with RAT (Reservoir Assessment Tool) for improved reservoir inflow estimation in human-dominated basins.

## What is CWatM?

CWatM (Community Water Model) is a comprehensive global hydrological model developed by IIASA's Water Security group. Unlike VIC, CWatM explicitly models human water management activities including:

- **Reservoir operations** - Dam storage and releases
- **Irrigation** - Agricultural water demand and consumption
- **Groundwater extraction** - Aquifer dynamics and pumping
- **Water withdrawals** - Industrial and domestic water use

## Benefits of Using CWatM with RAT

### 1. Better Representation of Human Water Management

CWatM's explicit modeling of reservoirs, irrigation, and water withdrawals provides more realistic inflow estimates for basins where human activities significantly affect the water cycle.

### 2. Improved Accuracy for Regulated Rivers

For rivers with upstream dams, diversions, or significant water abstractions, CWatM can better capture the modified flow regime compared to VIC's natural flow approach.

### 3. Consistent Global Parameterization

CWatM comes with global parameter datasets, making it easier to set up for new basins without extensive calibration.

### 4. Flexible Model Selection

RAT now supports switching between VIC and CWatM based on basin characteristics, allowing you to use the most appropriate model for each application.

## Quick Start

### 1. Install CWatM

Follow the CWatM installation guide: https://cwatm.iiasa.ac.at/

```bash
# Create conda environment
conda create -n cwatm_env python=3.9
conda activate cwatm_env

# Install CWatM dependencies
pip install netCDF4 numpy scipy pandas xarray
```

### 2. Configure RAT for CWatM

In your RAT configuration file, set:

```yaml
GLOBAL:
  hydrological_model: 'cwatm'  # Switch from 'vic' to 'cwatm'

CWATM:
  cwatm_path: '/path/to/cwatm/run_cwatm.py'
  cwatm_env: 'cwatm_env'
  cwatm_settings_file: '/path/to/settings_template.ini'
  mask_map: '/path/to/basin_mask.nc'
  ldd_map: '/path/to/ldd.nc'
```

### 3. Run RAT

```bash
rat run --config your_config.yml
```

## Configuration Options

### Required CWatM Settings

| Setting | Description |
|---------|-------------|
| `cwatm_path` | Path to CWatM's main run script |
| `cwatm_env` | Conda environment name with CWatM installed |
| `cwatm_settings_file` | Template CWatM settings file |
| `mask_map` | Basin mask in NetCDF format |
| `ldd_map` | Local Drain Direction map |

### Optional CWatM Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `cwatm_workspace` | auto | Working directory for CWatM runs |
| `use_metsim` | true | Whether to use MetSim for forcing preparation |
| `cwatm_global_data` | false | Use global parameter files |
| `reservoir_file` | null | CWatM reservoir configuration |
| `init_load` | false | Load initial state from file |

## Workflow Comparison

### VIC Workflow (Original)
```
Meteorological Data → MetSim → VIC → Routing → RAT Postprocessing
```

### CWatM Workflow (New)
```
Meteorological Data → [MetSim] → CWatM → Routing → RAT Postprocessing
```

Note: MetSim can optionally be used for forcing preparation with CWatM.

## Variable Mapping

RAT automatically handles variable name mapping between VIC and CWatM outputs:

| RAT Variable | VIC Name | CWatM Name |
|--------------|----------|------------|
| Runoff | OUT_RUNOFF | runoff |
| Baseflow | OUT_BASEFLOW | baseflow |
| Evaporation | OUT_EVAP | totalET |
| Air Temperature | OUT_AIR_TEMP | Tavg |
| Net Radiation | OUT_R_NET | Rn |

## Preparing CWatM Input Data

### Option 1: Through MetSim (Recommended)

RAT can use MetSim to prepare meteorological forcing and then convert to CWatM format:

```yaml
CWATM:
  use_metsim: true
```

### Option 2: Direct Conversion

For faster processing, RAT can convert combined NetCDF directly to CWatM format:

```yaml
CWATM:
  use_metsim: false
```

## CWatM Settings File Template

RAT modifies a template CWatM settings file (INI format) for each run. Example template:

```ini
[FILE_PATHS]
PathRoot = /path/to/data
PathOut = /path/to/output
PathMaps = /path/to/maps
PathMeteo = /path/to/forcing

[TIME-RELATED_CONSTANTS]
StepStart = 2020-01-01
StepEnd = 2020-12-31
SpinUp = 0

[OPTIONS]
includeReservoirs = True
includeWaterDemand = False
includeGroundwater = True

[OUTPUT]
OUT_Dir = /path/to/output
OUT_Map_Daily = runoff,baseflow,discharge,totalET
```

RAT will automatically update dates and paths in this template.

## Troubleshooting

### CWatM Not Found

Ensure `cwatm_path` points to the correct run script and the conda environment is properly set up.

### Missing Variables in Output

CWatM may output different variables based on configuration. Check that your settings file includes required outputs:

```ini
[OUTPUT]
OUT_Map_Daily = runoff,baseflow,discharge,totalET
```

### State File Issues

For continuous runs, ensure CWatM state files are being saved:

```ini
[INITIAL_CONDITIONS]
save_initial = True
initSave = /path/to/state/directory
```

## References

- CWatM Documentation: https://cwatm.iiasa.ac.at/
- CWatM GitHub: https://github.com/iiasa/CWatM
- CWatM Paper: https://gmd.copernicus.org/articles/13/3267/2020/
- RAT Documentation: https://github.com/UW-SASWE/RAT

## Support

For issues with the RAT-CWatM integration, please open an issue at:
https://github.com/UW-SASWE/RAT/issues
