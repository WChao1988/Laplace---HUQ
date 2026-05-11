# Laplace-HUQ

**Heteroscedastic Uncertainty Quantification for GGM‑Derived Bathymetry**

A spatially adaptive confidence interval estimation model based on the **zero‑mean Laplace distribution**. It quantifies heteroscedastic uncertainty in bathymetry obtained by the **Gravity‑Geologic Method (GGM)**, delivering reliable uncertainty bounds across different marine terrains (abyssal plains, seamounts, trenches).

## Overview

This codebase provides modular tools for marine terrain data processing, GGM inversion, neural‑based uncertainty quantification, and result visualisation.  
The core logic resides in the `method/` directory, while low‑level operations are implemented as reusable utilities in `utls/`.

## Key Features

- **GGM bathymetric inversion** from gravity anomalies
- **Zero‑mean Laplace distribution** modelling for heteroscedastic aleatoric uncertainty
- **Spatially adaptive confidence intervals** that vary with local terrain complexity
- Support for three typical seafloor morphologies:
  - Abyssal plains
  - Seamounts
  - Oceanic trenches
- Modular utilities for:
  - Distance computation
  - Long‑wave gravity field separation
  - Matrix interpolation
  - Coordinate transformation (projection, geodetic conversions)

## Directory Structure

```
├── data_plain                 # Example data for abyssal plain areas
├── data_seamount              # Example data for seamount features
├── data_trench                # Example data for trench areas
├── utls                       # General utility library (called by method/)
│   ├── distance               # Distance calculation
│   ├── longwave               # Long‑wave gravity field separation (construct long‑wave matrix)
│   ├── mat_interpolator       # Matrix interpolation routines
│   └── transform              # Coordinate transformation (projection, geodetic conversion, etc.)
├── method                     # Main scripts: GGM inversion + uncertainty quantification
└── utls_plot                  # Visualisation helpers (plotting based on utility results)
```
