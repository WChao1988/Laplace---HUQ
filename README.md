# Laplace---HUQ
 A heteroscedastic uncertainty quantification model based on the zero-mean Laplace distribution to deliver spatially adaptive confidence intervals for gravity-geologic method (GGM) derived bathymetry.
# Code Documentation: Marine Terrain Data Processing and Uncertainty Quantification
This codebase provides a set of modular tools for processing marine terrain data (e.g., abyssal plains, seamounts, trenches), supporting data loading, gravity‑geologic method (GGM) bathymetric inversion, neural network–based uncertainty quantification, and result visualization. The core logic resides in 'method', while low‑level operations are implemented in the 'utls' utility set.

├── data_plain # Abyssal plain data
├── data_seamount # Seamount terrain data
├── data_trench # Trench terrain data
├── utls # General utility library (called by method)
│ ├── distance # Distance calculation
│ ├── longwave # Long‑wave gravity field separation (construct long‑wave matrix)
│ ├── mat_interpolator# Matrix interpolation
│ └── transform # Coordinate transformation (projection, geodetic conversion, etc.)
├── method # Main script: GGM inversion + uncertainty quantification
└── utls_plot # Visualization helpers (plotting based on utility results)
