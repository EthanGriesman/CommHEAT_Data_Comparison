# CommHEAT_Data_Comparison

This repository contains an optimized, production-ready data analysis pipeline for the CommHEAT project.
The pipeline processes HOBO temperature sensor data, compares it against EnergyPlus simulation outputs, and generates:

hourly aggregated datasets

mean squared error (MSE) metrics

AC vs No-AC comparisons

heat event plots

period-intersection summaries

publication-ready figures and Excel outputs

The pipeline is designed for performance, scalability, and reproducibility, using vectorized operations, caching, and parallel execution.

Key Features

vectorized pandas and NumPy operations for speed

centralized file I/O via a reusable DataLoader

global caching of archetype simulations to avoid redundant loads

automatic handling of EnergyPlus datetime edge cases (e.g. 24:00:00)

parallel execution using ThreadPoolExecutor

modular plotting via PlottingManager

consistent output structure for downstream analysis
