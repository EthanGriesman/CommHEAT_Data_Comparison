<h1 align="center">CommHEAT Data Analysis Pipeline</h1>

## Overview
The **CommHEAT Data Analysis Pipeline** processes HOBO temperature sensor data, compares it with EnergyPlus building simulation outputs, and generates statistical analyses and visualizations.

The system is designed to **validate residential building energy models** by comparing simulated indoor temperatures against measured sensor data during **summer 2025 heat events**.

---

## Installation
Install the required Python dependencies:

```bash
pip install pandas numpy matplotlib openpyxl tqdm
