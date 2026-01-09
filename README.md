## Configure Paths and Run the Pipeline

### 5. Configure Paths in `config.py`

Open `config.py` and update the directory paths so they match your local file structure.

```python
# Directory paths for input data
hobo_dir: Path = Path(
    r"C:\Users\Ethan\Downloads\2025SCC_OnsetHobo_InHome_Dataloggers\2025SCC_OnsetHobo_InHome_Dataloggers"
)

mapping_file: Path = Path(
    r"C:\Users\Ethan\Downloads\2025SCC_OnsetHobo_InHome_Dataloggers\2025SCC_OnsetHobo_InHome_Dataloggers\Sensor Contact_101325_PickUP.xlsx"
)

latest_ep_dir: Path = Path(
    r"C:\Users\Ethan\Downloads\Latest_EP_Output_Files\Latest_EP_Output_Files"
)

output_dir: Path = Path(
    r"C:\Users\Ethan\OneDrive - Iowa State University\Desktop\CommHEAT Output"
)
