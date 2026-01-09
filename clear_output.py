from pathlib import Path
import shutil

OUTPUT_DIR = Path(r"C:\Users\Ethan\OneDrive - Iowa State University\Desktop\CommHeat Output")

if not OUTPUT_DIR.exists():
    raise FileNotFoundError(f"Directory does not exist: {OUTPUT_DIR}")

for item in OUTPUT_DIR.iterdir():
    try:
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
        print(f"Deleted: {item}")
    except Exception as e:
        print(f"Failed to delete {item}: {e}")

print("\n✔ CommHeat Output directory cleared.")
