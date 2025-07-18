"""
# Recreation.gov API example
"""

# %%
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

# %%

# Download the data from the Recreation.gov website cause API doesn't go back as far

# Destination folder
output_dir = Path("../data/reservations")
output_dir.mkdir(parents=True, exist_ok=True)

# List of URLs to download
BASE_URL = "https://ridb.recreation.gov/downloads/reservations{}.zip"
years = range(2006, 2025)

for year in tqdm(years, desc="Processing ZIP files"):
    url = BASE_URL.format(year)
    ZIP_NAME = f"data_{year}.zip"
    zip_path = Path(ZIP_NAME)

    try:
        # Download zip file
        response = requests.get(url, stream=True, timeout=10)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract contents to a temporary folder
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            temp_dir = output_dir / f"temp_{year}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            zip_ref.extractall(temp_dir)

        # Find the CSV file in the extracted contents
        csv_path = list(temp_dir.glob("*.csv"))[0]

        # Rename and move the CSV
        final_csv_path = output_dir / f"{year}.csv"
        csv_path.rename(final_csv_path)

        # Clean up: delete zip and temporary folder
        zip_path.unlink()
        for file in temp_dir.iterdir():
            file.unlink()
        temp_dir.rmdir()

    except requests.exceptions.RequestException as e:
        print(f"Download failed for {url}: {e}")
    except zipfile.BadZipFile as e:
        print(f"Invalid zip file {ZIP_NAME}: {e}")
    except OSError as e:
        print(f"Filesystem error with {ZIP_NAME}: {e}")

# %%
