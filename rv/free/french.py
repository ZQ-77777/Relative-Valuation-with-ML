"""Download three files from Kenneth French's website (five factors, momentum, FF49 SIC definitions) and unzip them.
If a URL changes, download them manually from the Kenneth French Data Library and keep the file names below."""
import io
import zipfile
from pathlib import Path

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "F-F_Research_Data_5_Factors_2x3.csv": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "F-F_Momentum_Factor.csv": "F-F_Momentum_Factor_CSV.zip",
    "Siccodes49.txt": "Siccodes49.zip",
}


def download_french(dest: Path) -> None:
    import requests
    dest.mkdir(parents=True, exist_ok=True)
    for target, zname in FILES.items():
        if (dest / target).exists():
            continue
        try:
            r = requests.get(BASE + zname, timeout=60)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = z.namelist()[0]                 # each archive holds one file
                (dest / target).write_bytes(z.read(name))
            print("Downloaded", target)
        except Exception as e:
            print(f"Automatic download of {target} failed ({e}). Download {zname} manually from the "
                  f"Kenneth French Data Library, unzip it, rename it to {target} and place it in {dest}")
