"""Phase 1 - Step 1: Download IMD 0.25-degree daily gridded rainfall.

Source: IMD Climate Research & Services, Pune (Pai et al. 2014).
https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html

Each yearly file is a NetCDF with one daily RAINFALL map (mm) over India.
We download one file per year for 2001-2020 into data/raw/imd_rainfall/.
"""
import os
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parents[2]

YEARS = list(range(2001, 2021))
POST_URL = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php"
OUT_DIR = str(BASE / "data" / "raw" / "imd_rainfall")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for year in YEARS:
        out_path = os.path.join(OUT_DIR, f"imd_rainfall_{year}.nc")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
            print(f"[skip] {year} already present")
            continue
        print(f"[download] IMD rainfall {year} ...")
        try:
            resp = requests.post(POST_URL, data={"RF25": str(year)}, timeout=180)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[error] {year}: {exc}")
            continue
        if len(resp.content) < 1_000_000:
            print(f"[warn] {year}: response only {len(resp.content)} bytes, skipping")
            continue
        with open(out_path, "wb") as handle:
            handle.write(resp.content)
        print(f"[ok] {year}: {len(resp.content) / 1e6:.1f} MB")
        time.sleep(1)


if __name__ == "__main__":
    main()
