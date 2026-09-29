"""Hourly 2-m air temperature for every grid area (NASA POWER, MERRA-2 based).

Each area is represented by one to four load centres; its temperature is the
simple average over them.  Output: data/interim/weather_t2m.parquet with a UTC
hourly index and one column per area code of the global panel.
"""
import os
import json
import time
import subprocess
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "raw", "weather")

POINTS = {
    # --- United States (balancing authorities) ---
    "US-SCL": [(47.61, -122.33)], "US-TAL": [(30.44, -84.28)], "US-TEC": [(27.95, -82.46)],
    "US-TEPC": [(32.22, -110.97)], "US-TIDC": [(37.49, -120.85)], "US-TPWR": [(47.25, -122.44)],
    "US-TVA": [(36.16, -86.78), (35.96, -83.92), (35.15, -90.05)], "US-AECI": [(37.21, -93.29)],
    "US-AZPS": [(33.45, -112.07)], "US-BANC": [(38.58, -121.49)],
    "US-BPAT": [(45.52, -122.68), (46.60, -120.51)], "US-CHPD": [(47.42, -120.31)],
    "US-CISO": [(34.05, -118.24), (37.77, -122.42), (36.74, -119.79)], "US-CPLE": [(35.78, -78.64)],
    "US-CPLW": [(35.60, -82.55)], "US-DOPD": [(47.65, -120.07)], "US-DUK": [(35.23, -80.84)],
    "US-EPE": [(31.76, -106.49)], "US-ERCO": [(32.78, -96.80), (29.76, -95.37), (29.42, -98.49)],
    "US-FMPP": [(28.54, -81.38)], "US-FPL": [(25.76, -80.19), (26.71, -80.05)],
    "US-GCPD": [(47.13, -119.28)], "US-GVL": [(29.65, -82.32)], "US-HST": [(25.47, -80.48)],
    "US-IID": [(32.79, -115.56)], "US-IPCO": [(43.62, -116.21)],
    "US-ISNE": [(42.36, -71.06), (41.76, -72.68)], "US-JEA": [(30.33, -81.66)],
    "US-LDWP": [(34.05, -118.24)], "US-LGEE": [(38.25, -85.76), (38.04, -84.50)],
    "US-MISO": [(44.98, -93.27), (42.33, -83.05), (29.95, -90.07), (38.63, -90.20), (39.77, -86.16)],
    "US-NEVP": [(36.17, -115.14)], "US-NWMT": [(45.78, -108.50), (46.87, -113.99)],
    "US-NYIS": [(40.71, -74.01), (42.65, -73.75)], "US-PACE": [(40.76, -111.89)],
    "US-PACW": [(45.52, -122.68), (42.33, -122.87)], "US-PGE": [(45.52, -122.68)],
    "US-PJM": [(39.95, -75.17), (38.91, -77.04), (39.96, -83.00), (41.88, -87.63)],
    "US-PNM": [(35.08, -106.65)], "US-PSCO": [(39.74, -104.99)], "US-SC": [(33.20, -80.01)],
    "US-SCEG": [(34.00, -81.03)], "US-SOCO": [(33.75, -84.39), (33.52, -86.80)],
    "US-SWPP": [(35.47, -97.52), (39.10, -94.58), (41.26, -95.93)], "US-WALC": [(34.48, -114.32)],
    "US-AVA": [(47.66, -117.43)],
    # --- Europe (national systems) ---
    "EU-AT": [(48.21, 16.37)], "EU-BE": [(50.85, 4.35)], "EU-BG": [(42.70, 23.32)],
    "EU-CH": [(47.37, 8.54)], "EU-CZ": [(50.08, 14.44)],
    "EU-DE": [(52.52, 13.40), (50.11, 8.68), (48.14, 11.58), (51.46, 7.01)],
    "EU-DK": [(55.68, 12.57), (56.16, 10.20)], "EU-EE": [(59.44, 24.75)],
    "EU-ES": [(40.42, -3.70), (41.39, 2.17), (37.39, -5.98)], "EU-FI": [(60.17, 24.94)],
    "EU-FR": [(48.86, 2.35), (45.76, 4.84), (43.30, 5.37), (50.63, 3.06)], "EU-GR": [(37.98, 23.73)],
    "EU-HR": [(45.81, 15.98)], "EU-HU": [(47.50, 19.04)], "EU-IE": [(53.35, -6.26)],
    "EU-IT": [(45.46, 9.19), (41.90, 12.50), (40.85, 14.27)], "EU-LT": [(54.69, 25.28)],
    "EU-LU": [(49.61, 6.13)], "EU-LV": [(56.95, 24.11)], "EU-NL": [(52.37, 4.90), (51.92, 4.48)],
    "EU-NO": [(59.91, 10.75), (60.39, 5.32)], "EU-PL": [(52.23, 21.01), (50.26, 19.02)],
    "EU-PT": [(38.72, -9.14), (41.16, -8.63)], "EU-RO": [(44.43, 26.10)], "EU-RS": [(44.79, 20.45)],
    "EU-SE": [(59.33, 18.07), (57.71, 11.97)], "EU-SI": [(46.06, 14.51)], "EU-SK": [(48.15, 17.11)],
    "EU-UK": [(51.51, -0.13), (53.48, -2.24), (55.86, -4.25)], "EU-BA": [(43.86, 18.41)],
    "EU-ME": [(42.44, 19.26)], "EU-MK": [(42.00, 21.43)], "EU-MD": [(47.01, 28.86)],
    # --- Oceania ---
    "AU-NSW": [(-33.87, 151.21)], "AU-QLD": [(-27.47, 153.03)], "AU-VIC": [(-37.81, 144.96)],
    "AU-SA": [(-34.93, 138.60)], "AU-TAS": [(-42.88, 147.33)],
    # --- Asia ---
    "CN-01": [(39.90, 116.41)], "CN-02": [(39.13, 117.20)], "CN-03": [(38.04, 114.51)],
    "CN-04": [(37.87, 112.55)], "CN-05": [(40.84, 111.75)], "CN-06": [(41.80, 123.43)],
    "CN-07": [(43.82, 125.32)], "CN-08": [(45.80, 126.53)], "CN-09": [(31.23, 121.47)],
    "CN-10": [(32.06, 118.80)], "CN-11": [(30.27, 120.16)], "CN-12": [(31.82, 117.23)],
    "CN-13": [(26.07, 119.30)], "CN-14": [(28.68, 115.86)], "CN-15": [(36.65, 117.12)],
    "CN-16": [(34.75, 113.63)], "CN-17": [(30.59, 114.31)], "CN-18": [(28.23, 112.94)],
    "CN-19": [(23.13, 113.26)], "CN-20": [(22.82, 108.37)], "CN-21": [(20.04, 110.20)],
    "CN-22": [(29.56, 106.55)], "CN-23": [(30.57, 104.07)], "CN-24": [(26.65, 106.63)],
    "CN-25": [(25.04, 102.71)], "CN-26": [(29.65, 91.14)], "CN-27": [(34.34, 108.94)],
    "CN-28": [(36.06, 103.83)], "CN-29": [(36.62, 101.78)], "CN-30": [(38.49, 106.23)],
    "CN-31": [(43.83, 87.62)],
    "TW-N": [(25.03, 121.57)], "TW-C": [(24.15, 120.67)], "TW-S": [(22.63, 120.30)],
    "TW-E": [(23.99, 121.60)],
    "TH-N": [(18.79, 98.98)], "TH-S": [(7.01, 100.47)], "TH-BKK": [(13.76, 100.50)],
    "TH-C": [(14.35, 100.57), (13.36, 100.98)], "TH-NE": [(16.44, 102.84)],
    # --- Africa ---
    "DZ": [(36.75, 3.06), (35.70, -0.63), (36.37, 6.61)],
}


def fetch(lat, lon):
    p = os.path.join(OUT, f"{lat:.2f}_{lon:.2f}.json")
    if os.path.exists(p) and os.path.getsize(p) > 10000:
        return p
    url = ("https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M&community=RE"
           f"&longitude={lon}&latitude={lat}&start=20180101&end=20260928&format=JSON&time-standard=UTC")
    for k in range(5):
        r = subprocess.run(["curl", "-sS", "--max-time", "180", "-o", p, "-w", "%{http_code}", url],
                           capture_output=True, text=True)
        if r.stdout.strip() == "200" and os.path.getsize(p) > 10000:
            return p
        time.sleep(2 ** (k + 1))
    raise RuntimeError(f"failed {lat},{lon}")


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = pd.date_range("2018-01-01", "2026-09-29", freq="h", tz="UTC", inclusive="left")
    cols = {}
    for code, pts in POINTS.items():
        series = []
        for lat, lon in pts:
            d = json.load(open(fetch(lat, lon)))["properties"]["parameter"]["T2M"]
            s = pd.Series(d, dtype="float64")
            s.index = pd.to_datetime(s.index, format="%Y%m%d%H", utc=True)
            series.append(s.where(s > -900).reindex(idx))
        cols[code] = pd.concat(series, axis=1).mean(axis=1)
        print(code, len(pts), round(float(cols[code].mean()), 1), flush=True)
    df = pd.DataFrame(cols, index=idx)
    df.to_parquet(os.path.join(ROOT, "data", "interim", "weather_t2m.parquet"))
    print("areas", df.shape[1], "missing share", float(df.isna().mean().mean()), "last valid", df.dropna(how="all").index.max())


if __name__ == "__main__":
    main()
