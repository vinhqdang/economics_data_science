"""Download the non-U.S. raw data for the global panel (all public, no API key).

Europe     Energy-Charts API (Fraunhofer ISE): national load and cross-border
           trading, 15-min, 2018-2026, 33 countries
Oceania    AEMO price-and-demand files: 5 NEM regions, 5/30-min, 2018-2026
Asia       Zenodo 8322210 (China, 31 provinces, hourly, 2018)
           Zenodo 7537890 (Taiwan grid areas, 10-min, 2017-2022)
           Zenodo 17109911 (Thailand, hourly, 2023-2024)
Africa     Mendeley z5x2d3mhw7 (Algeria, Sonelgaz, hourly, 2008-2020)
Files are written to data/raw/global/.
"""
import os
import json
import time
import subprocess

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "raw", "global")
EU = ["at", "be", "bg", "ch", "cz", "de", "dk", "ee", "es", "fi", "fr", "gr", "hr", "hu",
      "ie", "it", "lt", "lu", "lv", "nl", "no", "pl", "pt", "ro", "rs", "se", "si", "sk",
      "uk", "ba", "me", "mk", "md"]
NEM = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]
FILES = {
    "china_load.csv": "https://zenodo.org/api/records/8322210/files/Appendix%201_Hourly%20electric%20power%20load%20final.csv/content",
    "taiwan_loadarea.csv": "https://zenodo.org/api/records/7537890/files/loadarea_10min_2017Jan_2022Jun.csv/content",
    "thailand_2023.csv": "https://zenodo.org/api/records/17109911/files/system_2023.csv/content",
    "thailand_2024.csv": "https://zenodo.org/api/records/17109911/files/system_2024.csv/content",
    "algeria.xlsx": "https://data.mendeley.com/public-files/datasets/z5x2d3mhw7/files/049d33a5-6d4f-465a-8a65-b613487eb1c9/file_downloaded",
}


def get(url, path, tries=4):
    for k in range(tries):
        r = subprocess.run(["curl", "-sSL", "--max-time", "180", "-o", path, "-w", "%{http_code}", url],
                           capture_output=True, text=True)
        if r.stdout.strip() == "200" and os.path.getsize(path) > 500:
            return True
        time.sleep(2 ** (k + 1))
    return False


def main():
    os.makedirs(os.path.join(OUT, "europe"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "aemo"), exist_ok=True)
    for name, url in FILES.items():
        p = os.path.join(OUT, name)
        if not os.path.exists(p):
            print(name, get(url, p), flush=True)
    for c in EU:
        for y in range(2018, 2027):
            p = os.path.join(OUT, "europe", f"{c}_{y}.json")
            if os.path.exists(p):
                continue
            end = f"{y + 1}-01-01" if y < 2026 else "2026-09-28"
            ok = get(f"https://api.energy-charts.info/public_power?country={c}&start={y}-01-01&end={end}", p)
            if not ok:
                print("failed", c, y, flush=True)
        print("europe", c, flush=True)
    for reg in NEM:
        for y in range(2018, 2027):
            for m in range(1, 13):
                if y == 2026 and m > 8:
                    break
                p = os.path.join(OUT, "aemo", f"{reg}_{y}{m:02d}.csv")
                if os.path.exists(p):
                    continue
                url = f"https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{y}{m:02d}_{reg}.csv"
                if not get(url, p):
                    print("failed", reg, y, m, flush=True)
        print("aemo", reg, flush=True)


if __name__ == "__main__":
    main()
