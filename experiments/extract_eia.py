"""Stream the EIA-930 bulk file (EBA.zip) and keep, for every balancing
authority (BA), hourly demand (D), the BA's own day-ahead demand forecast (DF)
and total interchange (TI).  Output: data/interim/eia_{D,DF,TI}.parquet with a
UTC hourly index and one column per BA.
"""
import io
import json
import os
import re
import zipfile
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
PAT = re.compile(r'^\{"series_id":"EBA\.([A-Z0-9]+)-ALL\.(D|DF|TI)\.H"')


def main():
    out = {"D": {}, "DF": {}, "TI": {}}
    names = {}
    with zipfile.ZipFile(os.path.join(ROOT, "data", "raw", "EBA.zip")) as z:
        with z.open("EBA.txt") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                m = PAT.match(line)
                if not m:
                    continue
                rec = json.loads(line)
                ba, kind = m.group(1), m.group(2)
                if "data" not in rec:
                    continue
                s = pd.Series({k: v for k, v in rec["data"]}, dtype="float64")
                s.index = pd.to_datetime(s.index, format="%Y%m%dT%H", utc=True)
                out[kind][ba] = s.sort_index()
                if kind == "D":
                    names[ba] = rec["name"]
    os.makedirs(os.path.join(ROOT, "data", "interim"), exist_ok=True)
    for kind, d in out.items():
        df = pd.DataFrame(d).sort_index()
        df.to_parquet(os.path.join(ROOT, "data", "interim", f"eia_{kind}.parquet"))
        print(kind, df.shape, df.index.min(), df.index.max())
    pd.Series(names).to_csv(os.path.join(ROOT, "data", "interim", "ba_names.csv"))


if __name__ == "__main__":
    main()
