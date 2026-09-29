"""Income of every grid area, for the fairness audit.

All areas: GDP per capita at purchasing-power parity of the area's country
(World Bank WDI, NY.GDP.PCAP.PP.CD, latest year up to 2024; Taiwan from the IMF
World Economic Outlook datamapper) and the World Bank income group.
U.S. balancing authorities: per-capita personal income of the states they serve
(BEA via FRED, series <ST>PCPI, 2024), averaged with approximate
service-territory weights.
Output: results/income_by_area.csv
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "income")

ISO = {"US": "USA", "AU": "AUS", "CN": "CHN", "TH": "THA", "DZ": "DZA", "TW": "TWN",
       "AT": "AUT", "BE": "BEL", "BG": "BGR", "CH": "CHE", "CZ": "CZE", "DE": "DEU", "DK": "DNK",
       "EE": "EST", "ES": "ESP", "FI": "FIN", "FR": "FRA", "GR": "GRC", "HR": "HRV", "HU": "HUN",
       "IE": "IRL", "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "NL": "NLD", "NO": "NOR",
       "PL": "POL", "PT": "PRT", "RO": "ROU", "RS": "SRB", "SE": "SWE", "SI": "SVN", "SK": "SVK",
       "UK": "GBR", "BA": "BIH", "ME": "MNE", "MK": "MKD", "MD": "MDA"}

# approximate shares of each balancing authority's load by state
BA_STATES = {
    "SCL": {"WA": 1}, "TAL": {"FL": 1}, "TEC": {"FL": 1}, "TEPC": {"AZ": 1}, "TIDC": {"CA": 1},
    "TPWR": {"WA": 1}, "TVA": {"TN": .7, "AL": .15, "MS": .1, "KY": .05}, "AECI": {"MO": 1},
    "AZPS": {"AZ": 1}, "BANC": {"CA": 1}, "BPAT": {"WA": .5, "OR": .4, "ID": .1}, "CHPD": {"WA": 1},
    "CISO": {"CA": 1}, "CPLE": {"NC": .85, "SC": .15}, "CPLW": {"NC": 1}, "DOPD": {"WA": 1},
    "DUK": {"NC": .65, "SC": .35}, "EPE": {"TX": .8, "NM": .2}, "ERCO": {"TX": 1}, "FMPP": {"FL": 1},
    "FPL": {"FL": 1}, "GCPD": {"WA": 1}, "GVL": {"FL": 1}, "HST": {"FL": 1}, "IID": {"CA": 1},
    "IPCO": {"ID": .95, "OR": .05},
    "ISNE": {"MA": .45, "CT": .25, "NH": .1, "ME": .1, "RI": .07, "VT": .03}, "JEA": {"FL": 1},
    "LDWP": {"CA": 1}, "LGEE": {"KY": 1},
    "MISO": {"MI": .2, "MN": .15, "IL": .12, "IN": .12, "WI": .12, "LA": .1, "MO": .07, "IA": .06,
             "AR": .04, "MS": .02},
    "NEVP": {"NV": 1}, "NWMT": {"MT": 1}, "NYIS": {"NY": 1}, "PACE": {"UT": .7, "WY": .2, "ID": .1},
    "PACW": {"OR": .8, "WA": .1, "CA": .1}, "PGE": {"OR": 1},
    "PJM": {"PA": .2, "NJ": .14, "VA": .14, "OH": .14, "IL": .12, "MD": .1, "WV": .04, "DE": .03,
            "DC": .02, "KY": .02, "IN": .02, "MI": .02, "NC": .01},
    "PNM": {"NM": 1}, "PSCO": {"CO": 1}, "SC": {"SC": 1}, "SCEG": {"SC": 1},
    "SOCO": {"GA": .6, "AL": .35, "MS": .05},
    "SWPP": {"OK": .3, "KS": .25, "NE": .15, "TX": .1, "AR": .08, "LA": .05, "MO": .05, "NM": .02},
    "WALC": {"AZ": .8, "CA": .1, "NV": .1}, "AVA": {"WA": .8, "ID": .2},
}


def latest(series, year_max=2024):
    s = {int(k): v for k, v in series.items() if v is not None and int(k) <= year_max}
    y = max(s)
    return s[y], y


def main():
    wb = json.load(open(os.path.join(RAW, "wb_gdppc_ppp.json")))[1]
    gdp = {}
    for r in wb:
        gdp.setdefault(r["countryiso3code"], {})[r["date"]] = r["value"]
    gdp = {k: latest(v)[0] for k, v in gdp.items()}
    gdp["TWN"] = latest(json.load(open(os.path.join(RAW, "imf_twn.json")))["values"]["PPPPC"]["TWN"])[0]
    groups = {c["id"]: c["incomeLevel"]["value"] for c in json.load(open(os.path.join(RAW, "wb_countries.json")))[1]}
    groups["TWN"] = "High income"   # Taiwan is not classified by the World Bank; high income by IMF data

    pcpi = {}
    for f in os.listdir(RAW):
        if f.startswith("pcpi_"):
            s = pd.read_csv(os.path.join(RAW, f), index_col=0).iloc[:, 0]
            pcpi[f[5:7]] = float(s.loc[s.index <= "2024-12-31"].dropna().iloc[-1])

    d = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)
    rows = []
    for code in d["code"]:
        pre = code.split("-")[0]
        iso = ISO[code.split("-")[1]] if pre == "EU" else ISO[pre]
        row = dict(code=code, iso3=iso, gdp_pc_ppp=gdp[iso], income_group=groups[iso])
        if pre == "US":
            w = BA_STATES[code[3:]]
            tot = sum(w.values())
            row["us_state_pcpi"] = sum(pcpi[s] * v for s, v in w.items()) / tot
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "income_by_area.csv"), index=False)
    print(df.groupby("income_group").size())
    print(df[df.us_state_pcpi.notna()].us_state_pcpi.describe().round(0))


if __name__ == "__main__":
    main()
