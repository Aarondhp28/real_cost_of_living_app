# =============================================================
# build_preview_data.py
# ONE-TIME SETUP SCRIPT. Run it once (and again only if the raw
# climate files change). It combines the 4 NOAA climate files into
# one small file, county_extras.csv, that the app loads.
# =============================================================

import glob                  # finds files by pattern
import pandas as pd

# The 4 NOAA files in the raw folder, and the column name each becomes.
CLIMATE_FILES = {
    "raw/climate_avg_temp.csv": "avg_temp_2025",    # degrees F, all of 2025
    "raw/climate_precip.csv": "precip_2025",        # inches, all of 2025
    "raw/climate_july_high.csv": "july_high_2025",  # degrees F, July 2025
    "raw/climate_jan_low.csv": "jan_low_2025",      # degrees F, January 2025
}

# --- Step 1: a lookup from state abbreviation to state FIPS code -----
# NOAA labels counties like "AL-001" (state letters + county number).
# Our files use "01001" (state number + county number).
# The first 2 digits of our FIPS codes are the state number, so the
# name file tells us that "AL" = "01", "AK" = "02", and so on.
try:
    names = pd.read_csv("county_names.csv", dtype=str, sep=None,
                        engine="python", encoding="utf-8-sig")
except UnicodeDecodeError:
    names = pd.read_csv("county_names.csv", dtype=str, sep=None,
                        engine="python", encoding="cp1252")
state_code = (names.assign(st=names["fips"].str[:2])
                   .drop_duplicates("state_abbr")
                   .set_index("state_abbr")["st"])


def read_noaa(path, new_name):
    """Read one NOAA file and return a column of values indexed by FIPS."""
    # comment="#" skips NOAA's description lines at the top of the file.
    # encoding="latin-1" can read any file without crashing on letters
    # like the ñ in "Doña Ana County" (we only use the ID and Value columns).
    df = pd.read_csv(path, comment="#", dtype={"ID": str}, encoding="latin-1")

    # Print the file's title line so you can confirm it's the right file.
    with open(path, encoding="latin-1") as f:
        print(f"{path}:  {f.readline().strip()}")

    # Split "AL-001" into "AL" and "001", then build "01001".
    parts = df["ID"].str.split("-", expand=True)
    df["fips"] = parts[0].map(state_code) + parts[1].str.zfill(3)

    # Drop any rows whose state isn't in our name file (they can't match).
    df = df.dropna(subset=["fips"])

    return df.set_index("fips")["Value"].rename(new_name)


# --- Step 2: read all 4 files and line them up side by side ----------
extras = pd.concat([read_noaa(p, n) for p, n in CLIMATE_FILES.items()],
                   axis=1)
extras.index.name = "fips"

# --- Step 3: check how many of OUR counties got climate data ---------
ours = pd.read_csv("cs_scaled_2023.csv")["county"].astype(str).str.zfill(5)
matched = ours.isin(extras.index)
print(f"\nCounties in our data: {len(ours):,}")
print(f"Counties with climate data: {matched.sum():,}")
missing = ours[~matched]
if len(missing):
    # Show which states the unmatched counties are in.
    print("Missing, by state code:",
          missing.str[:2].value_counts().to_dict())

# --- Step 4: add 2023 population from the master panel ------------
# The master panel has one row per county per year (2022 and 2023).
# Keep only 2023, rebuild the 5-digit FIPS code, and take POPESTIMATE.
panel = pd.read_csv("Master_Panel_PerCapita.csv")
panel = panel[panel["Year"] == 2023]
panel["fips"] = panel["county"].astype(str).str.zfill(5)
population = panel.set_index("fips")["POPESTIMATE"]
print(f"Counties with 2023 population: {ours.isin(population.index).sum():,}")

# --- Step 4b: add map coordinates from the Census Gazetteer file -----
# Finds the Gazetteer counties file in the raw folder, whatever its
# exact name is (for example 2024_Gaz_counties_national.txt).
gaz_path = glob.glob("raw/*Gaz_counties*.txt")[0]
# sep="|" because the Census separates columns with | instead of commas.
gaz = pd.read_csv(gaz_path, sep="|", dtype={"GEOID": str}, encoding="latin-1")
gaz.columns = gaz.columns.str.strip()   # remove stray spaces in column names
coords = gaz.set_index("GEOID")[["INTPTLAT", "INTPTLONG"]]
print(f"Counties with map coordinates: {ours.isin(coords.index).sum():,}")

# --- Step 5: one row for EVERY county in our data, then save ---------
# reindex keeps all 3,021 counties. Counties NOAA doesn't cover get a
# blank (NaN) in the climate columns, but still get their population.
extras = extras.reindex(ours)
extras.index.name = "fips"
extras["population_2023"] = population.reindex(ours).values
extras["lat"] = coords["INTPTLAT"].reindex(ours).values    # latitude
extras["lon"] = coords["INTPTLONG"].reindex(ours).values   # longitude
# Round climate to 1 decimal; keep full precision for coordinates.
extras = extras.round({"avg_temp_2025": 1, "precip_2025": 1,
                       "july_high_2025": 1, "jan_low_2025": 1})
extras.to_csv("county_extras.csv")
print(f"\nSaved county_extras.csv with {len(extras):,} rows.")
