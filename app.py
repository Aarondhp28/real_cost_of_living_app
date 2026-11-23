# =============================================================
# app.py - "Real Cost of Living" County Finder
# BLOCK 1: Load the data files and check that they match up
# =============================================================

# --- The tools this app uses ---------------------------------
import json                  # reads the .json settings files
import pandas as pd          # works with tables, like Excel inside Python
import streamlit as st       # turns this Python script into a web page

# Our team's scoring function from matching.py.
# Importing it now proves the app can find that file.
from matching import quality_score

# --- Page setup ----------------------------------------------
# Sets the browser tab title and uses the full width of the screen.
st.set_page_config(page_title="Real Cost of Living", layout="wide")
st.title("Real Cost of Living: County Finder")


# --- Load the data -------------------------------------------
# @st.cache_data tells Streamlit: read these files ONCE and remember them.
# Without it, the app would re-read every file each time a user clicks something.
@st.cache_data
def load_data():
    # 1. The scaled scores: one row per county, one column per feature.
    z = pd.read_csv("cs_scaled_2023.csv")

    # The county codes lost their leading zeros (1001 instead of 01001).
    # This turns them into text and pads them back to 5 digits
    # so they match the name file.
    z["county"] = z["county"].astype(str).str.zfill(5)
    z = z.set_index("county")   # use the county code as the row label

    # 2. The county names.
    #    dtype=str          -> keep everything as text, so the zeros stay
    #    sep=None           -> let pandas detect commas OR tabs on its own
    #    encoding=utf-8-sig -> ignore a hidden marker Excel sometimes adds
    # Text files can be saved in different "encodings" (ways of storing
    # letters like ñ). First try UTF-8, the modern standard. If that fails,
    # try Windows' older default (cp1252), which Notepad and Excel often use.
    try:
        names = pd.read_csv("county_names.csv", dtype=str, sep=None,
                            engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        names = pd.read_csv("county_names.csv", dtype=str, sep=None,
                            engine="python", encoding="cp1252")
    names = names.set_index("fips")

    # 3. The two settings files from the analysis.
    with open("cat_map.json") as f:
        cat_map = json.load(f)       # which category each feature belongs to
    with open("direction.json") as f:
        direction = json.load(f)     # +1 higher is better, -1 lower is better, 0 ignored

    return z, names, cat_map, direction


z, names, cat_map, direction = load_data()


# =============================================================
# BLOCK 2: Let the user rank their priorities
# =============================================================
from streamlit_sortables import sort_items   # the drag-and-drop add-on

# The names users see on screen (left) matched to the category
# names used in cat_map.json (right). To reword what users see,
# change only the left side. The right side must match the data files.
PRIORITY_LABELS = {
    "Healthcare access": "Healthcare",
    "Affordable real estate": "Housing",
    "Business growth & opportunity": "Business",
    "Education funding": "Education",
    "Transportation accessibility": "Transportation",
}


def rank_weights(n, style):
    """Turn a ranking (1st, 2nd, 3rd...) into weights that add up to 100%."""
    if style == "Balanced":
        # Linear: 1st gets 5 points, 2nd gets 4 ... last gets 1.
        raw = [n - i for i in range(n)]
    else:
        # "Rank-order centroid": a standard method where the top
        # priority gets a much bigger share than the rest.
        raw = [sum(1 / k for k in range(i + 1, n + 1)) for i in range(n)]
    total = sum(raw)
    return [r / total for r in raw]   # divide by the total so it adds to 100%


# Everything inside "with st.sidebar:" appears in the left-hand panel.
with st.sidebar:
    st.header("1. Rank your priorities")
    st.caption("Drag to reorder. Top = most important to you.")

    # Shows the draggable list. 'ranked' is the list in the user's order.
    ranked = sort_items(list(PRIORITY_LABELS.keys()), direction="vertical")

    st.header("2. How much should rank matter?")
    style = st.radio(
        "Weighting style",
        ["Balanced", "Top priority dominates"],
        help="Balanced: weights drop off gently. "
             "Top priority dominates: your #1 choice counts for almost half.",
    )

# Work out the weights and pair each one with its data category.
weights = rank_weights(len(ranked), style)
cat_weights = {PRIORITY_LABELS[label]: w for label, w in zip(ranked, weights)}

# --- Show the weights (temporary, so we can check them) -------
st.subheader("Your priority weights")
st.dataframe(
    pd.DataFrame({
        "Rank": range(1, len(ranked) + 1),
        "Priority": ranked,
        "Weight": [f"{w:.0%}" for w in weights],   # e.g. 0.333 -> "33%"
    }),
    hide_index=True,
)

# =============================================================
# BLOCK 3: Score every county and show the best matches
# =============================================================

# Show the user's current weights and the results-count slider in the sidebar.
with st.sidebar:
    st.caption("Your weights: " + ", ".join(
        f"{label} {w:.0%}" for label, w in zip(ranked, weights)))

    st.header("3. How many results?")
    top_n = st.slider("Number of counties to show", min_value=10,
                      max_value=20, value=10)


@st.cache_data
def category_percentiles(z, cat_map, direction):
    """Score each county in each of the 5 categories, as a percentile.

    These are the same steps quality_score() in matching.py uses, stopped
    before the weights are applied. They don't depend on the user's
    choices, so they're calculated once and remembered.
    """
    # Keep only features that count (+1 or -1). Skip the 0s.
    scored = [f for f in z.columns if direction[f] != 0]
    # Flip lower-is-better features so that higher always means better.
    signed = z[scored] * pd.Series({f: direction[f] for f in scored})
    # Average the features inside each category.
    cats = pd.Series({f: cat_map[f] for f in scored})
    cat_scores = signed.T.groupby(cats).mean().T
    # Percentile: 90 means "better than about 90% of counties."
    return cat_scores.rank(pct=True) * 100


cat_pct = category_percentiles(z, cat_map, direction)

# Overall score for every county, using the team's own function.
overall = quality_score(z, cat_map, direction, cat_weights)

# Rescale to 0-100: best county = 100, worst county = 0.
match_score = (overall - overall.min()) / (overall.max() - overall.min()) * 100

# The FIPS codes of the top counties, best first.
# --- Population floor: hide counties below a minimum size ------
@st.cache_data
def load_population():
    """2023 population for each county, from county_extras.csv."""
    pop = pd.read_csv("county_extras.csv", dtype={"fips": str})
    return pop.set_index("fips")["population_2023"]


population = load_population()

# A slider with fixed stops, shown in the sidebar.
min_pop = st.sidebar.select_slider(
    "Minimum county population",
    options=[0, 10_000, 25_000, 50_000, 100_000, 250_000],
    value=0,
    format_func=lambda n: "No minimum" if n == 0 else f"{n:,}",
)

# Keep only counties at or above the minimum, then take the best ones.
eligible = overall[population.reindex(overall.index) >= min_pop]
top = eligible.nlargest(top_n).index

# Build the results table one column at a time.
results = pd.DataFrame({
    "Rank": range(1, top_n + 1),
    "County": names.loc[top, "county_name"].values,
    "State": names.loc[top, "state_abbr"].values,
    "FIPS": top,
    "Match score": match_score[top].round().astype(int).values,
})
# Add one percentile column per priority, in the user's ranked order.
for label in ranked:
    results[label] = cat_pct.loc[top, PRIORITY_LABELS[label]].round().astype(int).values

# Show the scores as progress bars from 0 to 100, which are easier to read.
bar_columns = ["Match score"] + ranked
st.subheader(f"Your top {top_n} counties")
event = st.dataframe(
    results,                        # the table to show
    hide_index=True,                # hide pandas' row numbers (we have Rank)
    on_select="rerun",              # rerun the app when a row is clicked
    selection_mode="single-row",    # only one county can be selected at a time
    use_container_width=True,       # stretch the table across the page
    column_config={
        # Show each score column as a 0-100 progress bar.
        col: st.column_config.ProgressColumn(col, min_value=0,
                                             max_value=100, format="%d")
        for col in bar_columns
    },
)
st.caption("Match score: 100 = best overall fit for your priorities. "
           "Priority columns: percentile, so 90 = better than 90% of counties.")

# =============================================================
# BLOCK 4a: County preview card (click a row in the table)
# =============================================================

@st.cache_data
def load_clean():
    """Real (unscaled) values, so the preview can show dollars and percents."""
    clean = pd.read_csv("cs_clean_2023.csv")
    clean["county"] = clean["county"].astype(str).str.zfill(5)  # restore leading zeros
    return clean.set_index("county")


clean = load_clean()


def housing_label(vacancy_pct):
    """Turn a vacancy rate (a percent) into a plain-English label."""
    if vacancy_pct < 10:
        return "Tight housing market"
    elif vacancy_pct <= 20:
        return "Balanced"
    else:
        return "High vacancy"


# Which county to preview: the row the user clicked,
# or the #1 county if nothing has been clicked yet.
selected_rows = event.selection.rows
row = selected_rows[0] if selected_rows else 0
fips = results.iloc[row]["FIPS"]
info = clean.loc[fips]          # this county's real values

# Vacancy rate = vacant homes / all homes, as a percent.
vacancy = (info["ACS_vacant_units_PerCapita"]
           / info["ACS_housing_units_PerCapita"] * 100)

# border=True draws a box around everything inside this section.
card = st.container(border=True)   # the preview box; later blocks add to it
with card:
    st.subheader(f"{names.loc[fips, 'county_name']}, {names.loc[fips, 'state']}")
    st.caption("Click any row in the table above to preview a different county.")

    # Four headline numbers, side by side.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Match score", f"{match_score[fips]:.0f} / 100")
    c2.metric("Median home value (Census)",
              f"${info['ACS_median_home_value']:,.0f}")
    c3.metric("Typical home value (Zillow)",
              f"${info['ZVHI_Median_Annual_Home_Value']:,.0f}")
    c4.metric("School spending per student",
              f"${info['EFD_Panel.Spend_Per_Student']:,.0f}")

    # Housing availability label.
    st.markdown(f"**Housing:** {housing_label(vacancy)} "
                f"({vacancy:.1f}% of homes vacant)")
    st.caption("Vacant homes include seasonal and vacation homes, so a high "
               "rate doesn't always mean homes are for sale or rent.")

    # One bar per priority, in the order the user ranked them.
    st.markdown("**How this county ranks on your priorities**")
    for label in ranked:
        pct = int(round(cat_pct.loc[fips, PRIORITY_LABELS[label]]))
        st.progress(pct, text=f"{label}: percentile {pct}")

# =============================================================
# BLOCK 4b: Climate section inside the preview card
# =============================================================

@st.cache_data
def load_extras():
    """Climate values made by build_preview_data.py (one row per county)."""
    # dtype={"fips": str} keeps the leading zeros in the county codes.
    extras = pd.read_csv("county_extras.csv", dtype={"fips": str})
    return extras.set_index("fips")


extras = load_extras()

# "with card:" adds this inside the existing preview box.
with card:
    st.markdown("**Climate (2025)**")

    # Only 3,015 of 3,021 counties have NOAA climate data,
    # so check that this county is in the file before showing numbers.
    if pd.notna(extras.loc[fips, "avg_temp_2025"]):   # skip counties with blank climate data
        cl = extras.loc[fips]               # this county's climate row
        k1, k2, k3, k4 = st.columns(4)      # four numbers side by side
        k1.metric("Average temperature", f"{cl['avg_temp_2025']:.1f} °F")
        k2.metric("July average high", f"{cl['july_high_2025']:.1f} °F")
        k3.metric("January average low", f"{cl['jan_low_2025']:.1f} °F")
        k4.metric("Total precipitation", f"{cl['precip_2025']:.1f} in")
        st.caption("Source: NOAA Climate at a Glance. Values are for 2025 "
                   "only, so one unusually hot, cold or wet year shows here.")
    else:
        # Hawaii, Washington DC and one Virginia city aren't in NOAA's data.
        st.info("Climate data isn't available for this county. NOAA's "
                "county data covers the contiguous U.S. only.")

# =============================================================
# BLOCK 4a-2: Extra details inside the preview card
# =============================================================

# Each row: (what users see, column name in cs_clean_2023.csv, unit)
# Doctors are per 100,000 residents because they're rare.
# Everything else is per 1,000 residents.
DETAILS = [
    ("Primary care doctors", "AHRF_phys_nf_prim_care_pc_exc_rsdt_PerCapita",
     "per 100,000 residents"),
    ("Hospital-based doctors", "AHRF_md_nf_pc_hosp_all_PerCapita",
     "per 100,000 residents"),
    ("Business establishments", "BUSGROW_establishments_PerCapita",
     "per 1,000 residents"),
    ("Jobs located in the county", "BUSGROW_employment_PerCapita",
     "per 1,000 residents"),
    ("Live and work in this county", "CCT_within_PerCapita",
     "per 1,000 residents"),
    ("Commute in from other counties", "CCT_inbound_PerCapita",
     "per 1,000 residents"),
    ("Commute out to other counties", "CCT_outbound_PerCapita",
     "per 1,000 residents"),
]

# "with card:" adds this inside the existing preview box.
with card:
    # An expander is a collapsible section: click it to open or close.
    with st.expander("More details"):
        detail_table = pd.DataFrame({
            "Measure": [d[0] for d in DETAILS],
            # This county's value, rounded to 1 decimal place.
            "This county": [f"{info[d[1]]:,.1f}" for d in DETAILS],
            # The median: the middle value across all 3,021 counties.
            "Typical county": [f"{clean[d[1]].median():,.1f}" for d in DETAILS],
            "Unit": [d[2] for d in DETAILS],
        })
        st.dataframe(detail_table, hide_index=True, use_container_width=True)
        st.caption("Typical county = the middle (median) value of all 3,021 counties.")

# =============================================================
# BLOCK 5: "How this works" - a plain-English explanation
# =============================================================

st.divider()   # a thin line separating the app from the explanation

# A collapsible section at the bottom of the page.
with st.expander("How this works"):

    # Builds a line like "Healthcare access 33%, Affordable real estate 27%..."
    current = ", ".join(f"{label} **{w:.0%}**" for label, w in zip(ranked, weights))

    # TEAM: fill in the three [SOURCE NEEDED] spots in the "Where the data
    # comes from" list with the real dataset names before presenting.
    st.markdown(f"""
### 1. Your ranking becomes weights
The order you drag the priorities into is turned into percentages that add
up to 100%. Higher on the list means a bigger share.

- **Balanced:** 33%, 27%, 20%, 13%, 7%
- **Top priority dominates:** 46%, 26%, 16%, 9%, 4%

Your current weights: {current}

### 2. Every county is scored in each priority
We start with 2023 data for 3,021 U.S. counties. Each measure is converted
to a common scale that shows how far above or below the typical county it
is, so dollars, doctors and jobs can be compared fairly. Measures where
**lower is better**, like home prices, are flipped so that a higher score
always means better. The measures within each priority are then averaged.

### 3. The Match score combines them
Each county's five priority scores are combined using your weights. The
result is rescaled so that the **best-matching county scores 100** and the
worst scores 0.

### 4. The priority columns are percentiles
A **90** means the county does better on that priority than about 90% of
all counties.

### What each priority measures
- **Healthcare access:** doctors per 100,000 residents (primary care,
  hospital-based, medical specialists and surgical specialists)
- **Affordable real estate:** median home values from the Census and
  Zillow. Lower prices score higher.
- **Business growth & opportunity:** business establishments and jobs
  per 1,000 residents
- **Education funding:** public school spending per student
- **Transportation accessibility:** residents who both live and work in
  the county, per 1,000 residents

### Limits to keep in mind
- Transportation uses a single commuting measure. It doesn't yet capture
  public transit.
- Affordable real estate looks at home prices only, not incomes or rents.
- Per-resident numbers can look extreme in small counties and independent
  cities, where a few hospitals or employers serve a much larger area.
- Climate figures cover 2025 only, so one unusual year shows up as-is.
- Counties whose geographic codes changed (for example, Connecticut's)
  aren't included.

### Where the data comes from
- **Home values:** U.S. Census Bureau American Community Survey (ACS);
  Zillow Home Value Index (ZHVI)
- **Housing units and vacancy:** U.S. Census Bureau ACS
- **Doctors:** HRSA Area Health Resources Files (AHRF)
- **Businesses and jobs:** [SOURCE NEEDED]
- **School spending:** [SOURCE NEEDED]
- **Commuting:** [SOURCE NEEDED]
- **Climate:** NOAA National Centers for Environmental Information,
  Climate at a Glance
""")