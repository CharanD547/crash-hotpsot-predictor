"""
Streamlit dashboard for the Traffic Crash Hotspot Project.

Run from the project root:
    streamlit run app/dashboard.py
"""

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Find project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Cleaned data file
DATA_FILE = PROJECT_ROOT / "data" / "cleaned_crashes.csv"

st.set_page_config(
    page_title="Traffic Crash Hotspot Dashboard",
    layout="wide"
)


@st.cache_data
def load_data():
    """
    Load the cleaned crash data.
    The parse_dates part makes sure Datetime is treated as a real date column.
    """
    return pd.read_csv(DATA_FILE, parse_dates=["Datetime"])


def make_hotspots(filtered_df):
    """
    Create simple hotspot areas by rounding latitude and longitude.

    Example:
    36.854322 becomes 36.85
    -76.288626 becomes -76.29

    Crashes with nearby rounded coordinates get grouped together.
    """
    hotspot_df = filtered_df.copy()

    hotspot_df["lat_area"] = hotspot_df["Latitude"].round(2)
    hotspot_df["lon_area"] = hotspot_df["Longitude"].round(2)

    hotspots = (
        hotspot_df.groupby(["lat_area", "lon_area"])
        .agg(
            crash_count=("Document Number", "count"),
            fatalities=("Number of Fatalities", "sum"),
            injured=("Number of People Injured", "sum"),
            serious_injuries=("Number of People with Suspected Serious Injury", "sum"),
            severity_score=("Severity Score", "sum"),
        )
        .reset_index()
    )

    # Beginner-friendly risk score.
    # You can improve this later with machine learning.
    hotspots["risk_score"] = (
        hotspots["crash_count"]
        + hotspots["fatalities"] * 10
        + hotspots["serious_injuries"] * 5
        + hotspots["injured"] * 2
    )

    hotspots = hotspots.sort_values("risk_score", ascending=False)

    return hotspots


def add_crash_heatmap(m, filtered_df, max_points):
    """
    Add a heatmap layer to the map.
    The heatmap shows where crashes are dense.
    """
    map_df = filtered_df.dropna(subset=["Latitude", "Longitude"])

    # Sampling keeps the app fast
    if len(map_df) > max_points:
        map_df = map_df.sample(max_points, random_state=42)

    heat_data = map_df[["Latitude", "Longitude"]].values.tolist()

    if heat_data:
        HeatMap(
            heat_data,
            radius=12,
            blur=18,
            min_opacity=0.25
        ).add_to(m)


def add_hotspot_markers(m, hotspots, limit=15):
    """
    Add circles for the top hotspot areas.
    Bigger circles usually mean more crashes.
    """
    for _, row in hotspots.head(limit).iterrows():
        radius = min(35, 6 + row["crash_count"] * 0.25)

        popup = (
            f"<b>Hotspot Area</b><br>"
            f"Crashes: {int(row['crash_count'])}<br>"
            f"Fatalities: {int(row['fatalities'])}<br>"
            f"People Injured: {int(row['injured'])}<br>"
            f"Risk Score: {int(row['risk_score'])}"
        )

        folium.CircleMarker(
            location=[row["lat_area"], row["lon_area"]],
            radius=radius,
            popup=popup,
            fill=True,
            fill_opacity=0.65,
            weight=2,
        ).add_to(m)


def add_crash_points(m, filtered_df, max_points):
    """
    Add individual crash dots to the map.
    This can slow down the app, so we limit how many points appear.
    """
    point_df = filtered_df.dropna(subset=["Latitude", "Longitude"])

    if len(point_df) > max_points:
        point_df = point_df.sample(max_points, random_state=10)

    for _, row in point_df.iterrows():
        popup = (
            f"<b>{row['Crash Severity']}</b><br>"
            f"Date: {row['Datetime']}<br>"
            f"Road: {row['Route or Street Name']}<br>"
            f"Weather: {row['Weather Condition']}<br>"
            f"Light: {row['Light Condition']}<br>"
            f"Fatalities: {int(row['Number of Fatalities'])}<br>"
            f"Injured: {int(row['Number of People Injured'])}"
        )

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=3,
            popup=popup,
            fill=True,
            fill_opacity=0.7,
            weight=1,
        ).add_to(m)


# ----------------------------
# Page title
# ----------------------------
st.title("Traffic Crash Hotspot Dashboard")

st.write(
    "This dashboard uses crash-level data to show crash patterns, "
    "map hotspots, and rank high-risk areas using a simple scoring method."
)

# Stop if cleaned data does not exist
if not DATA_FILE.exists():
    st.error(
        "Missing data/cleaned_crashes.csv. "
        "Run this command first from the project root: python scripts/clean_data.py"
    )
    st.stop()

# Load cleaned data
df = load_data()

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

available_years = sorted(df["Year"].dropna().unique())

selected_years = st.sidebar.multiselect(
    "Year",
    available_years,
    default=available_years[-3:] if len(available_years) >= 3 else available_years,
)

severity_options = sorted(df["Crash Severity"].dropna().unique())

selected_severity = st.sidebar.multiselect(
    "Crash Severity",
    severity_options,
    default=severity_options,
)

weather_options = sorted(df["Weather Condition"].dropna().unique())

selected_weather = st.sidebar.multiselect(
    "Weather Condition",
    weather_options,
    default=weather_options,
)

light_options = sorted(df["Light Condition"].dropna().unique())

selected_light = st.sidebar.multiselect(
    "Light Condition",
    light_options,
    default=light_options,
)

hour_range = st.sidebar.slider(
    "Hour of Day",
    0,
    23,
    (0, 23)
)

show_points = st.sidebar.checkbox(
    "Show individual crash points",
    value=False
)

max_heatmap_points = st.sidebar.slider(
    "Max heatmap points",
    500,
    10000,
    5000,
    step=500
)

max_marker_points = st.sidebar.slider(
    "Max crash markers",
    100,
    3000,
    750,
    step=100
)

# ----------------------------
# Apply filters
# ----------------------------
filtered = df[
    df["Year"].isin(selected_years)
    & df["Crash Severity"].isin(selected_severity)
    & df["Weather Condition"].isin(selected_weather)
    & df["Light Condition"].isin(selected_light)
    & df["Hour"].between(hour_range[0], hour_range[1])
].copy()

if filtered.empty:
    st.warning("No crashes match the selected filters. Try changing the sidebar filters.")
    st.stop()

# ----------------------------
# Main metrics
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Crashes",
    f"{len(filtered):,}"
)

col2.metric(
    "Fatalities",
    f"{int(filtered['Number of Fatalities'].sum()):,}"
)

col3.metric(
    "People Injured",
    f"{int(filtered['Number of People Injured'].sum()):,}"
)

col4.metric(
    "Serious Injuries",
    f"{int(filtered['Number of People with Suspected Serious Injury'].sum()):,}"
)

# ----------------------------
# Map
# ----------------------------
st.subheader("Crash Hotspot Map")

center_lat = filtered["Latitude"].median()
center_lon = filtered["Longitude"].median()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="OpenStreetMap"
)

hotspots = make_hotspots(filtered)

add_crash_heatmap(m, filtered, max_heatmap_points)
add_hotspot_markers(m, hotspots, limit=15)

if show_points:
    add_crash_points(m, filtered, max_marker_points)

st_folium(m, width=None, height=600)

# ----------------------------
# Hotspot table
# ----------------------------
st.subheader("Top Hotspot Areas")

st.write(
    "Hotspots are created by grouping nearby crashes using rounded latitude and longitude. "
    "The risk score is a beginner formula based on crash count, fatalities, and injuries."
)

hotspot_display = hotspots[
    [
        "lat_area",
        "lon_area",
        "crash_count",
        "fatalities",
        "injured",
        "serious_injuries",
        "risk_score",
    ]
].head(15)

st.dataframe(hotspot_display, use_container_width=True)

# ----------------------------
# Charts
# ----------------------------
st.subheader("Crash Trends")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("Crashes by Year")

    crashes_by_year = (
        filtered.groupby("Year")
        .size()
        .reset_index(name="Crashes")
    )

    st.bar_chart(
        crashes_by_year,
        x="Year",
        y="Crashes"
    )

with chart_col2:
    st.write("Crashes by Hour")

    crashes_by_hour = (
        filtered.groupby("Hour")
        .size()
        .reset_index(name="Crashes")
    )

    st.bar_chart(
        crashes_by_hour,
        x="Hour",
        y="Crashes"
    )

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.write("Crashes by Severity")

    severity_counts = (
        filtered["Crash Severity"]
        .value_counts()
        .reset_index()
    )

    severity_counts.columns = ["Crash Severity", "Crashes"]

    st.bar_chart(
        severity_counts,
        x="Crash Severity",
        y="Crashes"
    )

with chart_col4:
    st.write("Crashes by Weather")

    weather_counts = (
        filtered["Weather Condition"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    weather_counts.columns = ["Weather Condition", "Crashes"]

    st.bar_chart(
        weather_counts,
        x="Weather Condition",
        y="Crashes"
    )

# ----------------------------
# Road analysis
# ----------------------------
st.subheader("Top Roads by Crash Count")

top_roads = (
    filtered["Route or Street Name"]
    .value_counts()
    .head(15)
    .reset_index()
)

top_roads.columns = ["Road / Street", "Crash Count"]

st.dataframe(top_roads, use_container_width=True)

# ----------------------------
# Raw data preview
# ----------------------------
with st.expander("See filtered crash records"):
    preview_cols = [
        "Datetime",
        "Route or Street Name",
        "Crash Severity",
        "Weather Condition",
        "Light Condition",
        "Roadway Surface Condition",
        "Intersection Type",
        "Latitude",
        "Longitude",
        "Number of Fatalities",
        "Number of People Injured",
    ]

    st.dataframe(
        filtered[preview_cols].head(500),
        use_container_width=True
    )