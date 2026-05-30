"""
Clean the raw Traffic_Crashes.csv file and create data/cleaned_crashes.csv.

Run from the project root:
    python scripts/clean_data.py
"""

from pathlib import Path
import pandas as pd

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input and output files
RAW_FILE = PROJECT_ROOT / "data" / "Traffic_Crashes.csv"
CLEAN_FILE = PROJECT_ROOT / "data" / "cleaned_crashes.csv"


def main():
    # Make sure the raw file exists
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_FILE}. "
            "Make sure Traffic_Crashes.csv is inside the data folder."
        )

    print("Loading raw crash data...")
    df = pd.read_csv(RAW_FILE)

    print(f"Raw rows: {len(df):,}")
    print(f"Raw columns: {len(df.columns)}")

    # Convert Datetime column into real datetime format
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    # Convert Latitude and Longitude into numbers
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # These columns should be numeric
    numeric_cols = [
        "Number of Vehicles Involved",
        "Number of Fatalities",
        "Number of People with Suspected Serious Injury",
        "Number of People with Suspected Minor Injury",
        "Number of People with Possible Injury",
        "Number of People Injured",
        "Number of Pedestrian Fatalities",
        "Number of Pedestrians Injured",
        "Maximum Speed Difference",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Remove rows with missing date, latitude, or longitude
    df = df.dropna(subset=["Datetime", "Latitude", "Longitude"])

    # Remove impossible or bad coordinates
    # These bounds roughly cover Virginia and nearby areas.
    df = df[
        (df["Latitude"].between(36.0, 40.5))
        & (df["Longitude"].between(-84.5, -75.0))
    ]

    # Create useful time columns
    df["Year"] = df["Datetime"].dt.year
    df["Month"] = df["Datetime"].dt.month
    df["Month Name"] = df["Datetime"].dt.month_name()
    df["Day"] = df["Datetime"].dt.day
    df["Day Name"] = df["Datetime"].dt.day_name()
    df["Hour"] = df["Datetime"].dt.hour
    df["Is Weekend"] = df["Day Name"].isin(["Saturday", "Sunday"])

    # Create a beginner-friendly severity score.
    # This is not an official score. It is just for this project.
    df["Severity Score"] = (
        df["Number of Fatalities"] * 10
        + df["Number of People with Suspected Serious Injury"] * 5
        + df["Number of People with Suspected Minor Injury"] * 2
        + df["Number of People with Possible Injury"] * 1
    )

    # Fill missing category values
    category_cols = [
        "Crash Severity",
        "Weather Condition",
        "Light Condition",
        "Roadway Surface Condition",
        "Intersection Type",
        "Route or Street Name",
        "Type of Collision",
        "Alcohol Involved",
        "Speeding",
        "Night Crash",
    ]

    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    # Save cleaned data
    df.to_csv(CLEAN_FILE, index=False)

    print(f"Cleaned rows: {len(df):,}")
    print(f"Saved cleaned file to: {CLEAN_FILE}")


if __name__ == "__main__":
    main()