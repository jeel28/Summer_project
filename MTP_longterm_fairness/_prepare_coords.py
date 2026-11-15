"""Bridge the zone-based TLC parquet to the coordinate-based format the original
long_term_fairness_python_impl.py expects, WITHOUT modifying that script.

The 2016 TLC yellow-taxi re-release stores pickup/drop-off as taxi-zone IDs
(PULocationID/DOLocationID) instead of the GPS longitude/latitude the paper and
the script assume. We map each zone ID to its shapefile centroid (lat/lon) and
emit a parquet whose columns match what standardise_columns() picks:
    tpep_pickup_datetime, pickup_longitude, pickup_latitude,
    dropoff_longitude, dropoff_latitude, trip_distance
Rows in zones without geometry (264/265 "Unknown") are dropped.
"""
import pandas as pd

SRC = "yellow_tripdata_2016-03.parquet"
DST = "yellow_tripdata_2016-03_coords.parquet"
CENTROIDS = "taxi_zone_centroids.csv"

print("Loading zone centroids...")
cent = pd.read_csv(CENTROIDS)[["LocationID", "lat", "lon"]]
pu = cent.rename(columns={"LocationID": "PULocationID", "lat": "pickup_latitude", "lon": "pickup_longitude"})
do = cent.rename(columns={"LocationID": "DOLocationID", "lat": "dropoff_latitude", "lon": "dropoff_longitude"})

print("Loading taxi parquet...")
df = pd.read_parquet(SRC, columns=["tpep_pickup_datetime", "trip_distance", "PULocationID", "DOLocationID"])
n0 = len(df)
print(f"  rows: {n0:,}")

df = df.merge(pu, on="PULocationID", how="inner")
df = df.merge(do, on="DOLocationID", how="inner")
n1 = len(df)
print(f"  rows after centroid join (drops Unknown zones 264/265): {n1:,}  (removed {n0 - n1:,})")

out = df[[
    "tpep_pickup_datetime",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "trip_distance",
]].copy()

print("Writing coordinate-based parquet...")
out.to_parquet(DST, index=False)
print(f"wrote {DST} with {len(out):,} rows and columns: {list(out.columns)}")
print(out.head(3).to_string())
