"""
Simplified Python implementation of: Long-term Fairness in Ride-Hailing Platform

This is an educational/prototype implementation, not an exact reproduction of the paper.
It implements the main pipeline:
1. Load NYC taxi trip data
2. Convert pickup/drop-off coordinates into graph nodes
3. Train an MLP time-series request predictor
4. Use predicted requests to pre-train a centralised fair Q-allocation policy
5. Evaluate allocation using total utility, fairness variance, and normalised fairness

Expected data: NYC TLC Yellow Taxi March 2016 parquet/CSV.

Example:
    python long_term_fairness_python_impl.py \
        --data yellow_tripdata_2016-03.parquet \
        --sample 200000 \
        --drivers 50 \
        --max-nodes 60


python long_term_fairness_python_impl.py \
    --data yellow_tripdata_2016-03_coords.parquet \
    --sample 200000 \
    --drivers 50 \
    --max-nodes 60


Dependencies:
    pip install pandas numpy scikit-learn pyarrow
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# -----------------------------
# Geometry utilities
# -----------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def manhattan_bbox_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate Manhattan bounding box filter."""
    lon_min, lon_max = -74.03, -73.92
    lat_min, lat_max = 40.70, 40.88
    mask = (
        df["pu_lon"].between(lon_min, lon_max)
        & df["do_lon"].between(lon_min, lon_max)
        & df["pu_lat"].between(lat_min, lat_max)
        & df["do_lat"].between(lat_min, lat_max)
    )
    return df.loc[mask].copy()


def make_grid_node(lat: pd.Series, lon: pd.Series, grid_size: float) -> pd.Series:
    """Merge nearby locations into coarse grid nodes."""
    lat_bin = (lat / grid_size).round().astype(int)
    lon_bin = (lon / grid_size).round().astype(int)
    return lat_bin.astype(str) + "_" + lon_bin.astype(str)


# -----------------------------
# Data preparation
# -----------------------------

def load_taxi_data(path: str, sample: int | None = None, random_state: int = 7) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path_obj.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path_obj)
    else:
        raw = pd.read_csv(path_obj)

    if sample and len(raw) > sample:
        raw = raw.sample(sample, random_state=random_state)

    return raw


def standardise_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Handle common NYC TLC 2016 coordinate column names."""
    cols = {c.lower(): c for c in raw.columns}

    def pick(*names: str) -> str:
        for name in names:
            if name.lower() in cols:
                return cols[name.lower()]
        raise KeyError(f"None of these columns were found: {names}. Available: {list(raw.columns)}")

    pickup_time = pick("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime")
    pu_lon = pick("pickup_longitude", "start_lon", "pu_lon")
    pu_lat = pick("pickup_latitude", "start_lat", "pu_lat")
    do_lon = pick("dropoff_longitude", "end_lon", "do_lon", "dropoff_longitude")
    do_lat = pick("dropoff_latitude", "end_lat", "do_lat", "dropoff_latitude")

    out = pd.DataFrame(
        {
            "time": pd.to_datetime(raw[pickup_time], errors="coerce"),
            "pu_lon": pd.to_numeric(raw[pu_lon], errors="coerce"),
            "pu_lat": pd.to_numeric(raw[pu_lat], errors="coerce"),
            "do_lon": pd.to_numeric(raw[do_lon], errors="coerce"),
            "do_lat": pd.to_numeric(raw[do_lat], errors="coerce"),
        }
    )

    if "trip_distance" in cols:
        out["trip_distance"] = pd.to_numeric(raw[cols["trip_distance"]], errors="coerce")
    else:
        out["trip_distance"] = np.nan

    out = out.dropna(subset=["time", "pu_lon", "pu_lat", "do_lon", "do_lat"])
    out = out[(out["pu_lon"] != 0) & (out["do_lon"] != 0) & (out["pu_lat"] != 0) & (out["do_lat"] != 0)]
    return out


def build_requests(
    raw: pd.DataFrame,
    grid_size: float = 0.01,
    max_nodes: int = 60,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Create simplified request table and node coordinate dictionary."""
    df = standardise_columns(raw)
    df = manhattan_bbox_filter(df)

    df["pickup_node"] = make_grid_node(df["pu_lat"], df["pu_lon"], grid_size)
    df["dropoff_node"] = make_grid_node(df["do_lat"], df["do_lon"], grid_size)

    # Keep only most frequent pickup/drop-off nodes to keep Q-table manageable.
    top_nodes = (
        pd.concat([df["pickup_node"], df["dropoff_node"]])
        .value_counts()
        .head(max_nodes)
        .index
    )
    df = df[df["pickup_node"].isin(top_nodes) & df["dropoff_node"].isin(top_nodes)].copy()

    # Node centre coordinates.
    node_coords: Dict[str, Tuple[float, float]] = {}
    for node, g in pd.concat(
        [
            df[["pickup_node", "pu_lat", "pu_lon"]].rename(
                columns={"pickup_node": "node", "pu_lat": "lat", "pu_lon": "lon"}
            ),
            df[["dropoff_node", "do_lat", "do_lon"]].rename(
                columns={"dropoff_node": "node", "do_lat": "lat", "do_lon": "lon"}
            ),
        ]
    ).groupby("node"):
        node_coords[node] = (float(g["lat"].mean()), float(g["lon"].mean()))

    # OD utility as trip distance when available, otherwise haversine distance.
    distances = []
    for row in df.itertuples(index=False):
        if np.isfinite(row.trip_distance) and row.trip_distance > 0:
            # trip_distance is in miles in TLC data; convert to km roughly.
            distances.append(float(row.trip_distance) * 1.60934)
        else:
            distances.append(haversine_km(row.pu_lat, row.pu_lon, row.do_lat, row.do_lon))
    df["od_distance_km"] = distances

    keep = ["time", "pickup_node", "dropoff_node", "od_distance_km"]
    return df[keep].sort_values("time").reset_index(drop=True), node_coords


# -----------------------------
# MLP request prediction
# -----------------------------

def hourly_counts(requests: pd.DataFrame) -> pd.DataFrame:
    x = requests.copy()
    x["hour_ts"] = x["time"].dt.floor("h")
    return (
        x.groupby(["hour_ts", "pickup_node", "dropoff_node"])
        .size()
        .rename("count")
        .reset_index()
    )


def make_supervised_count_data(
    counts: pd.DataFrame,
    top_pairs: int = 200,
    lags: Tuple[int, ...] = (1, 2, 24),
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build supervised lag-feature data for OD hourly request counts."""
    pair_counts = (
        counts.groupby(["pickup_node", "dropoff_node"])["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_pairs)
    )
    pairs = pair_counts.index.tolist()

    if counts.empty:
        raise ValueError("No request counts available after preprocessing.")

    all_hours = pd.date_range(counts["hour_ts"].min(), counts["hour_ts"].max(), freq="h")
    pair_to_id = {pair: i for i, pair in enumerate(pairs)}

    rows = []
    for pu, do in pairs:
        series = (
            counts[(counts["pickup_node"] == pu) & (counts["dropoff_node"] == do)]
            .set_index("hour_ts")["count"]
            .reindex(all_hours, fill_value=0)
            .astype(float)
        )
        frame = pd.DataFrame({"hour_ts": all_hours, "count": series.values})
        for lag in lags:
            frame[f"lag_{lag}"] = frame["count"].shift(lag)
        frame["hour"] = frame["hour_ts"].dt.hour
        frame["dow"] = frame["hour_ts"].dt.dayofweek
        frame["pair_id"] = pair_to_id[(pu, do)]
        frame["pickup_node"] = pu
        frame["dropoff_node"] = do
        rows.append(frame.dropna())

    data = pd.concat(rows, ignore_index=True)
    feature_cols = [f"lag_{lag}" for lag in lags] + ["hour", "dow", "pair_id"]
    X = data[feature_cols]
    y = data["count"]
    meta = data[["hour_ts", "pickup_node", "dropoff_node", "count"]]
    return X, y, meta


def train_mlp_request_predictor(counts: pd.DataFrame) -> Tuple[object, List[Tuple[str, str]]]:
    X, y, meta = make_supervised_count_data(counts)
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64,), activation="relu", max_iter=200, random_state=7),
    )
    model.fit(X, y)
    pairs = list(meta[["pickup_node", "dropoff_node"]].drop_duplicates().itertuples(index=False, name=None))
    return model, pairs


def predict_future_counts(
    model: object,
    history_counts: pd.DataFrame,
    future_start: pd.Timestamp,
    future_end: pd.Timestamp,
    pairs: List[Tuple[str, str]],
    lags: Tuple[int, ...] = (1, 2, 24),
) -> pd.DataFrame:
    """Recursive future OD count prediction."""
    future_hours = pd.date_range(future_start.floor("h"), future_end.floor("h"), freq="h")
    all_rows = []

    pair_to_id = {pair: i for i, pair in enumerate(pairs)}
    hist_min = history_counts["hour_ts"].min().floor("h")
    hist_max = future_end.floor("h")
    all_hours = pd.date_range(hist_min, hist_max, freq="h")

    for pu, do in pairs:
        series = (
            history_counts[(history_counts["pickup_node"] == pu) & (history_counts["dropoff_node"] == do)]
            .set_index("hour_ts")["count"]
            .reindex(all_hours, fill_value=0)
            .astype(float)
        )
        # recursive prediction for each future hour
        for h in future_hours:
            feature = {}
            for lag in lags:
                feature[f"lag_{lag}"] = float(series.get(h - pd.Timedelta(hours=lag), 0.0))
            feature["hour"] = h.hour
            feature["dow"] = h.dayofweek
            feature["pair_id"] = pair_to_id[(pu, do)]
            X = pd.DataFrame([feature])
            pred = max(0.0, float(model.predict(X)[0]))
            series.loc[h] = pred
            all_rows.append(
                {
                    "hour_ts": h,
                    "pickup_node": pu,
                    "dropoff_node": do,
                    "pred_count": pred,
                }
            )

    return pd.DataFrame(all_rows)


def expand_predicted_requests(
    pred_counts: pd.DataFrame,
    node_coords: Dict[str, Tuple[float, float]],
    max_per_od_hour: int = 3,
) -> pd.DataFrame:
    """Convert predicted counts to pseudo requests for Q-policy pretraining."""
    rows = []
    for row in pred_counts.itertuples(index=False):
        n = int(min(round(row.pred_count), max_per_od_hour))
        if n <= 0:
            continue
        if row.pickup_node not in node_coords or row.dropoff_node not in node_coords:
            continue
        pu_lat, pu_lon = node_coords[row.pickup_node]
        do_lat, do_lon = node_coords[row.dropoff_node]
        dist = haversine_km(pu_lat, pu_lon, do_lat, do_lon)
        for _ in range(n):
            rows.append(
                {
                    "time": row.hour_ts,
                    "pickup_node": row.pickup_node,
                    "dropoff_node": row.dropoff_node,
                    "od_distance_km": dist,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["time", "pickup_node", "dropoff_node", "od_distance_km"])
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


# -----------------------------
# Centralised fair Q allocator
# -----------------------------

@dataclass
class Metrics:
    total_utility: float
    fairness_variance: float
    normalised_fairness: float
    min_utility: float
    mean_utility: float
    max_utility: float


class FairQAllocator:
    """
    Educational centralised fair Q-allocation model.

    It keeps one Q-table per driver and a central controller chooses the driver
    that maximises immediate utility + learned Q-value - fairness penalty.
    """

    def __init__(
        self,
        nodes: Iterable[str],
        node_coords: Dict[str, Tuple[float, float]],
        n_drivers: int = 50,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.05,
        fairness_lambda: float = 1.0,
        omega: float = 0.6,
        q_weight: float = 0.1,
        random_state: int = 7,
    ) -> None:
        self.nodes = list(nodes)
        self.node_coords = node_coords
        self.n_drivers = n_drivers
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.fairness_lambda = fairness_lambda
        self.omega = omega
        self.q_weight = q_weight
        self.rng = np.random.default_rng(random_state)

        if not self.nodes:
            raise ValueError("No nodes available for allocator.")
        self.driver_nodes = self.rng.choice(self.nodes, size=n_drivers, replace=True).tolist()
        self.utilities = np.zeros(n_drivers, dtype=float)
        self.q: Dict[Tuple[int, str, str, str], float] = defaultdict(float)

    def reset(self) -> None:
        self.driver_nodes = self.rng.choice(self.nodes, size=self.n_drivers, replace=True).tolist()
        self.utilities = np.zeros(self.n_drivers, dtype=float)

    def travel_cost(self, from_node: str, to_node: str) -> float:
        if from_node == to_node:
            return 0.0
        if from_node not in self.node_coords or to_node not in self.node_coords:
            return 0.0
        lat1, lon1 = self.node_coords[from_node]
        lat2, lon2 = self.node_coords[to_node]
        return haversine_km(lat1, lon1, lat2, lon2)

    def request_reward(self, driver_id: int, pickup_node: str, dropoff_node: str, od_distance_km: float) -> float:
        cost_to_pickup = self.travel_cost(self.driver_nodes[driver_id], pickup_node)
        return float(od_distance_km - cost_to_pickup)

    def scalar_score(self, driver_id: int, reward: float) -> float:
        new_utils = self.utilities.copy()
        new_utils[driver_id] += reward
        # Equivalent spirit of Eq. 7: utility minus scaled fairness variance.
        return float(reward - self.fairness_lambda * self.omega * np.var(new_utils))

    def choose_driver(self, pickup_node: str, dropoff_node: str, od_distance_km: float, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_drivers))

        scores = []
        for d in range(self.n_drivers):
            reward = self.request_reward(d, pickup_node, dropoff_node, od_distance_km)
            q_key = (d, self.driver_nodes[d], pickup_node, dropoff_node)
            score = self.scalar_score(d, reward) + self.q_weight * self.q[q_key]
            scores.append(score)
        return int(np.argmax(scores))

    def step(self, row, training: bool = True) -> None:
        pu = row.pickup_node
        do = row.dropoff_node
        dist = float(row.od_distance_km)

        driver = self.choose_driver(pu, do, dist, training=training)
        old_node = self.driver_nodes[driver]
        reward = self.request_reward(driver, pu, do, dist)
        old_key = (driver, old_node, pu, do)

        # TD update. Here next state is approximated by the driver's new location.
        if training:
            future_candidates = [self.q[(driver, do, pu2, do2)] for pu2 in self.nodes[:10] for do2 in self.nodes[:10]]
            next_best = max(future_candidates) if future_candidates else 0.0
            target = reward + self.gamma * next_best
            self.q[old_key] += self.alpha * (target - self.q[old_key])

        self.utilities[driver] += reward
        self.driver_nodes[driver] = do

    def run(self, requests: pd.DataFrame, training: bool = True, limit: int | None = None) -> Metrics:
        if limit:
            requests = requests.head(limit)
        for row in requests.itertuples(index=False):
            self.step(row, training=training)
        return self.metrics()

    def metrics(self) -> Metrics:
        total = float(np.sum(self.utilities))
        var = float(np.var(self.utilities))
        mean = float(np.mean(self.utilities))
        norm = float(np.std(self.utilities) / mean) if mean != 0 else float("nan")
        return Metrics(
            total_utility=total,
            fairness_variance=var,
            normalised_fairness=norm,
            min_utility=float(np.min(self.utilities)),
            mean_utility=mean,
            max_utility=float(np.max(self.utilities)),
        )


class GreedyUtilityAllocator(FairQAllocator):
    """Baseline: choose driver with maximum immediate utility only."""

    def choose_driver(self, pickup_node: str, dropoff_node: str, od_distance_km: float, training: bool = True) -> int:
        rewards = [self.request_reward(d, pickup_node, dropoff_node, od_distance_km) for d in range(self.n_drivers)]
        return int(np.argmax(rewards))


# -----------------------------
# Experiment runner
# -----------------------------

def split_train_test(requests: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Paper-inspired split. Fallback to 70/30 if requested dates are absent."""
    train_end = pd.Timestamp("2016-03-26 00:00:00")
    test_end = pd.Timestamp("2016-04-01 23:59:59")
    train = requests[requests["time"] < train_end].copy()
    test = requests[(requests["time"] >= train_end) & (requests["time"] <= test_end)].copy()

    if len(train) < 100 or len(test) < 100:
        cutoff = requests["time"].quantile(0.7)
        train = requests[requests["time"] < cutoff].copy()
        test = requests[requests["time"] >= cutoff].copy()

    return train.sort_values("time"), test.sort_values("time")


def print_metrics(name: str, m: Metrics) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Total utility       : {m.total_utility:,.3f}")
    print(f"Fairness variance   : {m.fairness_variance:,.3f}")
    print(f"Normalised fairness : {m.normalised_fairness:,.6f}")
    print(f"Min / Mean / Max    : {m.min_utility:,.3f} / {m.mean_utility:,.3f} / {m.max_utility:,.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to NYC taxi parquet or CSV file")
    parser.add_argument("--sample", type=int, default=200000, help="Sample rows to keep from raw data")
    parser.add_argument("--drivers", type=int, default=50, help="Number of simulated drivers")
    parser.add_argument("--max-nodes", type=int, default=60, help="Maximum graph nodes")
    parser.add_argument("--grid-size", type=float, default=0.01, help="Coordinate grid size")
    parser.add_argument("--train-limit", type=int, default=50000, help="Limit requests used for Q training")
    parser.add_argument("--test-limit", type=int, default=50000, help="Limit requests used for evaluation")
    args = parser.parse_args()

    print("Loading data...")
    raw = load_taxi_data(args.data, sample=args.sample)
    requests, node_coords = build_requests(raw, grid_size=args.grid_size, max_nodes=args.max_nodes)
    print(f"Prepared requests: {len(requests):,}; graph nodes: {len(node_coords):,}")

    train, test = split_train_test(requests)
    print(f"Train requests: {len(train):,}; Test requests: {len(test):,}")

    print("Training MLP request predictor...")
    counts = hourly_counts(train)
    predictor, pairs = train_mlp_request_predictor(counts)

    future_start = test["time"].min()
    future_end = test["time"].max()
    pred_counts = predict_future_counts(predictor, counts, future_start, future_end, pairs)
    predicted_requests = expand_predicted_requests(pred_counts, node_coords)
    print(f"Predicted pseudo requests for Q pretraining: {len(predicted_requests):,}")

    nodes = list(node_coords.keys())

    # Proposed method: pretrain with predicted future requests, then train on historical requests, test on actual future requests.
    proposed = FairQAllocator(
        nodes=nodes,
        node_coords=node_coords,
        n_drivers=args.drivers,
        gamma=0.9,
        fairness_lambda=1.0,
        omega=0.6,
        epsilon=0.05,
    )
    proposed.run(predicted_requests, training=True, limit=args.train_limit)
    proposed.run(train, training=True, limit=args.train_limit)
    # Keep learned Q table, reset driver state before evaluation for clean comparison.
    proposed.reset()
    proposed_metrics = proposed.run(test, training=False, limit=args.test_limit)

    # Baseline: utility-only greedy.
    greedy = GreedyUtilityAllocator(nodes=nodes, node_coords=node_coords, n_drivers=args.drivers)
    greedy_metrics = greedy.run(test, training=False, limit=args.test_limit)

    print_metrics("Proposed simplified fair Q allocation", proposed_metrics)
    print_metrics("Greedy utility-only baseline", greedy_metrics)


if __name__ == "__main__":
    main()
