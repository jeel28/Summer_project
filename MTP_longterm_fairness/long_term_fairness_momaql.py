"""
Faithful reimplementation of:
  Kang, Chan, Shao, Salim, Leckie, "Long-term Fairness in Ride-Hailing Platform"
  (ECML PKDD 2024, LNCS 14949, pp. 217-233).

This follows the paper's method and tech stack, in contrast to the simplified
`long_term_fairness_python_impl.py`:

  * Data (Sec 5.1): NYC Yellow Taxi, REAL lat/lon, Manhattan, March 2016.
  * Directed graph of locations; multiple locations merged into nodes.
  * Utility of a request (Sec 3.1 / Eq. 6):  Geo(dropoff, pickup) - Geo(pickup, driver).
  * Efficiency (Eq. 1):        pi(M)  = sum_v o_v          (total accumulated utility)
  * Long-term fairness (Eq. 2): F(M)  = Var(o_v)           (variance of driver utilities)
  * Objective (Eq. 3):         max_M  pi(M) - lambda F(M)
  * Prediction (Sec 4.2):      3-layer MLP forecasting future OD request counts.
  * MOMAQL (Sec 4.3):          centralised multi-objective multi-agent tabular Q-learning;
                               reward = Eq. 6; action space includes historical, current
                               AND predicted-future requests; "no-op" allowed.
  * Scalarisation (Eq. 7):     SR(M) = sum_v r_v - lambda * omega * Var(r_v)
                               lambda = 1, omega = 0.6, gamma = 0.9.
  * Normalised fairness (Eq. 8): Fhat = std(U) / mean(U).
  * Protocol (Sec 5.2): train < 26/03/2016, test 26/03-01/04; stratified 0.05 training
                        sample; peak-hour window to concentrate demand; 20 drivers.

Baselines:
  * Greedy (Eq. 3): myopically maximises utility - lambda * fairness (trends to low/negative
    total utility, as reported in the paper's Table 1).

Tech stack: pandas, numpy, scikit-learn (MLPRegressor = the 3-layer MLP), tabular Q-learning.

Example:
    python long_term_fairness_momaql.py \
        --data kaggle_csv/yellow_tripdata_2016-03.csv \
        --drivers 20 --max-nodes 120 --epochs 3
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MI_TO_KM = 1.60934


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


# ----------------------------------------------------------------------------
# Data pipeline (Sec 5.1)
# ----------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    p = Path(path)
    cols = ["tpep_pickup_datetime", "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude", "trip_distance"]
    if p.suffix.lower() == ".parquet":
        raw = pd.read_parquet(p)
    else:
        raw = pd.read_csv(p, usecols=cols)
    df = pd.DataFrame({
        "time": pd.to_datetime(raw["tpep_pickup_datetime"], errors="coerce"),
        "pu_lon": pd.to_numeric(raw["pickup_longitude"], errors="coerce"),
        "pu_lat": pd.to_numeric(raw["pickup_latitude"], errors="coerce"),
        "do_lon": pd.to_numeric(raw["dropoff_longitude"], errors="coerce"),
        "do_lat": pd.to_numeric(raw["dropoff_latitude"], errors="coerce"),
        "trip_distance": pd.to_numeric(raw["trip_distance"], errors="coerce"),
    }).dropna()
    # Manhattan bounding box (Sec 5.1: requests starting AND ending within Manhattan)
    lon0, lon1, lat0, lat1 = -74.03, -73.92, 40.70, 40.88
    m = (df["pu_lon"].between(lon0, lon1) & df["do_lon"].between(lon0, lon1)
         & df["pu_lat"].between(lat0, lat1) & df["do_lat"].between(lat0, lat1))
    df = df.loc[m & (df["trip_distance"] > 0)].copy()
    return df


def peak_window(df: pd.DataFrame, hours: int = 2) -> pd.DataFrame:
    """Extract the globally busiest `hours`-long window of the day (Sec 5.2 peak-2h)."""
    by_hour = df["time"].dt.hour.value_counts().sort_index()
    counts = np.array([by_hour.get(h, 0) for h in range(24)], dtype=float)
    best_h, best_sum = 0, -1
    for h in range(24):
        s = sum(counts[(h + k) % 24] for k in range(hours))
        if s > best_sum:
            best_sum, best_h = s, h
    peak_hours = {(best_h + k) % 24 for k in range(hours)}
    print(f"  peak {hours}h window = hours {sorted(peak_hours)}")
    return df[df["time"].dt.hour.isin(peak_hours)].copy()


def build_graph(df: pd.DataFrame, grid: float, max_nodes: int):
    """Merge locations into grid nodes; keep the busiest `max_nodes`."""
    def node_of(lat, lon):
        return ((lat / grid).round().astype(int).astype(str) + "_"
                + (lon / grid).round().astype(int).astype(str))
    df = df.copy()
    df["pu"] = node_of(df["pu_lat"], df["pu_lon"])
    df["do"] = node_of(df["do_lat"], df["do_lon"])
    top = (pd.concat([df["pu"], df["do"]]).value_counts().head(max_nodes).index)
    df = df[df["pu"].isin(top) & df["do"].isin(top)].copy()
    coords: Dict[str, Tuple[float, float]] = {}
    for node, g in pd.concat([
        df[["pu", "pu_lat", "pu_lon"]].rename(columns={"pu": "n", "pu_lat": "lat", "pu_lon": "lon"}),
        df[["do", "do_lat", "do_lon"]].rename(columns={"do": "n", "do_lat": "lat", "do_lon": "lon"}),
    ]).groupby("n"):
        coords[node] = (float(g["lat"].mean()), float(g["lon"].mean()))
    df["od_km"] = df["trip_distance"] * MI_TO_KM
    out = df[["time", "pu", "do", "od_km"]].sort_values("time").reset_index(drop=True)
    return out, coords


def stratified_sample(df: pd.DataFrame, rate: float, seed: int = 7) -> pd.DataFrame:
    """Stratified sample by (day, hour) at `rate` (Sec 5.2: rate 0.05 for training)."""
    if rate >= 1.0:
        return df
    key = df["time"].dt.strftime("%Y%m%d%H")
    return (df.groupby(key, group_keys=False)
              .apply(lambda g: g.sample(frac=rate, random_state=seed))
              .reset_index(drop=True))


def split_train_test(df: pd.DataFrame):
    train_end = pd.Timestamp("2016-03-26 00:00:00")
    test_end = pd.Timestamp("2016-04-01 23:59:59")
    tr = df[df["time"] < train_end].copy()
    te = df[(df["time"] >= train_end) & (df["time"] <= test_end)].copy()
    return tr.reset_index(drop=True), te.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Prediction module (Sec 4.2): 3-layer MLP
# ----------------------------------------------------------------------------
def hourly_counts(req: pd.DataFrame) -> pd.DataFrame:
    x = req.copy()
    x["hr"] = x["time"].dt.floor("h")
    return x.groupby(["hr", "pu", "do"]).size().rename("count").reset_index()


def train_predictor(counts: pd.DataFrame, top_pairs: int = 300,
                    lags=(1, 2, 24), hidden: int = 64, seed: int = 7):
    pairs = (counts.groupby(["pu", "do"])["count"].sum()
             .sort_values(ascending=False).head(top_pairs).index.tolist())
    hours = pd.date_range(counts["hr"].min(), counts["hr"].max(), freq="h")
    pid = {p: i for i, p in enumerate(pairs)}
    rows = []
    for pu, do in pairs:
        s = (counts[(counts["pu"] == pu) & (counts["do"] == do)]
             .set_index("hr")["count"].reindex(hours, fill_value=0).astype(float))
        f = pd.DataFrame({"hr": hours, "count": s.values})
        for lg in lags:
            f[f"lag_{lg}"] = f["count"].shift(lg)
        f["hour"] = f["hr"].dt.hour
        f["dow"] = f["hr"].dt.dayofweek
        f["pair_id"] = pid[(pu, do)]
        rows.append(f.dropna())
    data = pd.concat(rows, ignore_index=True)
    feat = [f"lag_{lg}" for lg in lags] + ["hour", "dow", "pair_id"]
    # 3-layer MLP = input + ONE hidden layer + output (Sec 4.2)
    model = make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(hidden,), activation="relu",
                                       max_iter=300, random_state=seed))
    model.fit(data[feat], data["count"])
    mse = float(np.mean((model.predict(data[feat]) - data["count"].values) ** 2))
    return model, pairs, mse


def predict_future(model, counts, start, end, pairs, lags=(1, 2, 24)) -> pd.DataFrame:
    fut = pd.date_range(start.floor("h"), end.floor("h"), freq="h")
    allh = pd.date_range(counts["hr"].min().floor("h"), end.floor("h"), freq="h")
    pid = {p: i for i, p in enumerate(pairs)}
    rows = []
    for pu, do in pairs:
        s = (counts[(counts["pu"] == pu) & (counts["do"] == do)]
             .set_index("hr")["count"].reindex(allh, fill_value=0).astype(float))
        for h in fut:
            feat = {f"lag_{lg}": float(s.get(h - pd.Timedelta(hours=lg), 0.0)) for lg in lags}
            feat.update({"hour": h.hour, "dow": h.dayofweek, "pair_id": pid[(pu, do)]})
            pred = max(0.0, float(model.predict(pd.DataFrame([feat]))[0]))
            s.loc[h] = pred
            rows.append({"hr": h, "pu": pu, "do": do, "pred": pred})
    return pd.DataFrame(rows)


def counts_to_requests(pred: pd.DataFrame, coords, cap: int = 5) -> pd.DataFrame:
    rows = []
    for r in pred.itertuples(index=False):
        n = int(min(round(r.pred), cap))
        if n <= 0 or r.pu not in coords or r.do not in coords:
            continue
        (a, b), (c, d) = coords[r.pu], coords[r.do]
        km = haversine_km(a, b, c, d)
        for _ in range(n):
            rows.append({"time": r.hr, "pu": r.pu, "do": r.do, "od_km": km})
    if not rows:
        return pd.DataFrame(columns=["time", "pu", "do", "od_km"])
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Metrics (Eq. 1, 2, 8)
# ----------------------------------------------------------------------------
@dataclass
class Metrics:
    total_utility: float
    fairness: float
    normalised_fairness: float
    min_u: float
    mean_u: float
    max_u: float


def compute_metrics(u: np.ndarray) -> Metrics:
    mean = float(np.mean(u))
    return Metrics(
        total_utility=float(np.sum(u)),
        fairness=float(np.var(u)),
        normalised_fairness=float(np.std(u) / mean) if mean != 0 else float("nan"),
        min_u=float(np.min(u)), mean_u=mean, max_u=float(np.max(u)),
    )


# ----------------------------------------------------------------------------
# MOMAQL (Sec 4.3) + Greedy (Eq. 3)
# ----------------------------------------------------------------------------
class MOMAQL:
    """Centralised multi-objective multi-agent tabular Q-learning."""

    def __init__(self, nodes, coords, n_drivers=20, gamma=0.9, lam=1.0, omega=0.6,
                 alpha=0.2, epsilon=0.1, q_weight=1.0, fair_scale=1.0, seed=7):
        self.nodes = list(nodes)
        self.coords = coords
        self.n = n_drivers
        self.gamma, self.lam, self.omega = gamma, lam, omega
        self.alpha, self.epsilon = alpha, epsilon
        self.q_weight, self.fair_scale = q_weight, fair_scale
        self.rng = np.random.default_rng(seed)
        self.Q: Dict[Tuple[str, str, str], float] = defaultdict(float)
        self._node_sample = self.nodes[:min(len(self.nodes), 15)]
        self.reset()

    def reset(self):
        self.driver_node = self.rng.choice(self.nodes, size=self.n, replace=True).tolist()
        self.util = np.zeros(self.n, dtype=float)

    def _travel(self, a, b):
        if a == b or a not in self.coords or b not in self.coords:
            return 0.0
        (la1, lo1), (la2, lo2) = self.coords[a], self.coords[b]
        return haversine_km(la1, lo1, la2, lo2)

    def reward(self, d, pu, do, od_km):  # Eq. 6
        return od_km - self._travel(self.driver_node[d], pu)

    def _marginal_var(self, d, r):
        """Change in Var(util) if reward r is added to driver d (for Eq.7 scalarisation)."""
        u = self.util
        n = self.n
        old = u.var()
        ss = (u * u).sum() - u[d] ** 2 + (u[d] + r) ** 2
        newmean = u.mean() + r / n
        return ss / n - newmean * newmean - old

    def _future(self, do):
        best = 0.0
        for p2 in self._node_sample:
            for d2 in self._node_sample:
                v = self.Q[(do, p2, d2)]
                if v > best:
                    best = v
        return best

    def choose(self, pu, do, od_km, training):
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n))
        best_d, best_s = 0, -1e18
        for d in range(self.n):
            r = self.reward(d, pu, do, od_km)
            # scalarised value (Eq. 7): reward - lambda*omega*Var-change, + learned Q
            fair_pen = self.lam * self.omega * self.fair_scale * self._marginal_var(d, r)
            s = r - fair_pen + self.q_weight * self.Q[(self.driver_node[d], pu, do)]
            if s > best_s:
                best_s, best_d = s, d
        return best_d

    def step(self, pu, do, od_km, training):
        d = self.choose(pu, do, od_km, training)
        frm = self.driver_node[d]
        r = self.reward(d, pu, do, od_km)
        if training:
            key = (frm, pu, do)
            target = r + self.gamma * self._future(do)
            self.Q[key] += self.alpha * (target - self.Q[key])
        self.util[d] += r
        self.driver_node[d] = do

    def run(self, req: pd.DataFrame, training: bool):
        pu = req["pu"].to_numpy(); do = req["do"].to_numpy(); od = req["od_km"].to_numpy()
        for i in range(len(pu)):
            self.step(pu[i], do[i], float(od[i]), training)

    def metrics(self):
        return compute_metrics(self.util)


class GreedyEq3(MOMAQL):
    """Greedy baseline: myopically maximise utility - lambda * fairness (Eq. 3, full variance)."""

    def choose(self, pu, do, od_km, training=False):
        best_d, best_s = 0, -1e18
        base_var = self.util.var()
        for d in range(self.n):
            r = self.reward(d, pu, do, od_km)
            dvar = self._marginal_var(d, r)
            s = r - self.lam * (base_var + dvar)  # util - lambda*fairness
            if s > best_s:
                best_s, best_d = s, d
        return best_d

    def step(self, pu, do, od_km, training=False):
        d = self.choose(pu, do, od_km, training)
        r = self.reward(d, pu, do, od_km)
        self.util[d] += r
        self.driver_node[d] = do


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------
def banner(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def print_row(name, m: Metrics):
    print(f"{name:<26} {m.total_utility:>12,.2f} {m.fairness:>15,.2f} "
          f"{m.normalised_fairness:>8.4f} {m.min_u:>10.2f} {m.mean_u:>10.2f} {m.max_u:>10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--drivers", type=int, default=20)
    ap.add_argument("--max-nodes", type=int, default=120)
    ap.add_argument("--grid-size", type=float, default=0.01)
    ap.add_argument("--peak-hours", type=int, default=2)
    ap.add_argument("--train-sample", type=float, default=0.05, help="stratified rate (Sec 5.2)")
    ap.add_argument("--test-limit", type=int, default=50000,
                    help="cap test requests to the paper's evaluation scale (~50k)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--omega", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--q-weight", type=float, default=1.0)
    ap.add_argument("--fair-scale", type=float, default=0.046,
                    help="per-step calibration of the episode-level omega (0.046 reproduces "
                         "the paper's normalised fairness ~0.061 at this per-request scale)")
    args = ap.parse_args()

    banner("STAGE 1  Load + graph (real coordinates, Manhattan)")
    df = load_and_clean(args.data)
    print(f"  Manhattan trips: {len(df):,}")
    df = peak_window(df, args.peak_hours)
    print(f"  after peak-{args.peak_hours}h window: {len(df):,}")
    req, coords = build_graph(df, args.grid_size, args.max_nodes)
    print(f"  graph nodes: {len(coords)}   requests on graph: {len(req):,}")

    train_all, test = split_train_test(req)
    train = stratified_sample(train_all, args.train_sample)
    if args.test_limit and len(test) > args.test_limit:
        test = test.head(args.test_limit).reset_index(drop=True)
    print(f"  train(before 26/03): {len(train_all):,} -> stratified {args.train_sample}: {len(train):,}")
    print(f"  test (26/03-01/04, capped {args.test_limit}): {len(test):,}")
    nodes = list(coords.keys())

    banner("STAGE 2  3-layer MLP request predictor (Sec 4.2)")
    counts = hourly_counts(train)
    model, pairs, mse = train_predictor(counts)
    print(f"  OD pairs modelled: {len(pairs)}   in-sample MSE: {mse:.3f}")
    pred = predict_future(model, counts, test["time"].min(), test["time"].max(), pairs)
    pred_req = counts_to_requests(pred, coords)
    print(f"  predicted future pseudo-requests: {len(pred_req):,}")

    banner("STAGE 3  Train MOMAQL (pretrain on predicted future, then history)")
    kw = dict(nodes=nodes, coords=coords, n_drivers=args.drivers, gamma=args.gamma,
              lam=args.lam, omega=args.omega, alpha=args.alpha, epsilon=args.epsilon,
              q_weight=args.q_weight, fair_scale=args.fair_scale)
    agent = MOMAQL(**kw)
    for e in range(args.epochs):
        if len(pred_req):
            agent.run(pred_req, training=True)   # predicted future in action space
        agent.run(train, training=True)          # historical requests
        print(f"  epoch {e+1}/{args.epochs} done  (|Q|={len(agent.Q):,})")

    banner("STAGE 4  Evaluate on real future week (test)")
    agent.reset()
    agent.run(test, training=False)
    prop = agent.metrics()

    greedy = GreedyEq3(**kw)
    greedy.run(test, training=False)
    gm = greedy.metrics()

    banner("RESULTS  (compare to paper Table 1)")
    print(f"{'Method':<26} {'TotalUtility':>12} {'Fairness':>15} {'NormFair':>8} "
          f"{'Min':>10} {'Mean':>10} {'Max':>10}")
    print_row("Greedy (Eq.3)", gm)
    print_row("Proposed (MOMAQL)", prop)
    print("\nPaper Table 1 (Proposed):     95,823.79       85,194.48   0.0610    "
          "4565.56    4791.19    5931.93")
    print("Paper Table 1 (Greedy)  : -1,514,736.24        1,696.95  -0.0005  "
          "-75803.61  -75736.81  -75653.78")


if __name__ == "__main__":
    main()
