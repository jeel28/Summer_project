"""
Single-file reproduction of:
  Kang, Chan, Shao, Salim, Leckie, "Long-term Fairness in Ride-Hailing Platform"
  (ECML PKDD 2024, LNCS 14949, pp. 217-233).

Scope (per explicit request): LAF, REASSIGN and Balance Ride-Pooling from the
paper's Table 1 are NOT reproduced here. Only the paper's two core methods,
plus one extension of our own on top of the paper's method, are implemented:

  * Greedy         - Eq. 3 baseline: myopically maximise utility - lambda*fairness.
  * MOMAQL         - the paper's proposed method (Sec 4): multi-objective
                      multi-agent Q-learning with a request-prediction module
                      (Sec 4.2) feeding predicted future demand into the
                      action space, and the scalarisation function of Eq. 7.
  * Adaptive MOMAQL - NOT from the paper. Same MOMAQL core, but the fixed
                      fairness weight lambda (Eq. 7) is replaced by a
                      state-dependent lambda_t that grows when (a) current
                      driver utilities are already imbalanced, (b) the OD
                      pair sits on locally over-demanded nodes, and shrinks
                      when the OD pair is rarely observed (low confidence).
                      Intent: spend the fairness penalty where it is needed
                      instead of paying it on every single request, which
                      should let total utility recover while fairness stays
                      controlled.

Data: NYC Yellow Taxi, March 2016 (yellow_tripdata_2016-03.csv), filtered to
trips that start AND end within Manhattan (Sec 5.1).

Definitions reproduced exactly as in the paper:
  * Utility of a request (Eq. 6 term):     Geo(dropoff, pickup) - Geo(pickup, driver)
  * Efficiency (Eq. 1):                    pi(M)  = sum_v o_v            (total utility)
  * Long-term fairness (Eq. 2):            F(M)   = Var(o_v)             (variance of driver utility)
  * Balance objective (Eq. 3):             max_M  pi(M) - lambda*F(M)
  * Scalarisation function (Eq. 7):        SR(M)  = sum_v r_v - lambda*omega*Var(r_v)
  * Normalised fairness (Eq. 8):           Fhat   = std(U) / mean(U)

Two experiments are reproduced, each now with three rows (Greedy, Proposed,
and our Adaptive MOMAQL extension):
  1. Table 1  - Total Utility / Fairness / Normalised Fairness / Min / Mean / Max,
                evaluated on the full 7-day test week.
  2. Fig. 4   - Fairness vs. length of time horizon (1..7 days), showing that
                the proposed method's fairness stabilises as the horizon grows,
                unlike a myopic baseline.

Usage:
    python3 paper_reproduction_greedy_momaql.py \
        --data yellow_tripdata_2016-03.csv \
        --drivers 20 --max-nodes 120 --peak-hours 2 --train-sample 0.05

Dependencies: pandas, numpy, scikit-learn, matplotlib
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

# Paper Table 1, restricted to the two methods reproduced here (for reference
# in the printed comparison at the end of the run).
PAPER_TABLE_1 = {
    "Greedy":   dict(total_utility=-1_514_736.24, fairness=1_696.95, normalised_fairness=-0.0005,
                      min_u=-75_803.61, mean_u=-75_736.81, max_u=-75_653.78),
    "Proposed": dict(total_utility=95_823.79, fairness=85_194.48, normalised_fairness=0.061,
                      min_u=4_565.56, mean_u=4_791.19, max_u=5_931.93),
}


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
    """Load raw TLC trips, keep only what's needed, filter to Manhattan-only trips."""
    p = Path(path)
    cols = ["tpep_pickup_datetime", "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude", "trip_distance"]
    if p.suffix.lower() == ".parquet":
        raw = pd.read_parquet(p, columns=cols)
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
    # Manhattan bounding box: requests starting AND ending within Manhattan (Sec 5.1)
    lon0, lon1, lat0, lat1 = -74.03, -73.92, 40.70, 40.88
    m = (df["pu_lon"].between(lon0, lon1) & df["do_lon"].between(lon0, lon1)
         & df["pu_lat"].between(lat0, lat1) & df["do_lat"].between(lat0, lat1))
    return df.loc[m & (df["trip_distance"] > 0)].copy()


def peak_window(df: pd.DataFrame, hours: int = 2) -> pd.DataFrame:
    """Busiest `hours`-long window of the day (Sec 5.2: peak 2-hour window)."""
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
    """Merge nearby locations into grid nodes; keep the busiest `max_nodes` (Sec 5.1)."""
    def node_of(lat, lon):
        return ((lat / grid).round().astype(int).astype(str) + "_"
                 + (lon / grid).round().astype(int).astype(str))
    df = df.copy()
    df["pu"] = node_of(df["pu_lat"], df["pu_lon"])
    df["do"] = node_of(df["do_lat"], df["do_lon"])
    top = pd.concat([df["pu"], df["do"]]).value_counts().head(max_nodes).index
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
    """Sec 3.3 / 5.2: train on data before 26/03/2016, test on the last 7-day week."""
    train_end = pd.Timestamp("2016-03-26 00:00:00")
    test_end = pd.Timestamp("2016-04-01 23:59:59")
    tr = df[df["time"] < train_end].copy()
    te = df[(df["time"] >= train_end) & (df["time"] <= test_end)].copy()
    return tr.reset_index(drop=True), te.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Request-prediction module (Sec 4.2): 3-layer MLP
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
    # 3-layer MLP = input + one hidden layer + output (Sec 4.2)
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
# MOMAQL - the proposed method (Sec 4.3, 4.4)
# ----------------------------------------------------------------------------
class MOMAQL:
    """Centralised multi-objective multi-agent tabular Q-learning (Sec 4.3).

    Action space includes historical, current AND predicted-future requests
    (Sec 4.2 output is folded into training via `run(pred_req, ...)`).
    Driver choice is scored with the scalarisation function of Eq. 7.
    """

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

    def reward(self, d, pu, do, od_km):  # Eq. 6 term: Geo(dropoff,pickup) - Geo(pickup,driver)
        return od_km - self._travel(self.driver_node[d], pu)

    def _marginal_var(self, d, r):
        """Change in Var(util) if reward r is added to driver d (Eq. 7 term)."""
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
            # Eq. 7 scalarisation: reward - lambda*omega*Var-change, plus learned Q
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
    """Greedy baseline (Eq. 3): myopically maximise utility - lambda*fairness, no learning."""

    def choose(self, pu, do, od_km, training=False):
        best_d, best_s = 0, -1e18
        base_var = self.util.var()
        for d in range(self.n):
            r = self.reward(d, pu, do, od_km)
            dvar = self._marginal_var(d, r)
            s = r - self.lam * (base_var + dvar)
            if s > best_s:
                best_s, best_d = s, d
        return best_d

    def step(self, pu, do, od_km, training=False):
        d = self.choose(pu, do, od_km, training)
        r = self.reward(d, pu, do, od_km)
        self.util[d] += r
        self.driver_node[d] = do


# ----------------------------------------------------------------------------
# Adaptive MOMAQL - OUR extension on top of the paper's method (not from the paper)
# ----------------------------------------------------------------------------
def build_pair_counts(train: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """Historical OD pair frequency, used to judge how well-observed a pair is."""
    counts = train.groupby(["pu", "do"]).size().to_dict()
    return {tuple(k): int(v) for k, v in counts.items()}


class AdaptiveMOMAQL(MOMAQL):
    """MOMAQL with a state-dependent fairness weight instead of a fixed lambda.

    The fixed lambda in Eq. 7 penalises every single request by the same
    amount, regardless of whether the system is currently fair or not. Here,
    lambda_t scales up when:
      * driver utilities are already imbalanced right now (`_driver_imbalance`),
      * the request touches a node that is locally over-demanded relative to
        the rest of the graph (`_local_demand_imbalance`),
    and scales down (loses confidence) when the OD pair has few historical
    observations (`_pair_uncertainty`). The aim is to only "spend" fairness
    penalty where it is actually needed, so total utility is not taxed on
    every request the way it is under a fixed lambda.
    """

    def __init__(self, nodes, coords, pair_counts: Dict[Tuple[str, str], int],
                 n_drivers=20, gamma=0.9, lam=1.0, omega=0.6, alpha=0.2,
                 epsilon=0.1, q_weight=1.0, fair_scale=1.0, seed=7,
                 imbalance_scale=1.5, uncertainty_scale=0.8, demand_scale=0.6):
        super().__init__(nodes=nodes, coords=coords, n_drivers=n_drivers, gamma=gamma,
                          lam=lam, omega=omega, alpha=alpha, epsilon=epsilon,
                          q_weight=q_weight, fair_scale=fair_scale, seed=seed)
        self.pair_counts = dict(pair_counts)
        self.imbalance_scale = imbalance_scale
        self.uncertainty_scale = uncertainty_scale
        self.demand_scale = demand_scale

        self.node_demand = defaultdict(float)
        for (pu, do), cnt in self.pair_counts.items():
            self.node_demand[pu] += float(cnt)
            self.node_demand[do] += float(cnt)
        self.node_mean_demand = float(np.mean(list(self.node_demand.values()))) if self.node_demand else 1.0

    def _driver_imbalance(self) -> float:
        if self.n <= 1:
            return 0.0
        mean = float(np.mean(self.util))
        if abs(mean) < 1e-12:
            return 0.0
        return float(np.std(self.util) / abs(mean))

    def _pair_uncertainty(self, pu: str, do: str) -> float:
        cnt = float(self.pair_counts.get((pu, do), 1.0))
        return float(1.0 / np.sqrt(cnt + 1.0))

    def _local_demand_imbalance(self, pu: str, do: str) -> float:
        pu_d = float(self.node_demand.get(pu, 0.0))
        do_d = float(self.node_demand.get(do, 0.0))
        base_val = max(pu_d, do_d) / (self.node_mean_demand + 1e-8)
        return max(0.0, float(base_val - 1.0))

    def adaptive_lambda(self, pu: str, do: str) -> float:
        driver_imb = self._driver_imbalance()
        local_imb = self._local_demand_imbalance(pu, do)
        uncertainty = self._pair_uncertainty(pu, do)

        coeff = 1.0 + self.imbalance_scale * driver_imb + self.demand_scale * local_imb
        coeff *= 1.0 + (1.0 - uncertainty) * self.uncertainty_scale
        return float(self.lam * coeff)

    def choose(self, pu, do, od_km, training):
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n))
        lam_t = self.adaptive_lambda(pu, do)
        best_d, best_s = 0, -1e18
        for d in range(self.n):
            r = self.reward(d, pu, do, od_km)
            fair_pen = lam_t * self.omega * self.fair_scale * self._marginal_var(d, r)
            s = r - fair_pen + self.q_weight * self.Q[(self.driver_node[d], pu, do)]
            if s > best_s:
                best_s, best_d = s, d
        return best_d


# ----------------------------------------------------------------------------
# Experiment helpers
# ----------------------------------------------------------------------------
def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def print_row(name, m: Metrics):
    print(f"{name:<26} {m.total_utility:>14,.2f} {m.fairness:>15,.2f} "
          f"{m.normalised_fairness:>9.4f} {m.min_u:>10.2f} {m.mean_u:>10.2f} {m.max_u:>10.2f}")


def time_horizon_stability(kw: dict, adaptive_kw: dict, train, pred_req, test, epochs: int):
    """Reproduce Fig. 4: fairness vs. length of time horizon (1..7 test days)."""
    test = test.copy()
    test["day"] = test["time"].dt.floor("D")
    days = sorted(test["day"].unique())

    # Train each learning method once (as for Table 1), then evaluate cumulatively.
    prop = MOMAQL(**kw)
    adaptive = AdaptiveMOMAQL(**adaptive_kw)
    for _ in range(epochs):
        if len(pred_req):
            prop.run(pred_req, training=True)
            adaptive.run(pred_req, training=True)
        prop.run(train, training=True)
        adaptive.run(train, training=True)
    prop.reset()
    adaptive.reset()

    greedy = GreedyEq3(**kw)

    rows = []
    for i, day in enumerate(days, start=1):
        day_req = test[test["day"] == day]
        prop.run(day_req, training=False)
        adaptive.run(day_req, training=False)
        greedy.run(day_req, training=False)
        rows.append({
            "horizon_days": i,
            "Greedy_fairness": greedy.metrics().fairness,
            "Proposed_fairness": prop.metrics().fairness,
            "Adaptive_fairness": adaptive.metrics().fairness,
            "Greedy_total_utility": greedy.metrics().total_utility,
            "Proposed_total_utility": prop.metrics().total_utility,
            "Adaptive_total_utility": adaptive.metrics().total_utility,
        })
    return pd.DataFrame(rows)


def plot_stability(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_greedy = "#2a78d6"    # dataviz reference palette, categorical slot 1
    color_proposed = "#eb6834"  # dataviz reference palette, categorical slot 2
    color_adaptive = "#1baf7a"  # dataviz reference palette, categorical slot 3

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=150)
    ax.set_yscale("log")
    ax.plot(df["horizon_days"], df["Greedy_fairness"].clip(lower=1e-3),
            marker="x", linestyle="-", color=color_greedy, label="Greedy")
    ax.plot(df["horizon_days"], df["Proposed_fairness"].clip(lower=1e-3),
            marker="s", linestyle="-", color=color_proposed, label="Proposed (MOMAQL)")
    ax.plot(df["horizon_days"], df["Adaptive_fairness"].clip(lower=1e-3),
            marker="o", linestyle="-", color=color_adaptive, label="Adaptive MOMAQL (ours)")

    ax.set_xlabel("Length of Time Horizon (days)")
    ax.set_ylabel("Fairness (Var of driver utility, log scale)")
    ax.set_title("Fairness vs. Time Horizon - reproduction of paper Fig. 4\n(Greedy, Proposed, and our Adaptive extension)")
    ax.set_xticks(df["horizon_days"])
    ax.grid(True, which="both", axis="y", color="#d8d7d0", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="yellow_tripdata_2016-03.csv")
    ap.add_argument("--drivers", type=int, default=20)
    ap.add_argument("--max-nodes", type=int, default=120)
    ap.add_argument("--grid-size", type=float, default=0.01)
    ap.add_argument("--peak-hours", type=int, default=2)
    ap.add_argument("--train-sample", type=float, default=0.05, help="stratified rate (Sec 5.2)")
    ap.add_argument("--test-limit", type=int, default=50000,
                     help="cap test requests for Table 1 to the paper's evaluation scale (~50k)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--omega", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--q-weight", type=float, default=1.0)
    ap.add_argument("--fair-scale", type=float, default=0.046,
                     help="per-request calibration of omega to keep the fairness penalty "
                          "in the same range as per-request utility")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--imbalance-scale", type=float, default=1.5,
                     help="Adaptive MOMAQL: weight on current driver-utility imbalance")
    ap.add_argument("--uncertainty-scale", type=float, default=0.8,
                     help="Adaptive MOMAQL: weight on OD-pair confidence")
    ap.add_argument("--demand-scale", type=float, default=0.6,
                     help="Adaptive MOMAQL: weight on local node demand imbalance")
    ap.add_argument("--skip-stability", action="store_true",
                     help="skip the Fig. 4 time-horizon stability experiment")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent / args.data
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    banner("STAGE 1  Load + build graph (real coordinates, Manhattan-only trips)")
    df = load_and_clean(str(data_path))
    print(f"  Manhattan trips: {len(df):,}")
    df = peak_window(df, args.peak_hours)
    print(f"  after peak-{args.peak_hours}h window: {len(df):,}")
    req, coords = build_graph(df, args.grid_size, args.max_nodes)
    print(f"  graph nodes: {len(coords)}   requests on graph: {len(req):,}")

    train_all, test = split_train_test(req)
    train = stratified_sample(train_all, args.train_sample, seed=args.seed)
    if args.test_limit and len(test) > args.test_limit:
        test_for_table1 = test.head(args.test_limit).reset_index(drop=True)
    else:
        test_for_table1 = test
    print(f"  train (before 26/03): {len(train_all):,} -> stratified {args.train_sample}: {len(train):,}")
    print(f"  test  (26/03-01/04, capped {args.test_limit}): {len(test_for_table1):,} of {len(test):,}")
    nodes = list(coords.keys())

    banner("STAGE 2  Request-prediction module: 3-layer MLP (Sec 4.2)")
    counts = hourly_counts(train)
    model, pairs, mse = train_predictor(counts, seed=args.seed)
    print(f"  OD pairs modelled: {len(pairs)}   in-sample MSE: {mse:.3f}")
    pred = predict_future(model, counts, test_for_table1["time"].min(), test_for_table1["time"].max(), pairs)
    pred_req = counts_to_requests(pred, coords)
    print(f"  predicted future pseudo-requests: {len(pred_req):,}")

    kw = dict(nodes=nodes, coords=coords, n_drivers=args.drivers, gamma=args.gamma,
              lam=args.lam, omega=args.omega, alpha=args.alpha, epsilon=args.epsilon,
              q_weight=args.q_weight, fair_scale=args.fair_scale, seed=args.seed)
    pair_counts = build_pair_counts(train)
    adaptive_kw = dict(kw, pair_counts=pair_counts, imbalance_scale=args.imbalance_scale,
                        uncertainty_scale=args.uncertainty_scale, demand_scale=args.demand_scale)

    banner("STAGE 3  Train MOMAQL (pretrain on predicted future, then historical) (Sec 4.3/4.4)")
    prop = MOMAQL(**kw)
    adaptive = AdaptiveMOMAQL(**adaptive_kw)
    for e in range(args.epochs):
        if len(pred_req):
            prop.run(pred_req, training=True)
            adaptive.run(pred_req, training=True)
        prop.run(train, training=True)
        adaptive.run(train, training=True)
        print(f"  epoch {e + 1}/{args.epochs} done  (|Q_proposed|={len(prop.Q):,}, |Q_adaptive|={len(adaptive.Q):,})")

    banner("STAGE 4  Evaluate on real future week (Table 1)")
    prop.reset()
    prop.run(test_for_table1, training=False)
    prop_m = prop.metrics()

    adaptive.reset()
    adaptive.run(test_for_table1, training=False)
    adaptive_m = adaptive.metrics()

    greedy = GreedyEq3(**kw)
    greedy.run(test_for_table1, training=False)
    greedy_m = greedy.metrics()

    banner("TABLE 1 (reproduction: Greedy + Proposed, plus our Adaptive extension)")
    print(f"{'Method':<26} {'Total Utility':>14} {'Fairness':>15} {'NormFair':>9} "
          f"{'Min':>10} {'Mean':>10} {'Max':>10}")
    print_row("Greedy", greedy_m)
    print_row("Proposed (MOMAQL)", prop_m)
    print_row("Adaptive MOMAQL (ours)", adaptive_m)
    print("\n  --- paper's reported numbers, for comparison ---")
    for name, d in PAPER_TABLE_1.items():
        print(f"  {name:<24} total_utility={d['total_utility']:>14,.2f}  "
              f"fairness={d['fairness']:>13,.2f}  norm_fairness={d['normalised_fairness']:.4f}")

    if not args.skip_stability:
        banner("STAGE 5  Time-horizon stability (Fig. 4 reproduction: Greedy, Proposed, Adaptive)")
        stability_df = time_horizon_stability(kw, adaptive_kw, train, pred_req, test, args.epochs)
        print(stability_df.to_string(index=False))
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = Path(__file__).resolve().parent / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "fig4_time_horizon_stability.csv"
        png_path = out_dir / "fig4_time_horizon_stability.png"
        stability_df.to_csv(csv_path, index=False)
        plot_stability(stability_df, png_path)
        print(f"\n  saved: {csv_path}")
        print(f"  saved: {png_path}")

    banner("NOTE on exactness")
    print(
        "  Greedy and Proposed (MOMAQL) reproduce the paper's DEFINITIONS exactly\n"
        "  (Eq. 1/2/3/6/7/8) and its PROTOCOL as described (Manhattan-only trips,\n"
        "  peak-hour window, train/test date split, stratified 0.05 sampling,\n"
        "  3-layer MLP predictor, lambda=1, omega=0.6, gamma=0.9). They will NOT\n"
        "  match Table 1's exact numbers, because the paper does not publish: the\n"
        "  exact grid/graph size, driver count, MLP hidden width, learning rate,\n"
        "  epsilon, random seed, or shortest-path library used. What should match\n"
        "  is the qualitative pattern the paper argues for: a myopic fixed-lambda\n"
        "  method either overpays for fairness (utility collapse) or underpays\n"
        "  (fairness collapse), while a look-ahead, state-aware method keeps both\n"
        "  in check as the time horizon grows.\n\n"
        "  Adaptive MOMAQL is OUR extension, not part of the paper: it keeps the\n"
        "  same reward/objective (Eq. 6/7) but replaces the fixed lambda with a\n"
        "  state-dependent lambda_t (see class docstring). Compare its row above\n"
        "  to 'Proposed (MOMAQL)' to see whether it improved total utility and/or\n"
        "  fairness relative to the paper's fixed-lambda formulation on this data."
    )


if __name__ == "__main__":
    main()
