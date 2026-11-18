"""Adaptive fairness extension of the paper's long-term ride-hailing MOMAQL.

This file keeps the paper's MOMAQL logic and replaces the fixed fairness
scalarisation with a state-dependent fairness controller. The fairness penalty is
scaled by the current system imbalance, local demand skew, and pair uncertainty.

This is the adaptive-fairness-only implementation: no extra greedy baselines,
no hard-bound variants, and no comparison runner. It only implements the paper's
MOMAQL structure with adaptive fairness weighting.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple
import pickle

import numpy as np
import pandas as pd

import long_term_fairness_momaql as base


class AdaptiveFairnessMOMAQL(base.MOMAQL):
    """MOMAQL with a dynamic fairness weight that depends on the system state."""

    def __init__(
        self,
        nodes,
        coords,
        pair_counts: Dict[Tuple[str, str], int],
        n_drivers=20,
        gamma=0.9,
        lam=1.0,
        omega=0.6,
        alpha=0.2,
        epsilon=0.1,
        q_weight=1.0,
        fair_scale=0.046,
        seed=7,
        imbalance_scale=1.5,
        uncertainty_scale=0.8,
        demand_scale=0.6,
        fairness_bound=None,
        use_driver=True,
        use_local=True,
        use_unc=True,
    ):
        super().__init__(
            nodes=nodes,
            coords=coords,
            n_drivers=n_drivers,
            gamma=gamma,
            lam=lam,
            omega=omega,
            alpha=alpha,
            epsilon=epsilon,
            q_weight=q_weight,
            fair_scale=fair_scale,
            seed=seed,
        )
        self.pair_counts = dict(pair_counts)
        self.imbalance_scale = imbalance_scale
        self.uncertainty_scale = uncertainty_scale
        self.demand_scale = demand_scale
        # Optional hard bound on projected Var(U); None = soft penalty only.
        self.fairness_bound = fairness_bound
        self.bound_enforced_count = 0   # steps where a bound-feasible driver existed
        self.bound_violation_count = 0  # steps with no feasible driver (fallback)
        # Component switches (for the ablation study).
        self.use_driver = use_driver
        self.use_local = use_local
        self.use_unc = use_unc

        self.node_demand = defaultdict(float)
        for (pu, do), cnt in self.pair_counts.items():
            self.node_demand[pu] += float(cnt)
            self.node_demand[do] += float(cnt)
        self.node_mean_demand = float(np.mean(list(self.node_demand.values()))) if self.node_demand else 1.0

    def reset(self):
        super().reset()

    def _driver_imbalance(self) -> float:
        if self.n <= 1:
            return 0.0
        u = np.asarray(self.util, dtype=float)
        mean = float(np.mean(u))
        if abs(mean) < 1e-12:
            return 0.0
        return float(np.std(u) / abs(mean))

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

        coeff = 1.0
        if self.use_driver:
            coeff += self.imbalance_scale * driver_imb
        if self.use_local:
            coeff += self.demand_scale * local_imb
        if self.use_unc:
            coeff *= 1.0 + (1.0 - uncertainty) * self.uncertainty_scale
        return float(self.lam * coeff)

    def _projected_variance(self, d: int, r: float) -> float:
        return float(self.util.var() + self._marginal_var(d, r))

    def choose(self, pu, do, od_km, training):
        lam_t = self.adaptive_lambda(pu, do)
        scores, proj = [], []
        for d in range(self.n):
            r = self.reward(d, pu, do, od_km)
            mv = self._marginal_var(d, r)
            fair_pen = lam_t * self.omega * self.fair_scale * mv
            scores.append(r - fair_pen + self.q_weight * self.Q[(self.driver_node[d], pu, do)])
            proj.append(self.util.var() + mv)

        if self.fairness_bound is not None:
            feasible = [d for d in range(self.n) if proj[d] <= self.fairness_bound]
            if feasible:
                self.bound_enforced_count += 1
                best_d = max(feasible, key=lambda d: scores[d])
            else:  # no driver keeps Var(U) within the bound: pick the least-violating
                self.bound_violation_count += 1
                best_d = int(np.argmin(proj))
        else:
            best_d = int(np.argmax(scores))

        if training and self.rng.random() < self.epsilon:
            # epsilon-greedy exploration for the adaptive fairness objective.
            return int(self.rng.integers(0, self.n))
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

    def save_q_table(self, path: Path) -> None:
        payload = {"Q": dict(self.Q)}
        with path.open("wb") as f:
            pickle.dump(payload, f)

    def load_q_table(self, path: Path) -> None:
        with path.open("rb") as f:
            payload = pickle.load(f)
        q_loaded = payload.get("Q", {})
        self.Q.clear()
        for k, v in q_loaded.items():
            self.Q[tuple(k)] = float(v)

class AdaptiveGreedy(base.GreedyEq3):
    """Greedy (Eq. 3) whose fairness weight lambda_t adapts to the driver imbalance."""

    def __init__(self, *args, imbalance_scale=1.5, **kw):
        super().__init__(*args, **kw)
        self.imbalance_scale = imbalance_scale

    def choose(self, pu, do, od_km, training=False):
        u = self.util
        mean = float(np.mean(u))
        imb = float(np.std(u) / abs(mean)) if abs(mean) > 1e-12 else 0.0
        lam_t = self.lam * (1.0 + self.imbalance_scale * imb)
        base_var = u.var()
        best_d, best_s = 0, -1e18
        for d in range(self.n):
            r = self.reward(d, pu, do, od_km)
            s = r - lam_t * (base_var + self._marginal_var(d, r))
            if s > best_s:
                best_s, best_d = s, d
        return best_d


def build_pair_counts(train: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    counts = train.groupby(["pu", "do"]).size().to_dict()
    return {tuple(k): int(v) for k, v in counts.items()}


def add_arguments(ap):
    ap.add_argument("--data", default="yellow_tripdata_2016-03.csv")
    ap.add_argument("--drivers", type=int, default=20)
    ap.add_argument("--max-nodes", type=int, default=120)
    ap.add_argument("--grid-size", type=float, default=0.01)
    ap.add_argument("--peak-hours", type=int, default=2)
    ap.add_argument("--train-sample", type=float, default=0.05)
    ap.add_argument("--test-limit", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--omega", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--q-weight", type=float, default=1.0)
    ap.add_argument("--fair-scale", type=float, default=0.046)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--imbalance-scale", type=float, default=1.5)
    ap.add_argument("--uncertainty-scale", type=float, default=0.8)
    ap.add_argument("--demand-scale", type=float, default=0.6)
    ap.add_argument("--qtable-path", default="adaptive_qtable.pkl")
    ap.add_argument("--load-qtable", action="store_true")
    return ap


def run_experiment(args):
    base.banner("STAGE 1  Load + graph")
    df = base.load_and_clean(args.data)
    print(f"  Manhattan trips: {len(df):,}")
    df = base.peak_window(df, args.peak_hours)
    print(f"  after peak-{args.peak_hours}h window: {len(df):,}")
    req, coords = base.build_graph(df, args.grid_size, args.max_nodes)
    print(f"  graph nodes: {len(coords)}   requests on graph: {len(req):,}")

    train_all, test = base.split_train_test(req)
    train = base.stratified_sample(train_all, args.train_sample)
    if args.test_limit and len(test) > args.test_limit:
        test = test.head(args.test_limit).reset_index(drop=True)
    print(f"  train(before 26/03): {len(train_all):,} -> stratified {args.train_sample}: {len(train):,}")
    print(f"  test (26/03-01/04, capped {args.test_limit}): {len(test):,}")
    nodes = list(coords.keys())

    base.banner("STAGE 2  3-layer MLP request predictor")
    counts = base.hourly_counts(train)
    model, pairs, mse = base.train_predictor(counts)
    print(f"  OD pairs modelled: {len(pairs)}   in-sample MSE: {mse:.3f}")
    pred = base.predict_future(model, counts, test["time"].min(), test["time"].max(), pairs)
    pred_req = base.counts_to_requests(pred, coords)
    print(f"  predicted future pseudo-requests: {len(pred_req):,}")

    pair_counts = build_pair_counts(train)
    kw = dict(
        nodes=nodes,
        coords=coords,
        n_drivers=args.drivers,
        gamma=args.gamma,
        lam=args.lam,
        omega=args.omega,
        alpha=args.alpha,
        epsilon=args.epsilon,
        q_weight=args.q_weight,
        fair_scale=args.fair_scale,
        seed=args.seed,
        imbalance_scale=args.imbalance_scale,
        uncertainty_scale=args.uncertainty_scale,
        demand_scale=args.demand_scale,
    )

    base.banner("STAGE 3  Train adaptive fairness MOMAQL")
    agent = AdaptiveFairnessMOMAQL(pair_counts=pair_counts, **kw)
    if args.load_qtable:
        qtable_path = Path(args.qtable_path)
        if not qtable_path.is_absolute():
            qtable_path = Path(__file__).resolve().parent / qtable_path
        if qtable_path.exists():
            agent.load_q_table(qtable_path)
            print(f"  loaded Q-table: {qtable_path}")
        else:
            print(f"  Q-table not found, training from scratch: {qtable_path}")
    for e in range(args.epochs):
        if len(pred_req):
            agent.run(pred_req, training=True)
        agent.run(train, training=True)
        print(f"  epoch {e + 1}/{args.epochs} done  (|Q|={len(agent.Q):,})")
    qtable_path = Path(args.qtable_path)
    if not qtable_path.is_absolute():
        qtable_path = Path(__file__).resolve().parent / qtable_path
    agent.save_q_table(qtable_path)
    print(f"  saved Q-table: {qtable_path}")

    base.banner("STAGE 4  Evaluate on real future week")
    agent.reset()
    agent.run(test, training=False)
    prop = agent.metrics()

    base.banner("RESULTS")
    print(f"{'Method':<26} {'TotalUtility':>12} {'Fairness':>15} {'NormFair':>8} {'Min':>10} {'Mean':>10} {'Max':>10}")
    base.print_row("Adaptive MOMAQL (paper fairness)", prop)


def main():
    ap = argparse.ArgumentParser(description="Adaptive fairness variant of the MOMAQL ride-hailing model.")
    ap = add_arguments(ap)
    args = ap.parse_args()

    if not Path(args.data).exists():
        candidates = [
            Path(args.data),
            Path(__file__).resolve().parent / args.data,
            Path.cwd() / args.data,
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError(f"Data file not found: {args.data}")
        args.data = str(found)

    run_experiment(args)


if __name__ == "__main__":
    main()
