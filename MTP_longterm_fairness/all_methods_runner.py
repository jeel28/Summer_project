"""Unified runner for all fairness-dispatch methods.

Includes:
1) simple greedy (paper-style Eq.3 greedy)
2) adaptive greedy
3) MOMAQL
4) adaptive MOMAQL
5) adaptive + hard-bound MOMAQL
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import long_term_fairness_momaql as base
from adaptive_fairness import (
    AdaptiveFairnessMOMAQL,
    AdaptiveGreedy,
    build_pair_counts,
)


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument(
        "--method",
        choices=[
            "simple-greedy",
            "paper-greedy",
            "adaptive-greedy",
            "momaql",
            "adaptive-momaql",
            "adaptive-hard-bound-momaql",
            "all",
        ],
        default="all",
    )
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
    ap.add_argument(
        "--fairness-bound",
        type=float,
        default=3000.0,
        help="Used only by adaptive-hard-bound-momaql.",
    )
    ap.add_argument("--qtable-path", default="adaptive_qtable.pkl")
    ap.add_argument("--load-qtable", action="store_true")
    ap.add_argument("--save-qtable", action="store_true")
    return ap


def resolve_data_path(data_arg: str) -> str:
    p = Path(data_arg)
    if p.exists():
        return str(p)
    candidates = [
        Path(__file__).resolve().parent / data_arg,
        Path.cwd() / data_arg,
    ]
    found = next((x for x in candidates if x.exists()), None)
    if found is None:
        raise FileNotFoundError(f"Data file not found: {data_arg}")
    return str(found)


def prepare_context(args) -> Dict[str, object]:
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

    base.banner("STAGE 2  3-layer MLP request predictor")
    counts = base.hourly_counts(train)
    model, pairs, mse = base.train_predictor(counts)
    print(f"  OD pairs modelled: {len(pairs)}   in-sample MSE: {mse:.3f}")
    pred = base.predict_future(model, counts, test["time"].min(), test["time"].max(), pairs)
    pred_req = base.counts_to_requests(pred, coords)
    print(f"  predicted future pseudo-requests: {len(pred_req):,}")

    pair_counts = build_pair_counts(train)
    return {
        "coords": coords,
        "nodes": list(coords.keys()),
        "train": train,
        "test": test,
        "pred_req": pred_req,
        "pair_counts": pair_counts,
    }


def base_momaql_kwargs(args, nodes, coords):
    return dict(
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
    )


def adaptive_kwargs(args, nodes, coords, pair_counts, fairness_bound=None):
    kw = base_momaql_kwargs(args, nodes, coords)
    kw.update(
        pair_counts=pair_counts,
        imbalance_scale=args.imbalance_scale,
        uncertainty_scale=args.uncertainty_scale,
        demand_scale=args.demand_scale,
        fairness_bound=fairness_bound,
    )
    return kw


def maybe_load_qtable(agent: AdaptiveFairnessMOMAQL, args) -> Path:
    qtable_path = Path(args.qtable_path)
    if not qtable_path.is_absolute():
        qtable_path = Path(__file__).resolve().parent / qtable_path
    if args.load_qtable:
        if qtable_path.exists():
            agent.load_q_table(qtable_path)
            print(f"  loaded Q-table: {qtable_path}")
        else:
            print(f"  Q-table not found, training from scratch: {qtable_path}")
    return qtable_path


def run_simple_greedy(ctx: Dict[str, object], args) -> base.Metrics:
    greedy = base.GreedyEq3(**base_momaql_kwargs(args, ctx["nodes"], ctx["coords"]))
    greedy.reset()
    greedy.run(ctx["test"], training=False)
    return greedy.metrics()


def run_adaptive_greedy(ctx: Dict[str, object], args) -> base.Metrics:
    greedy = AdaptiveGreedy(**base_momaql_kwargs(args, ctx["nodes"], ctx["coords"]))
    greedy.reset()
    greedy.run(ctx["test"], training=False)
    return greedy.metrics()


def run_momaql(ctx: Dict[str, object], args) -> base.Metrics:
    agent = base.MOMAQL(**base_momaql_kwargs(args, ctx["nodes"], ctx["coords"]))
    for _ in range(args.epochs):
        if len(ctx["pred_req"]):
            agent.run(ctx["pred_req"], training=True)
        agent.run(ctx["train"], training=True)
    agent.reset()
    agent.run(ctx["test"], training=False)
    return agent.metrics()


def run_adaptive_momaql(ctx: Dict[str, object], args, fairness_bound=None, persist_qtable=False) -> Tuple[base.Metrics, AdaptiveFairnessMOMAQL]:
    agent = AdaptiveFairnessMOMAQL(
        **adaptive_kwargs(args, ctx["nodes"], ctx["coords"], ctx["pair_counts"], fairness_bound=fairness_bound)
    )
    qtable_path = maybe_load_qtable(agent, args)
    for _ in range(args.epochs):
        if len(ctx["pred_req"]):
            agent.run(ctx["pred_req"], training=True)
        agent.run(ctx["train"], training=True)
    if persist_qtable and args.save_qtable:
        agent.save_q_table(qtable_path)
        print(f"  saved Q-table: {qtable_path}")
    agent.reset()
    agent.run(ctx["test"], training=False)
    return agent.metrics(), agent


def print_results(rows: List[Tuple[str, base.Metrics]]) -> None:
    base.banner("RESULTS")
    print(f"{'Method':<32} {'TotalUtility':>12} {'Fairness':>15} {'NormFair':>8} {'Min':>10} {'Mean':>10} {'Max':>10}")
    for name, metric in rows:
        print(
            f"{name:<32} {metric.total_utility:>12,.2f} {metric.fairness:>15,.2f} "
            f"{metric.normalised_fairness:>8.4f} {metric.min_u:>10.2f} {metric.mean_u:>10.2f} {metric.max_u:>10.2f}"
        )


def main() -> None:
    ap = add_arguments(argparse.ArgumentParser(description="Run fairness methods individually or together."))
    args = ap.parse_args()
    args.data = resolve_data_path(args.data)

    ctx = prepare_context(args)
    rows: List[Tuple[str, base.Metrics]] = []

    if args.method in ("simple-greedy", "paper-greedy", "all"):
        rows.append(("Simple Greedy (Eq.3)", run_simple_greedy(ctx, args)))

    if args.method in ("adaptive-greedy", "all"):
        rows.append(("Adaptive Greedy", run_adaptive_greedy(ctx, args)))

    if args.method in ("momaql", "all"):
        rows.append(("MOMAQL", run_momaql(ctx, args)))

    if args.method in ("adaptive-momaql", "all"):
        metric, _ = run_adaptive_momaql(ctx, args, fairness_bound=None, persist_qtable=True)
        rows.append(("Adaptive MOMAQL", metric))

    if args.method in ("adaptive-hard-bound-momaql", "all"):
        metric, agent = run_adaptive_momaql(
            ctx,
            args,
            fairness_bound=args.fairness_bound,
            persist_qtable=False,
        )
        rows.append((f"Adaptive + HardBound ({args.fairness_bound:g})", metric))
        print(
            f"  hard-bound checks (test): feasible_steps={agent.bound_enforced_count:,} "
            f"fallback_steps={agent.bound_violation_count:,} bound={args.fairness_bound}"
        )

    print_results(rows)


if __name__ == "__main__":
    main()

