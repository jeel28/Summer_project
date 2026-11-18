"""Experiment driver: multi-seed comparison, component ablation, lambda sweep (Pareto).

Data preparation (CSV load, graph, MLP predictor) is done once and cached in
ctx_cache.pkl; every experiment then reuses it. Jobs run in parallel processes.

    python run_experiments.py prepare            # build cache (slow, once)
    python run_experiments.py time               # time one MOMAQL run
    python run_experiments.py seeds   --seeds 5  # greedy / MOMAQL / adaptive
    python run_experiments.py ablation --seeds 3
    python run_experiments.py pareto  --seeds 2
    python run_experiments.py report              # plots + markdown tables

Seeds vary the agent only (initial driver positions, epsilon-greedy
exploration); the data split, graph and predictor are fixed.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import long_term_fairness_momaql as base
from adaptive_fairness import AdaptiveFairnessMOMAQL, AdaptiveGreedy, build_pair_counts

HERE = Path(__file__).resolve().parent
CACHE = HERE / "ctx_cache.pkl"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

KW = dict(n_drivers=20, gamma=0.9, lam=1.0, omega=0.6, alpha=0.2,
          epsilon=0.1, q_weight=1.0, fair_scale=0.046)
EPOCHS = 3
_CTX = None


def prepare(data=HERE / "yellow_tripdata_2016-03.csv", max_nodes=120, grid=0.01,
            peak_hours=2, train_sample=0.05, test_limit=50000):
    df = base.load_and_clean(str(data))
    df = base.peak_window(df, peak_hours)
    req, coords = base.build_graph(df, grid, max_nodes)
    train_all, test = base.split_train_test(req)
    train = base.stratified_sample(train_all, train_sample)
    test = test.head(test_limit).reset_index(drop=True)
    counts = base.hourly_counts(train)
    model, pairs, mse = base.train_predictor(counts)
    pred = base.predict_future(model, counts, test["time"].min(), test["time"].max(), pairs)
    pred_req = base.counts_to_requests(pred, coords)
    ctx = dict(coords=coords, nodes=list(coords), train=train, test=test,
               pred_req=pred_req, pair_counts=build_pair_counts(train))
    with CACHE.open("wb") as f:
        pickle.dump(ctx, f)
    print(f"cache saved: train={len(train):,} test={len(test):,} pred={len(pred_req):,} nodes={len(coords)}")
    return ctx


def ctx():
    global _CTX
    if _CTX is None:
        with CACHE.open("rb") as f:
            _CTX = pickle.load(f)
    return _CTX


def _metrics_row(m, **extra):
    return dict(total_utility=m.total_utility, variance=m.fairness, norm_fair=m.normalised_fairness,
                min_u=m.min_u, mean_u=m.mean_u, max_u=m.max_u, **extra)


def run_job(job):
    """job = dict(method, seed, lam, flags...). Returns a result row."""
    c = ctx()
    t0 = time.time()
    kw = dict(KW, nodes=c["nodes"], coords=c["coords"], seed=job["seed"])
    kw["lam"] = job.get("lam", 1.0)
    method = job["method"]
    if method == "greedy":
        a = base.GreedyEq3(**kw)
        a.run(c["test"], training=False)
    elif method == "adaptive_greedy":
        a = AdaptiveGreedy(**kw)
        a.run(c["test"], training=False)
    else:
        if method == "momaql":
            a = base.MOMAQL(**kw)
        else:
            a = AdaptiveFairnessMOMAQL(
                pair_counts=c["pair_counts"], imbalance_scale=1.5, uncertainty_scale=0.8,
                demand_scale=0.6, use_driver=job.get("use_driver", True),
                use_local=job.get("use_local", True), use_unc=job.get("use_unc", True), **kw)
        for _ in range(EPOCHS):
            if len(c["pred_req"]):
                a.run(c["pred_req"], training=True)
            a.run(c["train"], training=True)
        a.reset()
        a.run(c["test"], training=False)
    row = _metrics_row(a.metrics(), method=job["method"], seed=job["seed"], lam=kw["lam"],
                       variant=job.get("variant", ""), secs=round(time.time() - t0, 1))
    return row


def run_jobs(jobs, name, workers):
    """Serial loop; parallelism is done by launching several shards (see --shard)."""
    rows = []
    if True:
        for i, r in enumerate(map(run_job, jobs), 1):
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['method']:<16} {r['variant']:<14} seed={r['seed']} lam={r['lam']:<5} "
                  f"util={r['total_utility']:,.0f} var={r['variance']:,.1f} ({r['secs']}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    return pd.DataFrame(rows)


def jobs_seeds(n):
    return [dict(method=m, seed=s) for s in range(1, n + 1)
            for m in ("greedy", "adaptive_greedy", "momaql", "adaptive")]


def jobs_ablation(n):
    variants = {
        "none (=fixed)": dict(use_driver=False, use_local=False, use_unc=False),
        "driver only": dict(use_driver=True, use_local=False, use_unc=False),
        "local only": dict(use_driver=False, use_local=True, use_unc=False),
        "uncertainty only": dict(use_driver=False, use_local=False, use_unc=True),
        "driver+local": dict(use_driver=True, use_local=True, use_unc=False),
        "full": dict(),
    }
    return [dict(method="adaptive", seed=s, variant=v, **flags)
            for s in range(1, n + 1) for v, flags in variants.items()]


LAMS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]


def jobs_pareto(n):
    return [dict(method=m, seed=s, lam=lam) for s in range(1, n + 1)
            for m in ("momaql", "adaptive") for lam in LAMS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "time", "seeds", "ablation", "pareto"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", default="0/1", help="k/N: run every N-th job starting at k")
    a = ap.parse_args()
    if a.cmd == "prepare":
        t = time.time(); prepare(); print(f"prepare took {time.time()-t:.0f}s")
    elif a.cmd == "time":
        t = time.time(); r = run_job(dict(method="momaql", seed=1)); print(r, f"{time.time()-t:.0f}s")
    else:
        jobs = dict(seeds=jobs_seeds, ablation=jobs_ablation, pareto=jobs_pareto)[a.cmd](a.seeds)
        k, n = map(int, a.shard.split("/"))
        jobs = jobs[k::n]
        name = a.cmd if n == 1 else f"{a.cmd}_shard{k}"
        t = time.time(); run_jobs(jobs, name, a.workers); print(f"{a.cmd} took {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
