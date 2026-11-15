"""In-depth test harness for long_term_fairness_python_impl.

Imports the ORIGINAL script as a library (no modifications) and exercises it:
  1. Prepare data + train the MLP predictor once (shared across configs).
  2. Ablation study (paper Table 2 spirit): proposed vs w/o prediction vs
     w/o fairness, plus the greedy utility-only baseline (paper Table 1 spirit).
  3. Driver-count sensitivity sweep.
  4. Determinism check (re-run proposed, compare).

Run: .venv/bin/python _depth_test.py
"""
import time
import numpy as np
import long_term_fairness_python_impl as m

DATA = "yellow_tripdata_2016-03_coords.parquet"
SAMPLE = 200_000
MAX_NODES = 60
GRID = 0.01
TRAIN_LIMIT = 50_000
TEST_LIMIT = 50_000


def banner(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


t0 = time.time()
banner("STAGE 1: Load + prepare data (once)")
raw = m.load_taxi_data(DATA, sample=SAMPLE)
requests, node_coords = m.build_requests(raw, grid_size=GRID, max_nodes=MAX_NODES)
train, test = m.split_train_test(requests)
nodes = list(node_coords.keys())
print(f"requests={len(requests):,}  nodes={len(node_coords)}  "
      f"train={len(train):,}  test={len(test):,}")
print(f"train time span: {train['time'].min()} .. {train['time'].max()}")
print(f"test  time span: {test['time'].min()} .. {test['time'].max()}")

banner("STAGE 2: Train MLP request predictor (once)")
counts = m.hourly_counts(train)
predictor, pairs = m.train_mlp_request_predictor(counts)
# quick in-sample MLP sanity: MSE on the supervised training matrix
X, y, meta = m.make_supervised_count_data(counts)
pred_insample = predictor.predict(X)
mse = float(np.mean((pred_insample - y.values) ** 2))
print(f"MLP OD pairs modeled: {len(pairs)}   in-sample MSE: {mse:.3f}")
pred_counts = m.predict_future_counts(
    predictor, counts, test["time"].min(), test["time"].max(), pairs
)
predicted_requests = m.expand_predicted_requests(pred_counts, node_coords)
print(f"predicted future OD-count rows: {len(pred_counts):,}  "
      f"-> pseudo requests: {len(predicted_requests):,}")


def run_variant(with_prediction, with_fairness, drivers=50, seed=7):
    lam = 1.0 if with_fairness else 0.0
    alloc = m.FairQAllocator(
        nodes=nodes, node_coords=node_coords, n_drivers=drivers,
        gamma=0.9, fairness_lambda=lam, omega=0.6, epsilon=0.05, random_state=seed,
    )
    if with_prediction:
        alloc.run(predicted_requests, training=True, limit=TRAIN_LIMIT)
    alloc.run(train, training=True, limit=TRAIN_LIMIT)
    alloc.reset()
    return alloc.run(test, training=False, limit=TEST_LIMIT)


def run_greedy(drivers=50, seed=7):
    g = m.GreedyUtilityAllocator(
        nodes=nodes, node_coords=node_coords, n_drivers=drivers, random_state=seed,
    )
    return g.run(test, training=False, limit=TEST_LIMIT)


def fmt(name, mt, dt):
    print(f"{name:<28} util={mt.total_utility:>12,.1f}  var={mt.fairness_variance:>16,.1f}  "
          f"normF={mt.normalised_fairness:>8.4f}  min/mean/max="
          f"{mt.min_utility:>8.1f}/{mt.mean_utility:>8.1f}/{mt.max_utility:>8.1f}  ({dt:.1f}s)")


banner("STAGE 3: Ablation (Table 2) + baseline (Table 1) @ 50 drivers")
results = {}
for name, kw in [
    ("Proposed (pred+fair)", dict(with_prediction=True, with_fairness=True)),
    ("w/o Prediction", dict(with_prediction=False, with_fairness=True)),
    ("w/o Fairness", dict(with_prediction=True, with_fairness=False)),
]:
    ts = time.time()
    mt = run_variant(**kw)
    results[name] = mt
    fmt(name, mt, time.time() - ts)
ts = time.time()
mg = run_greedy()
results["Greedy (utility-only)"] = mg
fmt("Greedy (utility-only)", mg, time.time() - ts)

banner("STAGE 4: Driver-count sensitivity (proposed vs greedy)")
for d in (25, 50, 100):
    ts = time.time()
    mp = run_variant(with_prediction=True, with_fairness=True, drivers=d)
    fmt(f"Proposed  drivers={d}", mp, time.time() - ts)
    ts = time.time()
    mg = run_greedy(drivers=d)
    fmt(f"Greedy    drivers={d}", mg, time.time() - ts)

banner("STAGE 5: Determinism check (re-run proposed @50, compare)")
a = run_variant(with_prediction=True, with_fairness=True)
b = run_variant(with_prediction=True, with_fairness=True)
print(f"run A util={a.total_utility:.4f} var={a.fairness_variance:.4f}")
print(f"run B util={b.total_utility:.4f} var={b.fairness_variance:.4f}")
print("DETERMINISTIC:" , abs(a.total_utility - b.total_utility) < 1e-6
      and abs(a.fairness_variance - b.fairness_variance) < 1e-6)

print(f"\nTotal harness time: {time.time() - t0:.1f}s")
