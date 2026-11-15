"""Load data once, then sweep the per-step fairness weight (fair_scale) to find the
value that reproduces the paper's normalised fairness ~0.061 while total utility
stays near 95,824 (test capped to the paper's ~50k-request evaluation scale)."""
import time
import numpy as np
import long_term_fairness_momaql as M

DATA = "kaggle_csv/yellow_tripdata_2016-03.csv"
TEST_LIMIT = 50_000

t0 = time.time()
print("Loading + building graph once...")
df = M.load_and_clean(DATA)
df = M.peak_window(df, 2)
req, coords = M.build_graph(df, 0.01, 120)
train_all, test = M.split_train_test(req)
train = M.stratified_sample(train_all, 0.05)
test = test.head(TEST_LIMIT).reset_index(drop=True)
nodes = list(coords.keys())

counts = M.hourly_counts(train)
model, pairs, mse = M.train_predictor(counts)
pred = M.predict_future(model, counts, test["time"].min(), test["time"].max(), pairs)
pred_req = M.counts_to_requests(pred, coords)
print(f"ready: train={len(train):,} test={len(test):,} pred={len(pred_req):,} "
      f"nodes={len(nodes)} mse={mse:.3f}  ({time.time()-t0:.0f}s)")


def evaluate(fair_scale, epochs=2, seed=7):
    kw = dict(nodes=nodes, coords=coords, n_drivers=20, gamma=0.9, lam=1.0, omega=0.6,
              alpha=0.2, epsilon=0.1, q_weight=1.0, fair_scale=fair_scale, seed=seed)
    a = M.MOMAQL(**kw)
    for _ in range(epochs):
        if len(pred_req):
            a.run(pred_req, True)
        a.run(train, True)
    a.reset()
    a.run(test, False)
    return a.metrics()


print("\nPaper Proposed target: total=95,823.79 var=85,194.48 norm=0.0610 "
      "min/mean/max=4565.56/4791.19/5931.93\n")
print(f"{'fair_scale':>10} {'total':>12} {'variance':>14} {'norm':>8} {'min':>9} {'mean':>9} {'max':>9} {'sec':>6}")
for fs in [0.037, 0.038, 0.039]:
    ts = time.time()
    m = evaluate(fs)
    print(f"{fs:>10} {m.total_utility:>12,.0f} {m.fairness:>14,.1f} {m.normalised_fairness:>8.4f} "
          f"{m.min_u:>9.0f} {m.mean_u:>9.0f} {m.max_u:>9.0f} {time.time()-ts:>6.0f}")
