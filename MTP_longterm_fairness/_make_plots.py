"""Reproduce the exact main() experiment and produce 3 presentation-ready charts.

Uses the original long_term_fairness_python_impl as a library (no edits) with the
SAME defaults as the user's command (sample=200000, drivers=50, max-nodes=60,
train/test limit=50000, random_state=7), so the numbers match exactly:
  Proposed: util 66,701.09  var 976.02  normF 0.0234  min/mean/max 1246.8/1334.0/1374.6
  Greedy  : util 45,159.59  var 994,463.94 normF 1.1041 min/mean/max 0.0/903.2/4804.4
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import long_term_fairness_python_impl as m

DATA = "yellow_tripdata_2016-03_coords.parquet"
SAMPLE, DRIVERS, MAX_NODES, GRID = 200_000, 50, 60, 0.01
TRAIN_LIMIT = TEST_LIMIT = 50_000

PROPOSED_C = "#1b9e77"   # teal-green
GREEDY_C = "#d95f02"     # orange-red
GRID_C = "#dddddd"

# ----------------------------------------------------------------------------
# Reproduce main() exactly, but keep the allocator objects to read per-driver data
# ----------------------------------------------------------------------------
print("Preparing data + predictor (matches your command)...")
raw = m.load_taxi_data(DATA, sample=SAMPLE)
requests, node_coords = m.build_requests(raw, grid_size=GRID, max_nodes=MAX_NODES)
train, test = m.split_train_test(requests)
nodes = list(node_coords.keys())

counts = m.hourly_counts(train)
predictor, pairs = m.train_mlp_request_predictor(counts)
pred_counts = m.predict_future_counts(predictor, counts, test["time"].min(), test["time"].max(), pairs)
predicted_requests = m.expand_predicted_requests(pred_counts, node_coords)

proposed = m.FairQAllocator(nodes=nodes, node_coords=node_coords, n_drivers=DRIVERS,
                            gamma=0.9, fairness_lambda=1.0, omega=0.6, epsilon=0.05)
proposed.run(predicted_requests, training=True, limit=TRAIN_LIMIT)
proposed.run(train, training=True, limit=TRAIN_LIMIT)
proposed.reset()
pm = proposed.run(test, training=False, limit=TEST_LIMIT)
prop_util = np.sort(proposed.utilities.copy())[::-1]   # per-driver, high -> low

greedy = m.GreedyUtilityAllocator(nodes=nodes, node_coords=node_coords, n_drivers=DRIVERS)
gm = greedy.run(test, training=False, limit=TEST_LIMIT)
greedy_util = np.sort(greedy.utilities.copy())[::-1]

print(f"Proposed: total={pm.total_utility:,.2f} var={pm.fairness_variance:,.2f} "
      f"normF={pm.normalised_fairness:.4f} min/mean/max={pm.min_utility:.1f}/{pm.mean_utility:.1f}/{pm.max_utility:.1f}")
print(f"Greedy  : total={gm.total_utility:,.2f} var={gm.fairness_variance:,.2f} "
      f"normF={gm.normalised_fairness:.4f} min/mean/max={gm.min_utility:.1f}/{gm.mean_utility:.1f}/{gm.max_utility:.1f}")


def thousands(x, _pos):
    return f"{x:,.0f}"


# ============================================================================
# FIGURE 1 — Per-driver earnings: the fairness "money shot"
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
x = np.arange(1, DRIVERS + 1)

for ax, data, color, title, mt in [
    (axes[0], greedy_util, GREEDY_C, "GREEDY  (nearest-driver)", gm),
    (axes[1], prop_util, PROPOSED_C, "PROPOSED  (fair Q-learning)", pm),
]:
    ax.bar(x, data, color=color, width=0.9, edgecolor="white", linewidth=0.3)
    ax.axhline(mt.mean_utility, color="#333333", ls="--", lw=1.3,
               label=f"mean = {mt.mean_utility:,.0f}")
    ax.set_title(title, fontsize=13, fontweight="bold", color=color, pad=10)
    ax.set_xlabel("drivers (sorted highest → lowest earner)", fontsize=10)
    ax.grid(axis="y", color=GRID_C, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.margins(x=0.01)

axes[0].set_ylabel("accumulated utility  (net productive km ≈ earnings)", fontsize=11)
# annotate the smoking guns
axes[0].annotate(f"top driver\n{greedy_util[0]:,.0f}", xy=(1, greedy_util[0]),
                 xytext=(6, greedy_util[0]), fontsize=9.5, color=GREEDY_C, fontweight="bold",
                 va="center", arrowprops=dict(arrowstyle="->", color=GREEDY_C))
axes[0].annotate(f"poorest drivers\nearn ≈ 0", xy=(DRIVERS, 0),
                 xytext=(DRIVERS - 20, 900), fontsize=9.5, color=GREEDY_C, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEDY_C))
axes[1].annotate("everyone earns\nnearly the same", xy=(DRIVERS // 2, prop_util[DRIVERS // 2]),
                 xytext=(DRIVERS // 2 - 8, 300), fontsize=9.5, color=PROPOSED_C, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=PROPOSED_C))

fig.suptitle("What each of the 50 drivers earned over the test week",
             fontsize=15, fontweight="bold", y=0.99)
fig.text(0.5, 0.005,
         "Greedy: a few well-placed drivers grab everything, many earn nothing.   "
         "Proposed: work is spread so every driver earns a fair share.",
         ha="center", fontsize=10.5, style="italic", color="#555555")
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig("fig1_driver_earnings.png", dpi=200)
plt.close(fig)
print("saved fig1_driver_earnings.png")

# ============================================================================
# FIGURE 2 — More total AND more equal (efficiency + range)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
labels = ["Greedy", "Proposed"]
colors = [GREEDY_C, PROPOSED_C]

# (a) total utility
tot = [gm.total_utility, pm.total_utility]
bars = axes[0].bar(labels, tot, color=colors, width=0.6, edgecolor="white")
axes[0].set_title("Total utility (efficiency)", fontsize=13, fontweight="bold", pad=10)
axes[0].set_ylabel("total accumulated utility", fontsize=11)
axes[0].yaxis.set_major_formatter(FuncFormatter(thousands))
axes[0].grid(axis="y", color=GRID_C, lw=0.8); axes[0].set_axisbelow(True)
for b, v in zip(bars, tot):
    axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
gain = (pm.total_utility / gm.total_utility - 1) * 100
axes[0].text(1, pm.total_utility * 0.5, f"+{gain:.0f}%\nvs greedy", ha="center", va="center",
             fontsize=13, fontweight="bold", color="white")

# (b) min / mean / max spread
gvals = [gm.min_utility, gm.mean_utility, gm.max_utility]
pvals = [pm.min_utility, pm.mean_utility, pm.max_utility]
xx = np.arange(3); w = 0.38
axes[1].bar(xx - w / 2, gvals, w, color=GREEDY_C, label="Greedy", edgecolor="white")
axes[1].bar(xx + w / 2, pvals, w, color=PROPOSED_C, label="Proposed", edgecolor="white")
axes[1].set_xticks(xx); axes[1].set_xticklabels(["poorest\ndriver", "average\ndriver", "richest\ndriver"], fontsize=10)
axes[1].set_title("Earnings range across drivers", fontsize=13, fontweight="bold", pad=10)
axes[1].set_ylabel("accumulated utility", fontsize=11)
axes[1].yaxis.set_major_formatter(FuncFormatter(thousands))
axes[1].grid(axis="y", color=GRID_C, lw=0.8); axes[1].set_axisbelow(True)
axes[1].legend(fontsize=10, frameon=False)
for i, v in enumerate(gvals):
    axes[1].text(i - w / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9, color=GREEDY_C, fontweight="bold")
for i, v in enumerate(pvals):
    axes[1].text(i + w / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9, color=PROPOSED_C, fontweight="bold")

fig.suptitle("The fair method earns MORE in total — and spreads it evenly",
             fontsize=15, fontweight="bold")
fig.text(0.5, 0.005,
         "Left: proposed beats greedy on total output.   "
         "Right: greedy ranges from 0 to 4,804 per driver; proposed stays a tight 1,247–1,375.",
         ha="center", fontsize=10.5, style="italic", color="#555555")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("fig2_utility_and_range.png", dpi=200)
plt.close(fig)
print("saved fig2_utility_and_range.png")

# ============================================================================
# FIGURE 3 — The fairness gap (variance on log scale + normalised fairness)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

# (a) fairness variance, log scale
var = [gm.fairness_variance, pm.fairness_variance]
bars = axes[0].bar(labels, var, color=colors, width=0.6, edgecolor="white")
axes[0].set_yscale("log")
axes[0].set_title("Fairness = variance of driver earnings\n(lower = fairer)", fontsize=12.5, fontweight="bold", pad=10)
axes[0].set_ylabel("variance (log scale)", fontsize=11)
axes[0].grid(axis="y", color=GRID_C, lw=0.8, which="both"); axes[0].set_axisbelow(True)
for b, v in zip(bars, var):
    axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
factor = gm.fairness_variance / pm.fairness_variance
axes[0].text(0.5, 0.06, f"proposed is ≈ {factor:,.0f}× more equal",
             transform=axes[0].transAxes, ha="center", fontsize=12, fontweight="bold", color=PROPOSED_C)

# (b) normalised fairness, linear
nf = [gm.normalised_fairness, pm.normalised_fairness]
bars = axes[1].bar(labels, nf, color=colors, width=0.6, edgecolor="white")
axes[1].set_title("Normalised fairness\n(earnings gap ÷ average; lower = fairer)", fontsize=12.5, fontweight="bold", pad=10)
axes[1].set_ylabel("normalised fairness", fontsize=11)
axes[1].grid(axis="y", color=GRID_C, lw=0.8); axes[1].set_axisbelow(True)
for b, v in zip(bars, nf):
    axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
axes[1].text(1, nf[1], "  ≈ 2% spread\n  (almost perfectly equal)", ha="left", va="bottom",
             fontsize=9.5, color=PROPOSED_C, fontweight="bold")
axes[1].text(0, nf[0], "  ≈ 110% spread\n  (huge inequality)", ha="left", va="top",
             fontsize=9.5, color=GREEDY_C, fontweight="bold")

fig.suptitle("How unequal are driver earnings? Greedy vs Proposed",
             fontsize=15, fontweight="bold")
fig.text(0.5, 0.005,
         "Both panels measure inequality of driver earnings. The proposed fair method is dramatically fairer on both.",
         ha="center", fontsize=10.5, style="italic", color="#555555")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("fig3_fairness_gap.png", dpi=200)
plt.close(fig)
print("saved fig3_fairness_gap.png")

print("\nAll charts written.")
