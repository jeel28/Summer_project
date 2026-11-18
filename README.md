# Long-term Fairness in Ride-Hailing Allocation

This project reproduces and extends a fairness-aware ride-hailing allocation method based on the paper:

Kang, Chan, Shao, Salim, Leckie, "Long-term Fairness in Ride-Hailing Platform".

## Project Goal

The goal is to study the trade-off between:
- total utility / platform efficiency
- long-term fairness across drivers

The repository contains:
- a faithful reproduction of the paper’s MOMAQL method
- a greedy baseline
- an adaptive fairness extension that varies the fairness penalty based on system state

## Folder Structure

- `MTP_longterm_fairness/`
  - `long_term_fairness_momaql.py` : data pipeline, MLP predictor, MOMAQL and greedy (paper reproduction)
  - `adaptive_fairness.py` : adaptive fairness extension (component switches, optional hard bound, adaptive greedy)
  - `run_experiments.py` : multi-seed comparison, ablation and lambda sweep (writes `results/`)
  - `all_methods_runner.py` : command-line runner for the individual methods
  - `paper_reproduction_greedy_momaql.py` : compact script that also produces the time-horizon plot
  - `results/` : result CSVs and plots used in the presentation
  - `ADAPTIVE_FAIRNESS_DETAILED_SUMMARY.md` : formulas, worked example and measured results
  - `_make_plots.py`, `_depth_test.py`, `long_term_fairness_python_impl.py`, `run_no_bound.txt` : earlier iteration, kept for reference (not used by the experiments below)
- `Adaptive_Fairness_Final_Presentation.pptx` : final presentation

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download NYC Yellow Taxi trip data for March 2016 (`yellow_tripdata_2016-03.csv`, about 1.9 GB) and place it in `MTP_longterm_fairness/`. It is not stored in this repository.

## Run the experiments

From `MTP_longterm_fairness/`:

```bash
python3 run_experiments.py prepare              # once: load data, build graph, train predictor (~35 s)
python3 run_experiments.py seeds --seeds 5      # greedy / adaptive greedy / MOMAQL / adaptive MOMAQL
python3 run_experiments.py ablation --seeds 3   # which part of the adaptive weight helps
python3 run_experiments.py pareto --seeds 2     # lambda sweep
```

One learning run takes about 40-70 s. To run in parallel, start several shells with `--shard k/N` (for example `--shard 0/4` ... `--shard 3/4`) and merge the `*_shard*.csv` files.

## Results (lambda = 1, mean over 5 seeds)

| Method | Total utility | Variance | Std/mean |
|---|---|---|---|
| Greedy | 101,838 | 38 | 0.001 |
| MOMAQL (paper) | 94,645 | 79,456 | 0.059 |
| Adaptive MOMAQL | 97,853 | 2,991 | 0.011 |

### Ablation (3 seeds): which part of the adaptive weight helps

| Variant | Total utility | Variance |
|---|---|---|
| None (fixed lambda) | 94,810 | 73,455 |
| Driver imbalance only | 95,782 | 67,582 |
| Uncertainty only | 95,371 | 36,009 |
| Local demand only | 97,156 | 10,276 |
| Driver + local | 97,531 | 9,298 |
| All three | 98,058 | 3,118 |

### Lambda sweep (2 seeds)

| Base lambda | Fixed: utility | Fixed: variance | Adaptive: utility | Adaptive: variance |
|---|---|---|---|---|
| 0 | 76,875 | 8,544,214 | 76,875 | 8,544,214 |
| 0.5 | 92,648 | 217,553 | 96,238 | 13,181 |
| 1 | 95,043 | 79,510 | 98,296 | 2,561 |
| 2 | 96,690 | 26,177 | 100,258 | 572 |
| 5 | 99,878 | 4,066 | 101,254 | 115 |
| 10 | 101,386 | 902 | 101,271 | 16 |
| 25 | 102,520 | 188 | 94,039 | 8 |
| 50 | 101,950 | 26 | 80,869 | 2 |

At lambda = 0 both methods are identical, which checks the code. Seeds change only the agent (start positions, exploration); the data split, graph and predictor are fixed. Plots are in `MTP_longterm_fairness/results/plots/`, raw numbers in `MTP_longterm_fairness/results/*.csv`.

The paper's greedy numbers are not reproduced: the paper does not publish the graph size, driver count, MLP width, learning rate, epsilon, seed or shortest-path library, so only the definitions and the protocol are replicated. Our MOMAQL is close to the paper's (95,824 / 85,194 / 0.0610). A fixed-lambda MOMAQL with lambda about 10 matches or beats the adaptive method at lambda = 1, so part of the gain comes from a larger effective lambda. See the summary document for details.

## Key idea

The paper uses a fixed fairness weight `lambda` in the scalarised objective. In this project, the adaptive extension allows the fairness penalty to change according to:
- driver imbalance
- local demand skew
- OD-pair uncertainty

This attempts to improve fairness without losing too much total utility.

## Main Metrics

- Total utility
- Variance / fairness penalty
- Normalised fairness: `std(U) / mean(U)`
- Min / mean / max driver utility

## Notes

- Large data files and generated artifacts are intentionally ignored by `.gitignore`.
- For a clean GitHub repo, keep the source code and summary, and keep raw datasets and generated plots separate.

## License

This project is for academic and research use.
