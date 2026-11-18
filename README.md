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
