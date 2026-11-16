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
  - `long_term_fairness_momaql.py` : paper-style MOMAQL reproduction
  - `paper_reproduction_greedy_momaql.py` : compact reproduction with greedy + proposed + adaptive method
  - `adaptive_fairness.py` : adaptive fairness extension
  - `all_methods_runner.py` : runner for comparing methods
  - `ADAPTIVE_FAIRNESS_DETAILED_SUMMARY.md` : detailed summary of the method and formulas
  - other utility scripts for preprocessing and plotting

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the project

From the project root:

```bash
python3 MTP_longterm_fairness/all_methods_runner.py --method all --data MTP_longterm_fairness/yellow_tripdata_2016-03.csv --drivers 20 --max-nodes 120 --peak-hours 2 --train-sample 0.05
```

Or run the compact reproduction script:

```bash
python3 MTP_longterm_fairness/paper_reproduction_greedy_momaql.py --data MTP_longterm_fairness/yellow_tripdata_2016-03.csv --drivers 20 --max-nodes 120 --peak-hours 2 --train-sample 0.05
```

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
