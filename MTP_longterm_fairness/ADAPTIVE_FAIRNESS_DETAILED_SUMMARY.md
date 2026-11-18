# Adaptive Fairness in Long-Term Ride-Hailing Allocation
## Detailed Summary for PPT Generation

---

## SECTION 1: PROBLEM STATEMENT FROM THE PAPER

### Background
The paper "Long-term Fairness in Ride-Hailing Platform" (Kang et al., ECML PKDD 2024) addresses the fundamental challenge of balancing two competing objectives in ride-hailing platforms:
1. **Efficiency (Utility Maximization)**: Maximize total platform revenue and driver earnings
2. **Fairness (Equity)**: Ensure drivers receive equitable opportunities and earnings over time

### The Core Problem
**Problem 1: Purely Utility-Driven Allocation**
- If the dispatcher only maximizes immediate utility, some drivers are repeatedly favored
- Other drivers are consistently neglected
- Result: High utility but severe fairness collapse
- Example: Driver A earns $500/day while Driver B earns $50/day after 1 week

**Problem 2: Purely Fairness-Driven Allocation**
- If the dispatcher enforces perfect fairness, total utility/revenue plummets
- The platform cannot operate profitably
- Example: Equal allocation might reduce total utility by 40-50%

**The Challenge**: Find a balance where both utility and fairness are maintained over time.

### Why Long-Term Fairness Matters
- Short-term fairness: One day allocation doesn't matter if corrected later
- Long-term fairness: Over days/weeks, accumulated utility disparity matters
- Q-learning horizon: The model must look ahead and plan for future fairness

---

## SECTION 2: PAPER FORMULATION WITH EXACT FORMULAS

### Data and Setup
**Dataset**: NYC Yellow Taxi March 2016
- Filtered to Manhattan trips only (both pickup and dropoff within Manhattan)
- Peak-hour window: 2 busiest consecutive hours (hours 18-19)
- Training data: Before 2016-03-26 (stratified 5% sample)
- Test data: 2016-03-26 to 2016-04-01 (7-day evaluation week)

**Allocation Setup**:
- n = 20 drivers
- Graph nodes: 120 grid-based locations
- State: (driver_current_location, pickup_node, dropoff_node)

### Mathematical Formulation

#### Utility Definition (Eq. 6)
**Reward for assigning request to driver d:**

r(d, pu, do) = Geo(do, pu) - Geo(pu, driver_location_d)

Where:
- Geo(A, B) = haversine distance in km between locations A and B
- Geo(do, pu) = distance traveled by passenger (platform benefit)
- Geo(pu, driver) = distance driver must travel to pick up (driver cost)

**Interpretation**: Drivers earn high reward for short pickups and long trips. Drivers earn low/negative reward for long pickups.

#### Efficiency (Eq. 1)
**Total Platform Utility:**

π(M) = Σ_{d=1}^{n} o_d = Σ_{d=1}^{n} Σ_{t∈assigned_to_d} r_t

Where:
- o_d = accumulated utility of driver d
- M = allocation matching
- r_t = reward of trip t

#### Long-Term Fairness (Eq. 2)
**Fairness Penalty (Variance of Driver Utilities):**

F(M) = Var(o_d) = (1/n) * Σ_{d=1}^{n} (o_d - mean(o))²

Where:
- mean(o) = (1/n) * Σ_{d=1}^{n} o_d
- Lower variance = more fair (drivers have similar outcomes)
- Higher variance = less fair (drivers have different outcomes)

#### Balance Objective (Eq. 3)
**Optimization Goal:**

max_M  π(M) - λ * F(M)

Where:
- λ = fairness weight (fixed constant, typically λ=1.0)
- Higher λ = fairness matters more
- Lower λ = utility matters more

#### Scalarization Function (Eq. 7)
**What the dispatcher uses for real-time decisions:**

SR(M) = Σ_{d=1}^{n} r_d - λ * ω * fair_scale * ΔVar(d)

Where:
- r_d = immediate reward
- ω = 0.6 (paper parameter)
- fair_scale = 0.046 (per-request calibration)
- ΔVar(d) = marginal variance change if driver d gets this reward

**Marginal Variance Change:**

ΔVar(d) = Var(o_1, ..., o_d + r, ..., o_n) - Var(o_1, ..., o_d, ..., o_n)

Computed efficiently as:

ΔVar(d) = [ss + (u_d + r)²]/n - (mean + r/n)² - var(old)

Where:
- ss = Σ u_i²
- u_d = current utility of driver d
- mean = current mean utility
- var(old) = current variance

#### Normalized Fairness (Eq. 8)
**Fairness Metric Reported in Results:**

Fhat = std(U) / mean(U)

Where:
- std(U) = standard deviation of driver utilities
- mean(U) = mean driver utility
- Interpretation: Coefficient of variation

---

## SECTION 3: THE ADAPTIVE FAIRNESS METHOD (OUR EXTENSION)

### Core Insight
**Paper Limitation**: λ is fixed for all requests.
- Every trip gets the same fairness penalty
- Inflexible: Cannot respond to current system conditions
- May over-penalize when system is already fair
- May under-penalize when system is highly imbalanced

**Adaptive Solution**: Replace fixed λ with state-dependent λ_t

λ_t = λ * g(current_system_state)

### Key Components of Adaptive Fairness

#### 1. Driver Imbalance Measurement
**Formula:**

I_driver = std(U) / |mean(U)|

Where:
- U = [u_1, u_2, ..., u_n] = current utilities of all drivers
- std(U) = standard deviation
- mean(U) = mean utility

**What it measures:**
- How uneven driver utilities are at this moment
- Ranges from 0 (all equal) to large values (highly unequal)

**Example 1 - Balanced:**
- Utilities: [100, 105, 95]
- mean = 100, std ≈ 4.08
- I_driver = 4.08/100 = 0.041 (very balanced)

**Example 2 - Imbalanced:**
- Utilities: [10, 50, 190]
- mean = 83.33, std ≈ 80
- I_driver = 80/83.33 ≈ 0.96 (highly imbalanced)

**Interpretation**: When I_driver is high, fairness pressure should increase.

#### 2. Local Demand Imbalance
**Formula:**

I_local(pu, do) = max(0, max(D_pu, D_do) / D_mean - 1)

Where:
- D_pu = historical total demand at pickup node
- D_do = historical total demand at dropoff node
- D_mean = average demand across all nodes

**What it measures:**
- Whether this OD pair touches "hot spots" with concentrated demand
- High demand areas need more fair allocation
- 0 means demand is balanced

**Example 1 - Balanced Region:**
- Average demand: 100 trips
- Pickup node: 98 trips, Dropoff node: 105 trips
- max(98, 105) / 100 = 1.05
- I_local = max(0, 1.05 - 1) = 0.05 (low, balanced)

**Example 2 - Hot Spot:**
- Average demand: 100 trips
- Pickup node: 150 trips, Dropoff node: 80 trips
- max(150, 80) / 100 = 1.5
- I_local = max(0, 1.5 - 1) = 0.5 (moderate imbalance)

**Example 3 - Below Average:**
- Average demand: 100 trips
- Pickup node: 60 trips, Dropoff node: 70 trips
- max(60, 70) / 100 = 0.7
- I_local = max(0, 0.7 - 1) = 0 (capped at 0)

**Interpretation**: Hot spots need stronger fairness enforcement.

#### 3. Pair Uncertainty Measurement
**Formula:**

I_unc(pu, do) = 1 / sqrt(count(pu, do) + 1)

Where:
- count(pu, do) = historical frequency of this OD pair
- Higher count = lower uncertainty

**What it measures:**
- Confidence in this OD pair based on historical data
- Rare pairs should not trigger aggressive fairness adjustments
- Common pairs are reliable

**Example 1 - Very Common Pair:**
- count = 100 historical trips
- I_unc = 1/sqrt(101) ≈ 0.099 (low uncertainty, high confidence)

**Example 2 - Moderately Common:**
- count = 10 historical trips
- I_unc = 1/sqrt(11) ≈ 0.301 (moderate uncertainty)

**Example 3 - Never Seen Before:**
- count = 0 historical trips
- I_unc = 1/sqrt(1) = 1.0 (maximum uncertainty)

**Interpretation**: Common pairs are treated with confidence. Rare pairs get less aggressive adaptation.

#### 4. Adaptive Lambda Computation
**Complete Formula:**

λ_t = λ · c(pu, do)

Where:

c(pu, do) = [1 + α₁ · I_driver + α₂ · I_local] × [1 + α₃ · (1 - I_unc)]

Default parameters:
- α₁ = imbalance_scale = 1.5
- α₂ = demand_scale = 0.6
- α₃ = uncertainty_scale = 0.8
- λ = 1.0 (paper baseline)

**Breaking Down the Coefficient:**

**Part A: System State Response**
c_A = 1 + 1.5 · I_driver + 0.6 · I_local

- Base value: 1.0
- Increases if drivers are imbalanced (1.5 multiplier)
- Increases if demand is locally skewed (0.6 multiplier)
- Example: If I_driver=0.5 and I_local=0.3: c_A = 1 + 0.75 + 0.18 = 1.93

**Part B: Confidence Modulation**
c_B = 1 + 0.8 · (1 - I_unc)

- If pair is common (I_unc ≈ 0): (1 - 0) ≈ 1, c_B ≈ 1.8
- If pair is rare (I_unc ≈ 1): (1 - 1) ≈ 0, c_B ≈ 1.0
- Acts as damping for unreliable pairs

**Final λ_t:**
λ_t = 1.0 · c_A · c_B

### Complete End-to-End Example

**Scenario Setup:**
- 3 drivers with current utilities: [100, 50, 200]
- Average node demand: 100 trips
- This pickup node: 150 trips, Dropoff node: 120 trips
- This OD pair: 10 historical observations
- Request: 12 km trip
- Driver 1 (at pickup node): reward = 12 - 0 = 12
- Driver 2 (2 km away): reward = 12 - 2 = 10
- Driver 3 (3 km away): reward = 12 - 3 = 9

**Step 1: Compute I_driver**
- mean(U) = (100 + 50 + 200)/3 = 116.67
- std(U) = sqrt([(100-116.67)² + (50-116.67)² + (200-116.67)²]/3) = 62.36
- I_driver = 62.36 / 116.67 = 0.535

**Step 2: Compute I_local**
- max(150, 120) / 100 = 1.5
- I_local = 1.5 - 1 = 0.5

**Step 3: Compute I_unc**
- I_unc = 1/sqrt(10 + 1) = 1/sqrt(11) = 0.301

**Step 4: Compute Coefficients**
- c_A = 1 + 1.5(0.535) + 0.6(0.5) = 1 + 0.802 + 0.3 = 2.102
- c_B = 1 + 0.8(1 - 0.302) = 1.559
- c = 2.102 × 1.559 = 3.276

**Step 5: Compute λ_t**
- λ_t = 1.0 × 3.276 = 3.28

**Interpretation**: The fairness weight is increased by 3.28x because:
- The system is imbalanced (drivers have very different utilities)
- This region is a hot spot (pickup node has high demand)
- The pair is fairly well-observed (not rare, but not extremely common)

**Step 6: Decision Making**
For each driver, the score is:

score(d) = r - λ_t · ω · fair_scale · ΔVar(d)

Assume:
- ω = 0.6
- fair_scale = 0.046
- ΔVar(1) = 0.05 (assigning to driver 1 barely changes fairness)
- ΔVar(2) = 0.15 (assigning to driver 2 slightly improves fairness)
- ΔVar(3) = 0.35 (assigning to driver 3 worsens fairness a lot)

**Fixed Fairness (Paper, λ=1.0):**
- score(1) = 12 - 1.0 × 0.6 × 0.046 × 0.05 = 12.0 - 0.0014 ≈ 12.00 ← CHOSEN
- score(2) = 10 - 1.0 × 0.6 × 0.046 × 0.15 = 10.0 - 0.0041 ≈ 9.996
- score(3) = 9 - 1.0 × 0.6 × 0.046 × 0.35 = 9.0 - 0.0097 ≈ 8.99

**Result**: Chooses Driver 1 (highest reward, barely penalized for fairness)

**Adaptive Fairness (λ_t = 3.28):**
- score(1) = 12 - 3.28 × 0.6 × 0.046 × 0.05 = 12.0 - 0.0045 ≈ 11.996
- score(2) = 10 - 3.28 × 0.6 × 0.046 × 0.15 = 10.0 - 0.0134 ≈ 9.987
- score(3) = 9 - 3.28 × 0.6 × 0.046 × 0.35 = 9.0 - 0.0313 ≈ 8.969

**Result**: Still chooses Driver 1, BUT the fairness penalty is now 3x stronger.

**Key Difference**: If fairness impact was larger, adaptive might switch to Driver 2 to improve fairness in this imbalanced situation.

---

## SECTION 4: COMPARISON: FIXED vs ADAPTIVE

### When System is BALANCED
- I_driver ≈ 0, I_local ≈ 0, I_unc ≈ 0.1
- λ_t = 1.0 × [1 + 0 + 0] × [1 + 0.72] ≈ 1.72
- **Effect**: Fairness weight is ~1.72x baseline (the coefficient is never below 1, so λ_t is never smaller than λ)
- **Behavior**: Mild fairness pressure; the weight is only *lower relative to the imbalanced case*, not lower than the paper's λ
- **Rationale**: System is already fair, so little extra pressure is added

### When System is IMBALANCED
- I_driver ≈ 1.0, I_local ≈ 0.5, I_unc ≈ 0.3
- λ_t = 1.0 × [1 + 1.5 + 0.3] × [1 + 0.56] ≈ 3.89
- **Effect**: Fairness weight increased to ~3.89x baseline
- **Behavior**: Strongly prioritizes fairness
- **Rationale**: System is very unfair, need strong fairness intervention

### Benefits of Adaptive Approach
| Aspect | Fixed (λ=1.0) | Adaptive (λ_t) |
|--------|---|---|
| Balanced system | Over-penalizes utility | Allows more utility |
| Imbalanced system | Under-penalizes unfairness | Forces fairness correction |
| Utility-Fairness Tradeoff | Static, rigid | Dynamic, responsive |
| Total Utility | May be low due to constant penalty | Higher by allocating penalty selectively |
| Long-term Fairness | May be unstable | More stable due to responsive adjustment |

---

## SECTION 5: IMPLEMENTATION WORKFLOW

### Stage 1: Data Loading and Graph Construction
1. Load NYC Yellow Taxi March 2016 CSV
2. Filter to Manhattan trips (both pickup and dropoff within bounding box)
3. Select peak 2-hour window (hours 18-19)
4. Build grid-based graph: group nearby locations into 120 nodes
5. Create coordinate mapping for each node

### Stage 2: Training Data Preparation
1. Split data: train (before 2016-03-26) and test (2016-03-26 to 2016-04-01)
2. Stratified sample training at 5% rate (preserve day-hour distribution)
3. Compute historical OD pair frequencies for adaptive method

### Stage 3: Request Prediction (Sec 4.2)
1. Build hourly OD demand counts from training data
2. Train 3-layer MLP predictor:
   - Input: lag features (1h, 2h, 24h), hour, day-of-week, pair_id
   - Hidden layer: 64 neurons
   - Output: predicted count for next hour
3. Forecast future demand for test period

### Stage 4: Initialize Allocation Agents
**Greedy Agent:**
- No learning, myopic decisions
- Score = reward - λ × fairness_penalty

**MOMAQL Agent (Paper):**
- Q-learning with fixed λ
- Uses scalarization: r - λ·ω·fair_scale·ΔVar + Q

**Adaptive MOMAQL Agent (Ours):**
- Q-learning with adaptive λ_t
- Uses: r - λ_t·ω·fair_scale·ΔVar + Q
- Computes λ_t from driver/demand/uncertainty metrics

### Stage 5: Training (3 epochs)
1. For each epoch:
   - Train on predicted future requests (if available)
   - Train on historical requests
   - Update Q-tables
2. All three agents train on same data for fair comparison

### Stage 6: Evaluation on Test Week
1. Reset all agents (clear accumulated utilities, keep learned Q-tables)
2. Process all test requests (50,000 cap)
3. Compute metrics:
   - Total Utility: Σ o_d
   - Fairness: Var(o_d)
   - Normalized Fairness: std(o_d) / mean(o_d)
   - Min/Mean/Max driver utility

### Stage 7: Time Horizon Analysis (Optional)
1. Train agents (same as stage 5)
2. Evaluate day-by-day for 1-7 days
3. Track fairness and utility accumulation over horizon
4. Generate stability plot

---

## SECTION 6: EXPECTED RESULTS

### Paper's Reported Table 1 (for reference)
| Method | Total Utility | Fairness | Norm Fairness | Min | Mean | Max |
|--------|------|-----------|---------|-------|-------|-------|
| Greedy | -1,514,736 | 1,697 | -0.0005 | -75,804 | -75,737 | -75,654 |
| Proposed (MOMAQL) | 95,824 | 85,194 | 0.0610 | 4,566 | 4,791 | 5,932 |

### Expected Adaptive MOMAQL Results
- **Total Utility**: Similar to or higher than Proposed (due to targeted fairness spending)
- **Fairness**: Similar or slightly lower variance (more balanced allocation)
- **Norm Fairness**: Lower std/mean ratio (more equitable)
- **Min/Mean/Max**: More balanced across drivers

### Why Numbers May Differ from Paper
- Exact grid size, driver count, seeds not published
- MLP hidden width variations
- Shortest-path library differences
- Implementation details not specified

**The real validation** is qualitative: long-term fairness improves over time, unlike myopic greedy.

---

## SECTION 7: KEY TAKEAWAYS FOR PRESENTATION

### Main Contribution
We extend the paper's fixed-fairness MOMAQL with a state-dependent fairness weight that:
1. **Responds to imbalance**: Increases fairness pressure when drivers' utilities are unequal
2. **Avoids hotspot concentration**: Adjusts for regions with high demand
3. **Respects uncertainty**: Doesn't over-react to rare OD pairs
4. **Improves tradeoff**: Achieves better utility-fairness balance

### Why It Matters
- Fixed fairness is one-size-fits-all
- Adaptive fairness is context-aware
- Real platforms need dynamic fairness, not static rules

### Future Work
1. Multi-objective Pareto optimization
2. Hard-bound fairness constraints (not just soft penalty)
3. Driver preference learning
4. Dynamic parameter tuning based on real-time feedback

---

## FORMULAS SUMMARY (For Quick Reference)

**Driver Imbalance:**
I_driver = std(U) / |mean(U)|

**Local Demand Imbalance:**
I_local = max(0, max(D_pu, D_do) / D_mean - 1)

**Pair Uncertainty:**
I_unc = 1 / sqrt(count + 1)

**Adaptive Lambda:**
λ_t = λ · [1 + α₁·I_driver + α₂·I_local] · [1 + α₃·(1-I_unc)]

**Decision Score:**
score(d) = r - λ_t · ω · fair_scale · ΔVar(d) + Q

**Fairness (Variance):**
F = Var(U) = (1/n) Σ (U_i - mean(U))²

**Normalized Fairness:**
Fhat = std(U) / mean(U)

---

END OF SUMMARY


---

## UPDATE: MEASURED RESULTS (replaces the 'Expected Results' above)

Run with `MTP_longterm_fairness/run_experiments.py`; raw numbers in `MTP_longterm_fairness/results/*.csv`.
Seeds change only the agent (start positions, exploration); data split and predictor are fixed.

| Method (λ=1, 5 seeds) | Total utility | Variance | Std/mean |
|---|---|---|---|
| Greedy | 101,838 ± 144 | 38 ± 43 | 0.001 |
| Adaptive greedy | 101,749 ± 49 | 38 ± 49 | 0.001 |
| MOMAQL (paper) | 94,645 ± 680 | 79,456 ± 13,342 | 0.059 |
| Adaptive MOMAQL | 97,853 ± 535 | 2,991 ± 712 | 0.011 |

- Paper's Table 1 numbers for greedy are NOT reproduced (the paper does not publish graph size, driver count, MLP width, learning rate, epsilon, seed, shortest-path library). Our MOMAQL is close to the paper's (95,824 / 85,194 / 0.0610).
- Ablation (3 seeds): all three components together are best; local demand helps most.
- Lambda sweep (2 seeds): a fixed-lambda MOMAQL with lambda about 10 matches or beats adaptive at lambda=1, so part of the gain is a larger effective lambda. Adaptive over-penalises at base lambda >= 25.
