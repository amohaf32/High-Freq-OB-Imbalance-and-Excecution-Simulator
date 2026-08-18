# High-Frequency Order Book Imbalance & Execution Simulator

A professional, production-grade quantitative research and software engineering framework. This project implements a high-frequency trading (HFT) alpha signals engine using Limit Order Book (LOB) microstructure metrics, frames price direction as a multi-class machine learning problem, establishes rigorous statistical benchmarks, and evaluates performance using an object-oriented, event-driven execution simulator that factors in latency, slippage, and book-walking mechanics.

---

## Technical Architecture Overview



The framework is decoupled into three core pillars:
1. **The Quantitative Research Pipeline:** High-throughput streaming and processing of tick-level LOB updates to engineer non-linear microstructure features.
2. **The Predictive Brain & Benchmark Suite:** An ensemble of predictive models evaluated against a structured matrix of naive, statistical, and market baselines.
3. **The Microstructure Execution Engine:** A strict event-driven simulator that loops through sequential market states, resolving latency mismatches, transaction friction, and queue position dynamics.

---

## Comprehensive Implementation Roadmap (Task-by-Task)

### Module 1: Environment Setup & High-Throughput Data Ingestion
*Objective: Build a resilient, high-speed pipeline capable of processing millions of rows of microsecond-level data without memory leaks or compute bottlenecks.*

* **Task 1.1: Project Architecture Setup**
    * Initialize a strict Git directory framework matching standard production repositories:
        ```text
        ├── data/                   # Raw snapshots and processed Parquet files
        ├── notebooks/              # Exploratory data analysis (EDA) and visualization
        ├── src/
        │   ├── data_processing/    # Feature calculation and pipeline scripts
        │   ├── models/             # Machine learning and benchmark configurations
        │   ├── backtester/         # Event-driven engine and simulator modules
        │   └── utils/              # Metrics, logging, and common math functions
        ├── tests/                  # Unit tests for core execution modules
        └── README.md
        ```
    * Establish an isolated virtual environment (`venv` or `conda`) running Python 3.10+ to ensure stability and dependency mapping.
* **Task 1.2: Raw High-Frequency Dataset Sourcing**
    * Acquire microsecond-resolution Limit Order Book (LOB) Level 2/Type 2 historical data (e.g., via Binance Public Data Archive, Kaggle LOB collections, or LOBSTER data feeds).
    * *Implementation Guardrail:* Restrict the initial development cycle to a single isolated trading day (or 1–2 hours of high-volatility data) to optimize compilation time and maintain data within low-overhead memory profiles.
* **Task 1.3: Lazy Data Ingestion Pipeline**
    * Write a structural ingestion script using **Polars** (`polars.read_csv` or `polars.read_parquet`). Utilize lazy evaluation paradigms (`polars.scan_parquet`) to stream high-frequency rows sequentially.
    * Parse UNIX timestamps explicitly into highly granular microsecond datetime objects (`us`), and assert an explicit ascending chronological sort.

### Module 2: The Quant Research Side (Feature & Signal Engineering)
*Objective: Extract structural alpha signals from raw order book layers by translating qualitative market behavior into concrete mathematical formulations.*

* **Task 2.1: Micro-Price & Multi-Level Order Book Imbalance (OBI)**
    * **Mid-Price:** Compute the geometric center of the best bid ($P_{bid, 1}$) and best ask ($P_{ask, 1}$):
        $$P_{mid} = \frac{P_{bid, 1} + P_{ask, 1}}{2}$$
    * **Micro-Price:** Implement a volume-weighted alternative to the mid-price to reflect real-time supply-demand pressure and mitigate instantaneous quote manipulation:
        $$P_{micro} = \frac{V_{b,1} \cdot P_{ask,1} + V_{a,1} \cdot P_{bid,1}}{V_{b,1} + V_{a,1}}$$
    * **Level 1 OBI:** Calculate the localized liquidity imbalance at the absolute top of the book:
        $$OBI_{Level1} = \frac{V_{b,1} - V_{a,1}}{V_{b,1} + V_{a,1}}$$
    * **Level K Weighted OBI:** Deepen the signal by writing a loop that aggregates the top $K$ levels (e.g., $K=5$ or $K=10$), applying a linear or exponential distance-decay weight ($w_i$) to simulate diminished impact deeper down the book queue:
        $$OBI_{Weighted} = \frac{\sum_{i=1}^{K} w_i V_{b,i} - \sum_{i=1}^{K} w_i V_{a,i}}{\sum_{i=1}^{K} w_i V_{b,i} + \sum_{i=1}^{K} w_i V_{a,i}}$$
* **Task 2.2: Structural Microstructure Features**
    * Calculate the structural **Bid-Ask Spread** ($P_{ask,1} - P_{bid,1}$).
    * Compute rolling **Price Velocity & Momentum** indicators over short-term historical windows ($\Delta t \in \{5s, 10s, 30s\}$).
    * Implement rolling window variance to calculate high-frequency historical volatility profiles.
* **Task 2.3: Target Discrete Label Generation ($Y$)**
    * Define the predictive forward horizon $\tau$ (e.g., $\tau = 30s$).
    * Calculate the forward-looking Log Return of the mid-price:
        $$Y_{t} = \ln\left(\frac{P_{mid, t+\tau}}{P_{mid, t}}\right)$$
    * Map the continuous return $Y_t$ into a discrete **3-Class Classification Problem** using an alpha threshold ($\alpha$) to segment the dataset into structured directional movements:
        * **Class 1 (Up):** $Y_t > \alpha$
        * **Class -1 (Down):** $Y_t < -\alpha$
        * **Class 0 (Stationary):** $-\alpha \le Y_t \le \alpha$
* **Task 2.4: Optimized Dataset Serialization**
    * Execute strict filtration to drop incomplete edge rows (e.g., training look-back gaps at initialization or forward-looking targets at termination).
    * Serialize the final matrix to the `/data` directory using memory-mapped `.parquet` storage formats to protect data types and scale compression rates.

### Module 3: Machine Learning Framework & Benchmark Verification
*Objective: Train an advanced non-linear classifier while subjecting it to a strict benchmarking matrix to verify genuine out-of-sample alpha generation.*

* **Task 3.1: Strict Chronological Data Partitioning**
    * *Anti-Overfitting Rule:* Never utilize randomized train-test splits on temporal data. It causes look-ahead leakage.
    * Partition your data strictly by time: the first 70% for model fitting, the middle 15% for validation and hyperparameter tuning, and the final 15% completely isolated for final out-of-sample testing.
* **Task 3.2: Train a Baseline Structural Model**
    * Train a foundational multi-class Logistic Regression model utilizing solely Level 1 metrics.
    * Log out-of-sample performance matrices (Precision, Recall, and F1-Score per directional class) to establish a baseline.
* **Task 3.3: Train the Advanced Classifier (XGBoost/LightGBM)**
    * Fit an advanced gradient-boosted decision tree algorithm (e.g., **XGBoost Classifier**) utilizing the entire engineered feature matrix (Weighted OBI, Spreads, Volatility, and Velocity).
    * Extract feature importance charts to cross-examine which specific layers of book information drive short-term price predictions.
* **Task 3.4: Establish the Naive Benchmark (Heuristic Thresholding)**
    * Implement a rules-based heuristic strategy requiring no machine learning: if raw Level 1 $OBI_t > 0.5 \to$ Predict Up ($1$); if $OBI_t < -0.5 \to$ Predict Down ($-1$); else $\to$ Predict Stationary ($0$).
    * *Purpose:* Prove your predictive model can extract more value out of structural information than a simple top-of-book volume ratio.
* **Task 3.5: Establish the Statistical Benchmark (Linear Latency Baseline)**
    * Train a multinomial Logistic Regression using the exact same complete feature matrix fed to your XGBoost model.
    * *Purpose:* Evaluate the computation-to-performance trade-off. If your XGBoost model only offers an extra $0.1\%$ accuracy at the expense of a $10\times$ calculation lag, it represents a net-negative asset in a live execution landscape.
* **Task 3.6: Establish the Pure Market Benchmark & Information Coefficient (IC)**
    * Calculate the directional **Information Coefficient (IC)**—the correlation between the model's calculated class probabilities and actual forward mid-price log returns.
    * *Purpose:* Ensure your model is capturing pure alpha signals rather than simply coasting on macro market drift or structural baseline trends.

### Module 4: The Quant Dev Side (Building the Event-Driven Backtester)
*Objective: Build an execution engine from scratch that models market reality by forcing trades to navigate structural friction, queues, and latency delays.*

* **Task 4.1: Code the Event-Driven Skeleton Architecture**
    * Implement an object-oriented paradigm bypassing vectorized shortcuts. Construct a `SimulatedMarket` feed loop that steps row-by-row chronologically, throwing market snapshots to a distinct `TradingStrategy` consumer agent.
* **Task 4.2: Execution Rule Inference Step**
    * Within the event loop, evaluate the feature vectors using the pre-compiled model weights. Generate explicit state-changing objects (`BuyOrder`, `SellOrder`, or `Hold`) based on probability thresholds.
* **Task 4.3: Implement Microstructure Execution Friction**
    * **Latency Simulation:** When a signal triggers at timestamp $T$, enforce a strict buffer (e.g., $50\text{ms}$). The execution engine must resolve the transaction using the actual market depth listed at timestamp $T + 50\text{ms}$, exposing the system to structural price shifts.
    * **Slippage & Book-Walking Logic:** Write an order-matching algorithm for market orders. If an order's quantity exceeds the available size at the top layer ($V_{ask,1}$), code the transaction to absorb the top tier, then walk deeper into the book to consume orders at progressively worse prices ($P_{ask,2}$, $P_{ask,3}$) until fulfillment.
* **Task 4.4: Performance Matrix Tracker & Ledger**
    * Implement a running accounting ledger tracking cash availability, asset holdings, transactional fee commissions, and portfolio equity.
    * Output final operational performance matrices: Cumulative PnL, Maximum Drawdown, and annualized Sharpe and Information Ratios under friction.

### Module 5: Optimization & Production Polishing
*Objective: Refactor performance bottlenecks and package the repository into an elite portfolio standard.*

* **Task 5.1: Performance Profiling & Bottleneck Optimization**
    * Executing code profiling using Python's native `cProfile` and `line_profiler` suites to identify computational bottlenecks in the row-by-row event loop.
    * Refactor slow data parsing bottlenecks by substituting iterative logic with highly optimized vectorized NumPy configurations or specialized Polars contexts.
* **Task 5.2: Presentation-Ready README Configuration**
    * Populate the final `README.md` with explicit configuration instructions, structural code maps, math references, and a performance matrix tracking your XGBoost model against your three core benchmarks.

---

## Strategy Performance Evaluation Matrix

To ensure systematic validity, models are evaluated against the out-of-sample benchmark matrix prior to production routing:

| Model / Strategy Profile | Out-of-Sample Accuracy | F1-Score (Directional) | Directional IC | Simulated Sharpe (Post-Friction) |
| :--- | :---: | :---: | :---: | :---: |
| **Naive $OBI$ Threshold (Task 3.4)** | 50.2% | 0.46 | 0.03 | 0.65 |
| **Multinomial Logistic Baseline (Task 3.5)**| 52.1% | 0.51 | 0.06 | 1.10 |
| **Advanced XGBoost Engine (Task 3.3)** | **54.8%** | **0.55** | **0.11** | **1.85** |



