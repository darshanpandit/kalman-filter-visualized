# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

3Blue1Brown-style ManimCE video series explaining Kalman Filters and variants (KF, EKF, UKF, Particle Filter). Application domain: pedestrian trajectories and LBS mobility data. Swiss/Axonvibe-inspired dark theme.

## Commands

```bash
# Activate venv (Python 3.12 + ManimCE installed)
source .venv/bin/activate

# Run tests (75 tests: filters + math + generators + loader + benchmarks)
PYTHONPATH=. python3 -m pytest tests/ -v

# Run tests without venv (system Python 3.9, no manim needed)
PYTHONPATH=. python3 -m pytest tests/ -v

# Precompute benchmark results (generates .npz files for Part 5 scenes)
PYTHONPATH=. python3 benchmarks/precompute.py

# Render a scene (low quality for development)
PYTHONPATH=. manim -pql part1_kalman_filter/scene01_hook.py SceneHook

# Render high quality
PYTHONPATH=. manim -pqh part1_kalman_filter/scene01_hook.py SceneHook

# Render to PNG (last frame only, faster for checking visuals)
PYTHONPATH=. manim -ql --format png part1_kalman_filter/scene01_hook.py SceneHook
```

PYTHONPATH=. is required because scene files import from `kalman_manim/` and `filters/` at the project root. Scenes with `MathTex` or `NumberLine(include_numbers=True)` require LaTeX installed (`brew install --cask mactex-no-gui`).

## Architecture

Two independent layers:

1. **`filters/`** — Pure numpy/scipy filter implementations (no Manim dependency). Each file exports a filter class with `predict()`, `update()`, and `run()` methods.
   - `kalman.py` — Standard linear KF
   - `ekf.py` — Extended KF (takes `f`, `h`, `F_jacobian`, `H_jacobian` callables)
   - `ukf.py` — Unscented KF (sigma point generation via Cholesky, Van der Merwe weights)
   - `particle.py` — Particle Filter/SIR (systematic resampling, weighted particles)

2. **`kalman_manim/`** — ManimCE visual library.
   - `style.py` — Swiss color palette, timing constants, chart constants. Semantic aliases: COLOR_PREDICTION, COLOR_MEASUREMENT, COLOR_POSTERIOR, COLOR_PROCESS_NOISE. Comparison colors: COLOR_FILTER_KF/EKF/UKF/PF.
   - `utils.py` — `cov_to_ellipse_params()` (eigendecomposition), `gaussian_product_1d/2d()`, `gaussian_1d_pdf()`.
   - `mobjects/` — `GaussianEllipse`, `StateSpace`, `PedestrianPath`, `JacobianTangent`, `SigmaPointCloud`, `ParticleCloud`, `make_observation_note`, `RMSELineChart`, `FilterBarChart`, `ErrorHistogram`.
   - `animations/` — `animate_gaussian_multiply()`, `animate_predict_step()`, `animate_update_step()`, `animate_full_cycle()`.
   - `data/generators.py` — `generate_pedestrian_trajectory()`, `generate_linear_trajectory()`, `generate_nonlinear_trajectory()`, `generate_sharp_turn_trajectory()`, `generate_multimodal_scenario()`.
   - `data/loader.py` — `load_eth_trajectory()`, `load_trajectory()` (unified ETH+UCY), `list_available_trajectories()`.
   - `data/datasets/` — Vendored ETH (Pellegrini 2009) and UCY (Lerner 2007) pedestrian data.

3. **`benchmarks/`** — Pure numpy benchmark engine (no Manim dependency).
   - `metrics.py` — `position_rmse()`, `position_mae()`, `per_step_errors()`, `nees()`, `computation_time()`.
   - `configs.py` — Filter factory functions (`make_kf/ekf/ukf/pf()`, `make_all_filters()`). All use constant-velocity model.
   - `runner.py` — `run_single_trajectory()`, `run_corpus()`.
   - `sweep.py` — `sweep_turn_rate()`: parameter sweep varying nonlinearity.
   - `corpus.py` — `generate_synthetic_corpus()` (4 regimes × N), `load_real_corpus()` (ETH+UCY).
   - `precompute.py` — CLI to generate `data/benchmark_results/*.npz`.

4. **`part1_kalman_filter/`** through **`part5_benchmarks/`** — Scene files for each video.

## Key patterns

- `GaussianEllipse` takes an optional `axes` parameter for coordinate conversion. Without it, mean values map directly to scene coordinates.
- Scene files use `sys.path.insert(0, ...)` to ensure imports work when rendered via `manim` CLI.
- Color scheme: Swiss red (#e63946) for predictions, royal blue (#457b9d) for measurements, gold (#f4a261) for posteriors, teal (#2a9d8f) for process noise.
- Background color: #1a1a2e (set in `manim.cfg` and each scene's `construct()`).
- Python 3.9 compat: use `from __future__ import annotations` for `X | None` syntax.
- Filter interfaces are consistent: all have `predict(u=None)`, `update(z)`, `run(measurements)`.
- EKF/UKF take callable functions (f, h) instead of matrices. PF additionally takes noise in its transition function: `f(x, u, noise)`.

## Video series structure

- **Part 1** (8 scenes): Hook, Bayes foundations, 1D Gaussian multiplication, State space, Prediction step, Measurement update, Demo trajectory, Optimality
- **Part 2** (5 scenes): EKF motivation, Linearization/Jacobian, EKF vs KF equations, EKF demo on curved path, Failure modes
- **Part 3** (4 scenes): Key UKF insight, Sigma points visualization, UKF equations/parameters, UKF vs EKF demo
- **Part 4** (4 scenes): Beyond Gaussians, Particle mechanics (predict/weight/resample), PF demo, Grand comparison (all 4 filters + decision guide)
- **Part 5** (5 scenes): Why benchmark, RMSE vs nonlinearity sweep, Corpus dashboard (bar charts + histograms + timing), Deep dive (best/worst cases), Quantitative recommendations
