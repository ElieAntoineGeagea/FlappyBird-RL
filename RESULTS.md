# Measured results — 17 November 2026

These results were generated in this workspace; no results are simulated or estimated.

| Policy | Evaluation seeds | Mean pipes | Best pipes | Reach 100 pipes |
|---|---|---:|---:|---:|
| Untrained greedy table | 100000–100019 | 0.00 | 0 | 0% |
| 12,000-episode checkpoint | 100000–100019 | 3.35 | 11 | 0% |
| 24,000-episode final checkpoint | 100000–100019 | 20.45 | 68 | 0% |
| Final checkpoint, fresh seed set | 200000–200019 | 18.40 | 72 | 0% |

Each evaluation contains 20 episodes with exploration disabled. Both evaluation seed sets are outside the training sequence (42–24041). The benchmark set was reused to measure progress; the fresh seed set was evaluated only after final training. No hyperparameters changed between the two training stages. This is a modest benchmark sample, and the agent still collides; it has not solved the 100-pipe objective.

Training comprised two headless runs of 12,000 episodes using default configuration and exploration seed 42. Recorded cumulative training wall time: **430.81 seconds (7.18 minutes)**. Final epsilon: 0.01. Stored discrete states: 7,649. Final training rolling average: 5.46; this includes exploration and differs from greedy evaluation. Maximum observed training score: 42.

Training used Windows, Python 3.13.7 and existing NumPy 2.4.4 while the virtual environment installed. Final evaluation and tests used the isolated dependencies in requirements-lock.txt (NumPy 2.2.6, Pygame CE 2.5.6, Matplotlib 3.10.3, pytest 8.4.2). The same checkpoint loads successfully across these NumPy versions. Exact timings depend on hardware and concurrent work.

Validation: **18 automated tests passed**. Pygame was tested using SDL's dummy video/audio drivers, including menu transitions, comparison CSV export, restart, training pause/resume and stop. Game/menu PNGs were visually inspected. `pip check` found no broken requirements. Matplotlib emitted third-party Pyparsing deprecation warnings; they did not affect outputs. Physical keyboard feel and live display timing remain available for manual checking through play mode.

Artifacts:

- `artifacts/agent.npz`: final trained Q-table and full configuration.
- `artifacts/checkpoint_12000.npz`, `artifacts/checkpoint_24000.npz`: stage endpoints; intermediate checkpoints every 1,000 episodes.
- `artifacts/training_all.csv`, `artifacts/learning_curve.png`: complete 24,000-episode history and curve.
- `artifacts/training_1_12000.*`, `artifacts/training_12001_24000.*`: original per-run histories.
- `artifacts/evaluation/`: final benchmark CSV, graph and summary.
- `artifacts/evaluation_fresh/`: fresh-seed CSV, graph and summary.
- `artifacts/evaluation_12000/`, `artifacts/evaluation_untrained/`: earlier measurements.
- `artifacts/game_preview.png`, `artifacts/menu_preview.png`: actual procedural interface renders.

Reproduce the final benchmark with `.\.venv\Scripts\python.exe -m flappy_ai evaluate`. Retraining uses deterministic episode seeds and checkpointed exploration state, but floating-point/library differences can affect learning trajectories. Human comparison results are generated only after someone completes both flights; no human scores have been invented.
