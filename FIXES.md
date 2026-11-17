# Maintenance fixes and validation

The original project was uploaded first to the private GitHub repository https://github.com/ElieAntoineGeagea/FlappyBird-RL (baseline commit c1cfe8c).

## Changes

- Built-in training graphs now include episode rewards and their 100-episode mean, scores and their 100-episode mean, and epsilon decay. The `plot` command regenerates them from saved CSV files without training.
- Each training invocation gets a unique run folder. The bundled trained agent, original checkpoints and old statistics are never used as output destinations.
- Training saves initial, timed, periodic, on-demand, pause and final checkpoints. Checkpoint writes flush and fsync before atomic replacement; failed replacement leaves the previous checkpoint intact.
- Visual training supports P pause/resume, C checkpoint and S/Esc save-and-stop. Headless Windows terminals support P/C/S, cooperative Ctrl+C, and PAUSE/CHECKPOINT/STOP control files, including when stdin is redirected.
- Completed episode statistics are appended and fsynced immediately. A run status JSON records completion, stop, interruption or failure. Plotting errors cannot discard saved statistics or checkpoints.
- Validation rejects negative/zero pipe widths, non-finite values, wrong types, invalid dimensions, impossible pipe geometry, unsafe physics steps, invalid learning ranges and fractional counts. Corrupt checkpoint structures and metadata fail with useful messages.
- Every completed human/AI comparison uses a unique timestamp-and-UUID CSV. Save errors are visible in the result screen and terminal. Evaluation CLI runs also use unique output folders.
- A missing or corrupt AI model leaves human play available. CLI errors produce a concise message and exit code 2. The README documents new paths, controls, recovery and graph commands.
- Git attributes preserve raw artifact bytes, preventing line-ending normalization of experimental evidence.

## Validation

**76 automated tests passed on Windows with Python 3.13.7.** See `artifacts/maintenance_verification/tests.xml` for the machine-readable result. Tests include real subprocess graceful shutdown and forced termination, durable incremental statistics, checkpoint replacement failure, control files, Windows keyboard polling, corrupted files, invalid parameters, reward graphs, human flap/pause/restart, game speed, comparison speed lock, unique exports and deterministic evaluation.

All **53 original artifact files** retain their SHA-256 hashes, including `artifacts/agent.npz`, all **24 checkpoints**, and every previous training/evaluation result. The manifest is `.preservation-manifest.json`.

The original and updated ASTs of the agent constructor, observation discretization, Q lookup, action selection, Bellman update and epsilon-decay methods were compared and are unchanged. The original trained agent was **not retrained**. Tests use newly created disposable agents in temporary folders.

Both evaluation sets reproduce their original summaries exactly: 20.45 mean / 68 best for seeds 100000-100019 and 18.4 mean / 72 best for seeds 200000-200019. Success remains 0% for the 100-pipe target. Default play seed 2026 still scores 17. The CLI evaluation was also exercised against the saved policy and wrote a new maintenance output folder.

The revised built-in graph was generated from the existing 24,000-episode CSV and visually inspected. Dependency validation (`pip check`), compilation and Git whitespace checks passed.

## Remaining limitations

Forced termination or storage failure can bypass final cleanup. A final in-progress CSV row may be incomplete if killed during its write, and Q updates since the latest completed checkpoint may be lost. Earlier flushed rows and the prior atomic checkpoint remain available. A mid-episode checkpoint stores learned values and exploration RNG state, but not the live environment: resuming replays that episode's seed rather than continuing at the exact physics step.

UI automation tests use SDL dummy video/audio and simulated events; physical keyboard feel and live display timing remain hardware-dependent. The test run emits 14 harmless third-party Matplotlib/Pyparsing deprecation warnings. No new learning performance is claimed.

Launch in the project directory:

```powershell
.\.venv\Scripts\python.exe -m flappy_ai play
```
