# Flappy Bird AI — Reinforcement Learning

A university project with an original procedurally drawn Pygame game and **genuine sparse tabular Q-learning**. No NEAT, neural networks, pretrained policy, or scripted flight controller. Physics runs at 60 simulation steps per second independently of display speed.

## Windows quick start

Python 3.11+ is required (tested here with Python 3.13). Run PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m flappy_ai play
```

Activation is unnecessary, so PowerShell execution-policy changes are unnecessary. `pygame-ce` supplies the `pygame` module.

The game defaults to 1,000 pipes. Scoring milestones appear at 100, 250, 500, 750 and 1,000. Passing the target pipe ends the flight in victory. Collisions and time limits end the flight separately.

In the menu, **1** selects the original agent and **2** selects the improved agent (the default). **T** switches between 100 and 1,000 pipes. **H** starts a human flight, **A** watches the selected agent, **C** starts a comparison, and **Q** quits. To choose an agent and target at launch, run one of these commands:

```powershell
.\.venv\Scripts\python.exe -m flappy_ai play --target 1000 --agent original
.\.venv\Scripts\python.exe -m flappy_ai play --target 1000 --agent improved
```

In flight: **Space** flap, **P** pause, **R** restart, **Esc** menu. **+ / -** adjusts simulation speed (0.25–8x). A comparison starts at 1x and reuses the exact seed and obstacle sequence; speed controls are locked during comparison for fair human reaction times. Press **Enter** after the human flight to start the AI. Each completed comparison exports a separate timestamp-and-UUID CSV under `artifacts/comparisons/`; previous sessions are never overwritten. The terminal prints the exact filename. A failed save is reported in both the terminal and result screen.

## Recorded agent performance

In the saved [long-game validation](validations/long_game_20261117T151230_923685Z_609359fb/comparison.csv), each frozen policy flew the same 440 seeded courses at each target.

| Saved policy | Average at 1,000 target | Best at 1,000 target | 1,000-pipe wins | 100-pipe wins |
| --- | ---: | ---: | ---: | ---: |
| Original | 19.1 | 157 | 0/440 | 2/440 |
| Improved | 76.4 | 499 | 0/440 | 122/440 |

The automated gameplay suite reaches 1,000 on a controlled course. The evaluation results above are the performance of the trained policies. The validation folder includes seed cohorts, hashes, and per-flight CSVs. Run `python verify_long_games.py` to reproduce the evaluation with the saved policies.

## Training

```powershell
# Fast, no display; all physics steps and Q updates still execute
.\.venv\Scripts\python.exe -m flappy_ai train --config configs/default.json --episodes 12000
# Watch training, pause or stop it
.\.venv\Scripts\python.exe -m flappy_ai train --episodes 2000 --render
# Continue a checkpoint for an additional 5000 episodes
.\.venv\Scripts\python.exe -m flappy_ai train --resume artifacts/agent.npz --episodes 5000
```

Every invocation creates a unique `artifacts/run_UTC-TIMESTAMP_UUID/` folder (or under `--output`). **The bundled `artifacts/agent.npz`, all 24 original checkpoints, and earlier results are never replaced by training.** To play a new checkpoint, pass `play --model PATH_TO_NEW_RUN/agent.npz`.

Each run contains:

- `training.csv`: append-only episode statistics, flushed and fsynced after every completed episode.
- `training.png`: three panels showing episode rewards and their rolling mean, scores and their rolling mean, and epsilon decay.
- `latest.npz`: atomic recovery checkpoint, saved initially, every 60 wall seconds, on pause, and on manual request.
- `checkpoint_EPISODE.npz`: periodic episode checkpoints (default every 1,000 episodes).
- `agent.npz`: final snapshot after normal completion, cooperative stop, Ctrl+C, or a recoverable exception.
- `config.json` and `status.json`: configuration and latest run status.

Visual training: **P** pauses/resumes and saves a checkpoint on pause; **C** saves a checkpoint; **S**, **Esc**, or closing the training window saves and stops. **+ / -** changes simulation steps between displayed frames. A checkpoint confirmation appears in the window and terminal.

Headless training in an interactive Windows terminal: **P** pauses/resumes, **C** checkpoints, **S** saves and stops (no Enter needed). **Ctrl+C** requests a cooperative stop; disk writes occur outside the signal handler. When stdin is redirected or keyboard polling is unavailable, control files work from another PowerShell window. Copy the exact run directory printed by training:

```powershell
$runPath = 'C:\Users\User\Documents\FlappyBird-RL\artifacts\run_REPLACE_WITH_ACTUAL_NAME'
New-Item -ItemType File -Path (Join-Path $runPath 'PAUSE')
# Resume after a file-based pause:
Remove-Item -LiteralPath (Join-Path $runPath 'PAUSE')
# Save a checkpoint without stopping (the application consumes this request file):
New-Item -ItemType File -Path (Join-Path $runPath 'CHECKPOINT')
# Save and stop, including while paused:
New-Item -ItemType File -Path (Join-Path $runPath 'STOP')
```

Use `--checkpoint-seconds 15` to shorten automatic recovery intervals. Console progress every 100 episodes includes episode, epsilon, score, reward and rolling average. Rolling averages restart per run. `--resume` loads the exact checkpoint configuration and creates another unique run folder. A checkpoint taken partway through an episode preserves learned transitions but does not save live environment state: that episode's obstacle seed is replayed on resume, using the saved exploration RNG state. This is not an exact continuation of the partial episode. Duration measures wall time, including pauses and checkpointing, excluding final graph export.

Abrupt Task Manager termination, console closure, power failure, or filesystem failure may bypass cleanup. Completed rows are durably flushed per episode, but the last in-progress CSV row can be incomplete if termination happens during its write. Q-values since the last completed checkpoint may be lost (normally up to the configured wall-time interval); CSV statistics can therefore extend beyond a recovery checkpoint. Atomic replacement keeps the preceding checkpoint valid if a replacement fails. No software can guarantee persistence when storage itself fails.

Regenerate all three graph panels from existing statistics **without training**:

```powershell
.\.venv\Scripts\python.exe -m flappy_ai plot --csv artifacts/training_all.csv --output artifacts/plots
```

The command prints the unique PNG path. To open the existing verification graph from anywhere in Windows:

```powershell
Start-Process -FilePath "C:\Users\User\Documents\FlappyBird-RL\artifacts\verification\training_graphs.png"
```

## Reproducible evaluation

```powershell
.\.venv\Scripts\python.exe -m flappy_ai evaluate --model artifacts/agent.npz
.\.venv\Scripts\python.exe -m flappy_ai evaluate --seeds 100000 100001 100002 100003 --output artifacts/custom_evaluation
.\.venv\Scripts\python.exe -m pytest -q
```

Each evaluation CLI invocation creates a unique evaluation subfolder under `--output` and prints its location, preserving previous results. The default evaluation uses 20 fixed, held-out seeds 100000-100019. Exploration is disabled and the Q-table is not modified. Outputs: per-seed CSV, score bar chart, and summary JSON containing average, best, success rate, seeds and recorded training duration. **Success means reaching the configured target of 100 pipes**, not merely passing one pipe. Report seed count and target alongside success rate. Training scores include exploration; evaluate the greedy policy separately. Results depend on training budget and discretization. See `RESULTS.md` for measurements actually produced in this workspace.

## Configuration

Edit `configs/default.json`; pass `--config` to train or to human play without a loaded model. Fields cover world dimensions, gravity, flap impulse, pipe gap/spacing/speed, bird radius, finite episode steps, target score, reward values, alpha, gamma, exploration schedule, bins and training seed. Change game difficulty before training. Validation rejects non-finite values, wrong types, invalid dimensions, non-positive pipe widths, impossible gaps/spacing, unsafe physics steps, invalid learning ranges and non-integer counts. Configuration changes that affect the environment require a new training experiment. A loaded model always uses its own saved configuration, keeping its discretization and game physics consistent. CLI `--episodes` overrides the episode count. Fixed seeds reproduce obstacle geometry independently of actions. Restart deliberately replays the same course; choose `play --seed 42` for another course.

Observation order and normalization ranges:

| Feature | Normalization range |
|---|---|
| Bird y | 0 to ground |
| Vertical velocity | flap velocity to maximum fall velocity |
| Next pipe x minus bird x | -pipe width - bird radius to world width |
| Bird y minus opening top | -ground to ground |
| Opening bottom minus bird y | -ground to ground |

Each value becomes a clamped integer bin: `floor((value-low)/(high-low)*bins)`. The five-bin tuple indexes a sparse dictionary of two Q values. State aliasing is a deliberate tabular approximation and can cause failures. Finite target and step caps prevent indefinite episodes; collision ends the episode immediately, each pipe rewards only once after safely clearing its trailing edge, and no artificial proximity reward is available to exploit. Arbitrary custom rewards can still encourage undesirable policies; compare score as well as return.

## Architecture

- `flappy_ai/environment.py`: deterministic fixed-step physics, circle/rectangle collisions, rewards, obstacles, observations, termination.
- `flappy_ai/agent.py`: normalization, sparse table, epsilon-greedy policy, Bellman updates, atomic compressed checkpoints (no pickle).
- `flappy_ai/training.py`: training, checkpointing, fixed-seed evaluation, CSV and Matplotlib exports.
- `flappy_ai/ui.py`: menus, animated bird, procedural scenery, pause/restart, comparison and training view.
- `flappy_ai/config.py`: typed configuration and validation.
- `flappy_ai/storage.py`: unique run folders, durable CSV journals, atomic status output and comparison exports.
- `flappy_ai/controls.py`: cooperative signals, Windows terminal keys and control files.
- `tests/test_core.py`: physics, collision boundaries, scoring, caps, updates, bins, greedy evaluation and serialization.

Q-learning uses `Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',.) - Q(s,a)]`. Collision and target termination remove the bootstrap term. Time limits are truncations and retain it. Exploration selects a uniformly random action with probability epsilon; otherwise the maximum Q value wins (ties choose no flap). Epsilon decays after each completed episode with a configured minimum. Environment and exploration random generators are separate; checkpoints preserve exploration generator state and resume episode seed numbering.

## Maintenance verification

The original project was uploaded first to the private GitHub repository `ElieAntoineGeagea/FlappyBird-RL` in baseline commit `c1cfe8c`. `.preservation-manifest.json` records SHA-256 hashes of all 53 original artifact files. Regression tests verify every hash, load all preserved policy results, and reevaluate the saved agent without learning. The Bellman update, state discretization, action selection and epsilon schedule are unchanged. No additional training of the bundled agent was performed during maintenance; tests create disposable agents in temporary directories.

Tests cover all seven project features through environment, serialization, evaluation and simulated Pygame events, plus invalid parameters, corrupted checkpoints, atomic replacement failures, incremental durability, graceful STOP-file shutdown and forced process termination. UI tests use SDL dummy video/audio; manual interaction and display timing are still hardware-dependent. Matplotlib/Pyparsing currently emits harmless third-party deprecation warnings.

If an AI checkpoint is missing or corrupt, play mode explains the problem and keeps human mode available. CLI failures show a concise error and exit code 2. A graph export failure leaves saved training CSV/checkpoints available for the `plot` command. The learned policy remains imperfect and the recorded 100-pipe success rate remains zero; maintenance fixes do not imply improved learning results.

No installation or launch confirmation remains necessary after the dependency approval given for this workspace. Running `play` opens a local window; saving outputs writes only to the chosen output folder. The game, training and evaluation run locally without paid services; GitHub hosts the project backup.
