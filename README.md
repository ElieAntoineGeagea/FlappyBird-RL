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

Menu: **H** human, **A** trained AI, **C** comparison, **Q** quit. In flight: **Space** flap, **P** pause, **R** restart, **Esc** menu. **+ / -** adjusts simulation speed (0.25–8x). A comparison starts at 1x and reuses the exact seed and obstacle sequence; speed controls are locked during comparison for fair human reaction times. Press **Enter** after the human flight to start the AI. Completed comparisons export to `artifacts/comparison.csv` (one row per completed pair during the current application session).

## Training

```powershell
# Fast, no display; all physics steps and Q updates still execute
.\.venv\Scripts\python.exe -m flappy_ai train --config configs/default.json --episodes 12000
# Watch training, pause or stop it
.\.venv\Scripts\python.exe -m flappy_ai train --episodes 2000 --render
# Continue a checkpoint for an additional 5000 episodes
.\.venv\Scripts\python.exe -m flappy_ai train --resume artifacts/agent.npz --episodes 5000
```

Visual training: **P** pauses, **S** or **Esc** saves and stops, **+ / -** changes simulation steps between displayed frames. Headless training: **Ctrl+C** saves completed learning; restart with `--resume`. Console progress every 100 episodes includes episode, epsilon, score, reward, and rolling average. Checkpoints default to every 1,000 episodes. `agent.npz` is the latest policy. Each invocation writes a separate `training_FIRST_LAST.csv` and PNG, preserving earlier histories. A stopped partial episode's learned transitions are saved, but that episode is not counted in history and is replayed on resume. Duration measures active invocation wall time, including checkpointing and any visual pauses, excluding final plot generation.

## Reproducible evaluation

```powershell
.\.venv\Scripts\python.exe -m flappy_ai evaluate --model artifacts/agent.npz
.\.venv\Scripts\python.exe -m flappy_ai evaluate --seeds 100000 100001 100002 100003 --output artifacts/custom_evaluation
.\.venv\Scripts\python.exe -m pytest -q
```

The default evaluation uses 20 fixed, held-out seeds 10000–10019. Exploration is disabled and the Q-table is not modified. Outputs: per-seed CSV, score bar chart, and summary JSON containing average, best, success rate, seeds and recorded training duration. **Success means reaching the configured target of 100 pipes**, not merely passing one pipe. Report seed count and target alongside success rate. Training scores include exploration; evaluate the greedy policy separately. Results depend on training budget and discretization. See `RESULTS.md` for measurements actually produced in this workspace.

## Configuration

Edit `configs/default.json`; pass `--config` to train or to human play without a loaded model. Fields cover world dimensions, gravity, flap impulse, pipe gap/spacing/speed, bird radius, finite episode steps, target score, reward values, alpha, gamma, exploration schedule, bins and training seed. Change game difficulty before training. A loaded model always uses its own saved configuration, keeping its discretization and game physics consistent. CLI `--episodes` overrides the episode count. Fixed seeds reproduce obstacle geometry independently of actions. Restart deliberately replays the same course; choose `play --seed 42` for another course.

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
- `tests/test_core.py`: physics, collision boundaries, scoring, caps, updates, bins, greedy evaluation and serialization.

Q-learning uses `Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',.) - Q(s,a)]`. Collision and target termination remove the bootstrap term. Time limits are truncations and retain it. Exploration selects a uniformly random action with probability epsilon; otherwise the maximum Q value wins (ties choose no flap). Epsilon decays after each completed episode with a configured minimum. Environment and exploration random generators are separate; checkpoints preserve exploration generator state and resume episode seed numbering.

No installation or launch confirmation remains necessary after the dependency approval given for this workspace. Running `play` opens a local window; saving outputs writes only to the chosen output folder. No external account or paid service is used.
