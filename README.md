# Flappy Bird AI

A Pygame game with a trained tabular Q-learning agent. The game ends after 100 pipes, a collision, or the simulation step limit. The included reward-tuned policy, rewards_s424242, loads automatically when you launch the game.

## Install and play on Windows

Install Python 3.11 through 3.13. Extract the project ZIP and open PowerShell in the extracted folder.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m flappy_ai play
```

If Python is installed without the Windows launcher, use python in place of py. On macOS or Linux, create the environment with python3 -m venv .venv and use .venv/bin/python for the remaining commands.

- **H** starts Human Mode. Press **Space** to flap.
- **A** watches the included agent use its saved Q-table.
- **C** starts a human vs AI comparison. Fly first, then press **Enter** to watch the agent on the same course. The game writes a CSV to artifacts/comparisons/ after both flights.
- **P** pauses, **R** restarts, **Esc** returns to the menu and **Q** quits from the menu. **+ / -** adjusts playback speed outside comparison mode.

The default course seed is 2026. Use play --seed NUMBER for another course or play --model PATH to load a different checkpoint. Playback uses the checkpoint's game settings and runs with exploration disabled.

## Design and recorded results

The agent discretizes five observations into bins and stores two action values per visited state. The reward-tuned policy was trained for 24,000 episodes with seed 424242. On 200 held-out courses, it averaged 56.6 pipes and reached 100 in 55 runs (27.5%). The original policy averaged 19.3 and reached 100 in two runs (1%). The seed used for the demonstration is separate from that aggregate.

See TECHNICAL_SUMMARY.md for the design and artifacts/submission_metrics.csv for the recorded comparison. The full Git branch also preserves the original checkpoints and study archive.
