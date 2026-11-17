# Flappy Bird AI
University project | Tabular reinforcement learning

## Design
The Pygame environment advances at 60 simulation steps per second. The bird can flap or do nothing. A point is awarded for each pipe cleared. The episode ends after a collision, 100 cleared pipes, or the step limit. A seed reproduces the pipe sequence.

The agent observes bird height, vertical velocity, horizontal distance to the next pipe, and the two vertical clearances to that pipe's gap. Each component is discretized into bins (8, 12, 12, 24, 24). A sparse table stores two action values for each visited state.

During training, the agent selected actions with an epsilon-greedy policy and updated the chosen value toward the received reward plus the discounted best value at the next state. Terminal updates use the reward alone. The learning rate was 0.18 and the discount factor was 0.98. The improved reward schedule gives +0.01 per surviving step, +10 per pipe and -15 for a collision.

## Saved agent and results
The included rewards_s424242 policy was trained for 24,000 episodes. Playback loads its saved NumPy checkpoint and chooses actions greedily. On 200 held-out courses, the improved policy averaged 56.6 pipes and reached 100 in 55 runs (27.5%). The original averaged 19.3 and reached 100 in two runs (1%). The demo shows the improved policy on seed 2026. Success varies by course.

## Run the submission
Extract the ZIP, install Python 3.11 through 3.13, and open PowerShell in the extracted folder.

    py -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m flappy_ai play

Press A to watch the included model, H to fly manually (Space to flap), or C to compare both on the same course. In comparison mode, press Enter after the human flight to start the agent. The game saves the scores and seed under artifacts/comparisons/.
