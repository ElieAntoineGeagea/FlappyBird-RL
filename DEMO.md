# Three-minute demonstration

**0:00–0:25 — Goal.** Launch `python -m flappy_ai play` using the virtual-environment interpreter. Explain: two actions, five observations, a discrete Q-table, no neural network.

**0:25–0:55 — Human flight.** Choose H, use Space to flap, demonstrate P pause and R restart. Show procedural graphics and explain fixed-step physics.

**0:55–1:25 — Training.** In another terminal run `python -m flappy_ai train --resume artifacts/agent.npz --episodes 100 --render --output artifacts/demo`. Show epsilon, reward, episode and average. Use +/- and P; press S to save and stop. Explain the Bellman update and why headless training is faster.

**1:25–2:00 — AI and comparison.** Return to the game, choose A to show the bundled trained table. Choose C for a human attempt; Enter replays the course for the AI. Explain that both see the same seeded obstacles. If a flight lasts too long, Esc returns to the menu.

**2:00–2:35 — Evidence.** Show RESULTS.md, the learning curve and evaluation graph. State the measured average, best, target-based success rate and seed count, including any weak result. Explain training exploration versus greedy evaluation.

**2:35–3:00 — Engineering.** Show the JSON configuration, checkpoint files and `python -m pytest -q`. Mention state aliasing and finite episode limits. Finish with one improvement: a systematic bin/hyperparameter study on separate validation seeds.

Use `.\.venv\Scripts\python.exe` in place of `python` unless the environment is activated. Prepare the learning curve and results before the presentation; a full training run is not expected to finish during three minutes.
