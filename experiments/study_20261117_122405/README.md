# Flappy Bird tabular Q-learning study

Recommended policy: **rewards_s424242**. All 58 original artifact files and the main branch remain unchanged. Nothing from this study was pushed to GitHub.

On 200 reserved, previously unseen test seeds:

| Metric | Original | Recommended |
|---|---:|---:|
| Mean pipes | 19.265 | 56.615 |
| Median pipes | 12 | 56 |
| 10th percentile | 2 | 7 |
| Best (capped) | 100 | 100 |
| Reach 100 pipes | 1% | 27.5% |

Total experimental training: **260,000 episodes / 53 minutes 32 seconds**. Selected agent: 24,000 episodes / 8 minutes 15 seconds. The experiments stayed under both approved ceilings. At report generation the complete study, including analysis and tests, had taken about 71.0 minutes. There are 20 independently trained experimental policies, plus the original baseline. Three training seeds were used for each finalist.

The winning policy changes only the rewards: survival +0.01, pipe +10, death -15. Bins, alpha .18, gamma .98 and the epsilon schedule match the original. Model selection used separate validation seeds and was frozen before test access. A documented near-median tie rule favored the stronger lower tail. Focused state ranges improved some replicas but were less reliable across training seeds. Fine grids, combined settings and lower alpha were stopped early when they underperformed the pilot control.

Read [the full report](REPORT.md), [all-policy test metrics](all_agents_test_comparison.csv), [the protocol](STUDY_PROTOCOL.md) and [recorded decisions](decisions.json).

Graphs:

- [All finalist test distributions](graphs/final_test_distributions.png)
- [Paired improvement over the baseline](graphs/paired_improvement.png)
- [Training curves](graphs/learning_curves.png)
- [Rewards, scores and epsilon for the recommended family](graphs/recommended_learning.png)
- [Training cost](graphs/training_cost.png)
- [Equal-budget screening](graphs/screening10000_validation.png)

To try the separate experimental agent using the unmodified original game:

```powershell
Set-Location "C:\Users\User\Documents\FlappyBird-RL"
.\.venv\Scripts\python.exe -m flappy_ai play --model "C:\Users\User\Documents\FlappyBird-RL-Experiments\experiments\study_20261117_122405\recommended_compatible_agent.npz"
```

The compatibility export removes only an unused optional metadata key; its Q-table was verified with the original main-branch loader. On the old demo seed 2026 it scores 100 versus the original's 17; this extra post-selection check did not drive model choice. The original default agent is not replaced.

To open the main comparison graph:

```powershell
Start-Process -FilePath "C:\Users\User\Documents\FlappyBird-RL-Experiments\experiments\study_20261117_122405\graphs\final_test_distributions.png"
```

There were **87 passing tests**. See tests.xml and tests_console.log. All models, configurations, incremental training histories and checkpoints are under the per-configuration/per-seed directories. Model hashes are in model_hashes.json. The runner scripts and source changes are at the experiment worktree root, two directories above this README.

Limitations: the selected policy still has short failures (worst test score 1), succeeds at the 100-pipe target on 27.5% of tested courses, and was tested at only one difficulty. Results beyond 100 pipes are censored by the target. Only three finalist training seeds were used; bootstrap intervals describe test-course uncertainty for fixed policies. Early-stopped variants were not tested to convergence. See REPORT.md for the full limitations and reproducibility details.

## Reproducing the recommended run later

The recorded study runner is deliberately bound to this study's immutable folders and wall-time deadline. For a fresh reproduction, use the saved configuration and a new output root instead of reusing old evaluation stages. This command was **not executed** as additional training:

```powershell
Set-Location "C:\Users\User\Documents\FlappyBird-RL-Experiments"
& "C:\Users\User\Documents\FlappyBird-RL\.venv\Scripts\python.exe" -m flappy_ai train --config "C:\Users\User\Documents\FlappyBird-RL-Experiments\experiments\study_20261117_122405\recommended_compatible_config.json" --episodes 24000 --output "C:\Users\User\Documents\FlappyBird-RL-Experiments\reproductions"
```

This starts a new table using training seed 424242 and saves to a new unique run folder. The original project and study results stay unchanged. The experiment source and pinned dependencies accompany the local worktree; model snapshots and per-seed evaluation files provide the evidence for this completed study.
