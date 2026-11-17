# Flappy Bird AI — controlled tabular Q-learning experiments

All results below are measured in this workspace. No neural network, evolutionary algorithm or scripted flight controller was used. The original agent was neither overwritten nor promoted. The main branch and remote repository have not been changed by this study.

## Recommendation

Recommended experimental agent: **rewards_s424242**, copied unchanged to `recommended_agent.npz`. Selection was frozen using validation courses before any locked-test result was computed. It remains an experimental option; the original game still defaults to the original Q-table.

The recommended agent retains the original bins [8,12,12,24,24], alpha 0.18, gamma 0.98 and epsilon schedule. Its reward changes are survival +0.01 (was +0.05) and death -15 (was -10); passing remains +10. It was trained independently with seed 424242. No encoder change is needed for this recommended policy.

| Split | Policy | Courses | Mean | Median | P10 | Best | Success % |
|---|---|---|---|---|---|---|---|
| benchmark | Original | 40 | 19.43 | 10.50 | 2.90 | 72 | 0.00 |
| benchmark | Recommended | 40 | 50.23 | 41.00 | 11.70 | 100 | 15.00 |
| validation | Original | 100 | 18.37 | 12.50 | 1.00 | 100 | 2.00 |
| validation | Recommended | 100 | 57.80 | 52.00 | 11.80 | 100 | 34.00 |
| test | Original | 200 | 19.27 | 12.00 | 2.00 | 100 | 1.00 |
| test | Recommended | 200 | 56.62 | 56.00 | 7.00 | 100 | 27.50 |


P10 is the linearly interpolated 10th percentile. Success means reaching **100 pipes**; scores are capped at 100 by the same environment for every policy. These are course samples, not guarantees for every future flight.

On the 200 paired held-out courses, the mean score difference is **37.35**, with a 95% paired bootstrap interval **[31.92, 42.84]**. The median difference is **44.00** (interval [30.50, 53.00]). P10 difference: **5.00** (interval [2.90, 8.12]). The recommended policy outscored the original on 79.0% of matched courses and tied on 5.0%.

The worst recorded selected score is **1**, versus **0** for the original. The mean score of the worst 10% of test courses is **4.20**, versus **1.05**. These distinguish typical gains from remaining catastrophic failures.

Bootstrap intervals resample course pairs with fixed trained models (5,000 resamples); they do not represent uncertainty over all possible training seeds. The three independent finalist replicas below expose some training variability. All test results, including weak candidates, are retained in the full table and CSV.

## Comparable 24,000-episode policies on the locked test

| Policy / training seed | Episodes | Mean | Median | P10 | Best | Success % | Train min |
|---|---|---|---|---|---|---|---|
| baseline | 24000 | 19.27 | 12.00 | 2.00 | 100 | 1.00 | 7.18 |
| rewards_s424242 | 24000 | 56.62 | 56.00 | 7.00 | 100 | 27.50 | 8.24 |
| rewards_s848484 | 24000 | 55.24 | 50.00 | 6.00 | 100 | 31.50 | 9.22 |
| focused_s424242 | 24000 | 52.68 | 45.50 | 8.90 | 100 | 23.00 | 6.98 |
| rewards_s42 | 24000 | 50.28 | 43.50 | 6.00 | 100 | 23.50 | 8.16 |
| focused_s42 | 24000 | 40.59 | 31.00 | 4.00 | 100 | 13.00 | 6.89 |
| focused_s848484 | 24000 | 10.46 | 8.00 | 1.00 | 83 | 0.00 | 5.43 |


Each row has 200 identical obstacle seeds, exploration disabled and its own frozen Q-table. The original has one historical training replicate; each experimental finalist has three. Reusing identical obstacle courses across rows provides paired comparisons; these are not 1,400 independent courses.

## Why the original fails

The original discretizes both signed opening-edge distances across [-590,590] into 24 bins, roughly **49.17 pixels per bin**, and horizontal distance into roughly **46.92 pixels per bin**. The two edge features are correlated (their raw sum is the 175-pixel gap), so their joint quantization is finer than an individual 49-pixel bin; combined relative-distance boundaries are spaced about 21.7 or 27.5 pixels apart before other features are considered. Per-feature widths therefore overstate the full tuple's vertical aliasing. Nevertheless, the state can still combine materially different clearances and approach timings. On 100 validation courses, the original failed 53 times on upper pipes and 45 times on lower pipes; two reached the target. There were no ceiling or floor deaths. No terminal decision state was absent from the table, and the mean unknown-state fraction was about 0.0068%. Mere Q-table coverage therefore does not explain most original failures.

The median horizontal distance immediately before a pipe collision was 14 pixels, close to the 13-pixel collision radius. Lower-pipe deaths often occurred after a last-moment flap (about 76% of those deaths): the correction was too late. This supports timing/clearance aliasing as a plausible mechanism, but does not prove a unique causal explanation for every death.

Original gamma .98 corresponds to a rough effective horizon of 50 simulation steps; pipes arrive about 80 steps apart. Increasing gamma to .995 improved screening but was not one of the two most robust finalists. The original epsilon schedule wastes early episodes on frequent random flaps; lowering initial exploration helped early progress but did not dominate later screening. A smaller alpha .08 learned more slowly at the pilot budget and was stopped; this does not establish its asymptotic quality. Reducing survival reward and increasing death penalty improved screening under the original state grid. No reward exploitation or infinite episodes was observed; original score and step caps were retained.

## Pilot, pruning and controlled comparisons

`STUDY_PROTOCOL.md`, `study.json`, `decisions.json`, `finalists_plan.json` and `selection.json` preserve the protocol, amendments and timestamped decisions. All eight initial configurations received two 3,000-episode pilots. Every median was zero, so no arm was rejected from that insufficient pilot. All continued to 6,000 episodes. The fine grid, fine-grid combined settings and alpha .08 then underperformed control on both seeds. Their paired validation mean differences were negative, with bootstrap intervals excluding zero. They were stopped at 6,000 to prioritize sample efficiency, not labeled universally ineffective.

Five original survivors advanced to 10,000 episodes. An additional focused-range representation was preregistered after the failure analysis and tested first at 6,000, then 10,000. It uses **[4,8,12,24,24]** bins and observation ranges **[[0,590],[-6.5,10],[-83,280],[-80,255],[-80,255]]**. All five observations remain. Relative-distance resolution is about **13.96 pixels**, while redundant absolute-height and velocity resolution is reduced. Its nominal table has **221,184** possible keys versus **663,552** for the original; only encountered states are stored. This focuses representation rather than simply enlarging the table. The focused relative ranges are symmetric about half the opening height, making the two normalized edge features complementary; coarser redundant features and altered range alignment may both contribute. Their individual causal effects were not isolated, so the study does not attribute the gain exclusively to finer distance bins. Far-away values are clipped, which may hurt other tasks or difficulty settings.

All unspecified focused hyperparameters and rewards are original defaults. State normalization is the only policy-method change for that arm; the Bellman update, epsilon-greedy action selection and epsilon decay code remain unchanged. The standard intermediate grid [8,12,16,36,36] and fine grid [8,16,20,60,60] were also measured. Increasing uniform resolution alone did not win this budget-limited study.

Equal-budget 10,000-episode validation screening (two training replicas per configuration):

| Configuration | Worst replica median | Mean replica median | Mean score | Worst replica P10 |
|---|---|---|---|---|
| focused | 8.50 | 22.25 | 28.98 | 2.00 |
| rewards | 5.00 | 6.50 | 9.21 | 1.00 |
| gamma | 2.00 | 3.00 | 4.30 | 1.00 |
| control | 2.00 | 2.00 | 2.80 | 0.90 |
| epsilon | 2.00 | 2.75 | 4.00 | 0.00 |
| intermediate | 1.00 | 1.00 | 1.44 | 0.00 |


Finalists were ranked by **worst replica median**, then worst P10, worst lower-tail mean, mean replica median and mean score. Maximum score was never a selection criterion. The selected families each received three independent 24,000-episode runs. Before opening the locked test, a practical near-tie rule was recorded: within the winning family, replicas with validation median at least 90% of the best were considered practically close, then ranked by P10, lower-tail mean, median and mean score. This avoids choosing a marginally higher median with substantially worse lower-tail behavior. This validation-informed amendment was made after the first two reward replicas, not preregistered at study inception; it is documented in decisions.json. There was no retraining or retuning after test access.

## Seeds, preservation and implementation checks

Training exploration seeds are 42, 424242 and 848484; obstacle seed is training_seed + episode index. Even at 24,000 episodes their ranges exclude all evaluation sets. Historical benchmarks are 100000-100019 and 200000-200019; validation is 300000-300099; locked testing is 900000-900199. The original and every experimental agent were evaluated on the same 40 benchmark, 100 validation and 200 held-out test courses. Earlier pilot/screening snapshots retain their benchmark and validation evaluations; the final retained checkpoint for every training replica receives the complete test comparison.

The control trained with seed 42 reproduced **all Q-values exactly** in the original 10,000-episode checkpoint. The original **58 artifact files**, including the trained agent, all 24 original checkpoints and previous results, still match their SHA-256 hashes. Main remains at `bbda37d8219373350cf0173fec9291fb45233ba7` with a clean working tree. No push was made. Experimental code and results are in a separate local worktree/branch.

A transient Windows file-sharing lock interrupted the first control run after a completed episode. Its saved checkpoint and RNG state were resumed; no completed learning was discarded. Bounded file-replacement retries were added in this experiment worktree only, and status JSON refresh was reduced to every 100 episodes. CSV statistics still flush/fsync each episode and checkpoints save at least every 30 seconds. These infrastructure adjustments do not alter numerical learning. All **87 tests passed**; see `tests_console.log` and `tests.xml` for the regression suite, including original artifact preservation, experimental range/seed checks, robust ranking, paired statistics and deadline handling.

## Actual resource cost

- New experimental training: **260,000 completed episodes**, **75,822,323 simulation steps**.
- Sum of recorded training wall time across all independent policies: **3211.62 seconds (53.53 minutes)**. Cumulative continuation times are counted once per replica, including the recovered control run.
- End-to-end study elapsed time at report generation, including preparation, evaluation, tests, code and analysis: **71.05 minutes**.
- Original baseline's historical training time: **7.18 minutes**, excluded from new experimental cost.
- Recommended policy's cumulative training time: **8.24 minutes**.
- Approved limits: at most 264,000 experimental episodes and four wall hours. Both respected. The unused episode allowance was not consumed for its own sake.

Longer survival increases simulation steps and cost per episode. These are measured Windows wall times, including durable I/O; lightweight tests/analysis sometimes ran concurrently, so small timing differences are not algorithmic speedup evidence; they are not hardware-independent compute benchmarks. Better policies need not train faster. All game parameters, simulation timestep, target score and termination limits stayed fixed.

## All retained experimental agents on the test set

Stopped arms have 6,000 episodes, non-finalists 10,000, finalists 24,000. These unequal-budget rows are not a causal equal-budget ablation; use the 10,000-episode table for that comparison.

| Policy / training seed | Episodes | Mean | Median | P10 | Best | Success % | Train min |
|---|---|---|---|---|---|---|---|
| baseline | 24000 | 19.27 | 12.00 | 2.00 | 100 | 1.00 | 7.18 |
| rewards_s424242 | 24000 | 56.62 | 56.00 | 7.00 | 100 | 27.50 | 8.24 |
| rewards_s848484 | 24000 | 55.24 | 50.00 | 6.00 | 100 | 31.50 | 9.22 |
| focused_s424242 | 24000 | 52.68 | 45.50 | 8.90 | 100 | 23.00 | 6.98 |
| rewards_s42 | 24000 | 50.28 | 43.50 | 6.00 | 100 | 23.50 | 8.16 |
| focused_s42 | 24000 | 40.59 | 31.00 | 4.00 | 100 | 13.00 | 6.89 |
| focused_s848484 | 24000 | 10.46 | 8.00 | 1.00 | 83 | 0.00 | 5.43 |
| control_s424242 | 10000 | 3.69 | 3.00 | 1.00 | 18 | 0.00 | 0.86 |
| epsilon_s42 | 10000 | 4.65 | 3.00 | 1.00 | 23 | 0.00 | 0.90 |
| epsilon_s424242 | 10000 | 3.55 | 3.00 | 1.00 | 17 | 0.00 | 0.81 |
| gamma_s42 | 10000 | 4.99 | 3.00 | 1.00 | 29 | 0.00 | 0.83 |
| control_s42 | 10000 | 2.38 | 2.00 | 0.00 | 15 | 0.00 | 0.95 |
| gamma_s424242 | 10000 | 3.32 | 2.00 | 0.00 | 18 | 0.00 | 0.90 |
| intermediate_s42 | 10000 | 1.25 | 1.00 | 0.00 | 5 | 0.00 | 0.74 |
| intermediate_s424242 | 10000 | 1.60 | 1.00 | 0.00 | 10 | 0.00 | 0.77 |
| alpha_s42 | 6000 | 0.14 | 0.00 | 0.00 | 2 | 0.00 | 0.34 |
| alpha_s424242 | 6000 | 0.24 | 0.00 | 0.00 | 2 | 0.00 | 0.30 |
| combined_s42 | 6000 | 0.29 | 0.00 | 0.00 | 2 | 0.00 | 0.28 |
| combined_s424242 | 6000 | 0.18 | 0.00 | 0.00 | 3 | 0.00 | 0.31 |
| fine_s42 | 6000 | 0.20 | 0.00 | 0.00 | 2 | 0.00 | 0.30 |
| fine_s424242 | 6000 | 0.31 | 0.00 | 0.00 | 3 | 0.00 | 0.30 |


## Artifacts and launching

- `all_agents_test_comparison.csv`: all per-policy test metrics and cost.
- `locked_test_comparison.csv`: benchmark, validation and test metrics together.
- `screening_all.csv`: pilot and screening endpoint metrics.
- `evaluations/`: per-seed scores, rewards, termination causes and diagnostics.
- `<arm>/seed_<seed>/`: exact configuration, training histories, checkpoints, final models and console logs.
- `graphs/`: screening comparisons, learning curves, locked-test distributions, paired improvements, training cost and recommended-family learning curves.
- `final_summary.json`, `model_hashes.json`: machine-readable conclusions and checkpoint hashes.
- `recommended_agent.npz`: byte-for-byte copy of the validation-selected model, not a replacement for the original.
- `recommended_compatible_agent.npz` and `recommended_compatible_config.json`: metadata-compatible exports for the original loader, when experimental encoder ranges are unused.
- `demo_seed_check.json`: additional post-selection seed-2026 game check (recommended score 100 versus original 17); this single course did not drive selection.

A metadata-only compatible export was also verified with the unmodified original main-branch loader; it omits only the unused state_ranges=None key and preserves every Q value.

From PowerShell, launch the experimental model explicitly:

```powershell
Set-Location "C:\Users\User\Documents\FlappyBird-RL"
& "C:\Users\User\Documents\FlappyBird-RL\.venv\Scripts\python.exe" -m flappy_ai play --model "C:\Users\User\Documents\FlappyBird-RL-Experiments\experiments\study_20261117_122405\recommended_compatible_agent.npz"
```

The original launch command from the original folder continues to load the original agent. No promotion or GitHub publication is automatic.

## Remaining limitations

This is an adaptive, finite-budget study with only three finalist training seeds and one fixed game difficulty. Validation was used repeatedly, so its apparent improvements can be optimistic; the independently reserved test is the final confirmation. Early-stopped settings may improve with longer training. Many additional bin/range/learning schedules remain unexplored. Useful follow-ups within tabular Q-learning are an ablation separating range focusing from y/velocity coarsening, and separately controlled late-stage learning-rate/exploration decay schedules; neither is claimed tested here. Success is capped at 100 pipes, so performance beyond that target is unknown. The table is still an approximation, and state clipping may alias distant positions. A high median does not guarantee a high worst-case score; inspect P10 and failure rows. Checkpoint cleanup cannot guarantee saving during power/storage failure; previously completed checkpoints and fsynced statistics are preserved.

The course-bootstrap intervals condition on the selected trained model and do not fully quantify training-seed variability or repeated hyperparameter search. Raw episode rewards are not comparable between reward configurations; reported performance comparisons use pipe score and the common success target.

Exact recommended configuration:

```json
{
  "game": {
    "width": 480,
    "height": 640,
    "ground": 590,
    "bird_x": 115,
    "bird_radius": 13,
    "gravity": 0.42,
    "flap_velocity": -6.5,
    "max_velocity": 10,
    "pipe_speed": 3,
    "pipe_width": 70,
    "pipe_gap": 175,
    "pipe_spacing": 240,
    "max_steps": 18000,
    "target_score": 100
  },
  "rewards": {
    "survive": 0.01,
    "pipe": 10,
    "death": -15
  },
  "training": {
    "alpha": 0.18,
    "gamma": 0.98,
    "epsilon_start": 1,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.9995,
    "bins": [
      8,
      12,
      12,
      24,
      24
    ],
    "state_ranges": null,
    "episodes": 12000,
    "checkpoint_every": 1000,
    "seed": 424242
  }
}
```
