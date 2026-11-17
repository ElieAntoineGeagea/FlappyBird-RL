# Approved isolated study protocol

No changes are made to the main branch or original project artifacts. All experiment code, configuration, models, histories and analyses reside in this separate worktree.

The study uses the existing tabular QAgent and the same Environment. No neural networks, evolution or scripted controller are used. Training invokes the original Bellman update; the only infrastructure adjustments are bounded retries for transient Windows replace locks and status JSON refresh every 100 episodes (episode CSV flush/fsync remains per episode).

Eight arms: control, intermediate bins [8,12,16,36,36], fine bins [8,16,20,60,60], gamma .995, alpha .08, epsilon .30/.9997/.005, rewards .01/+10/-15, and a combined fine-grid candidate using the preceding parameter changes. All unspecified parameters equal the original saved model. The environment and task limits are identical for all policies.

Pilot: 3,000 episodes per arm and training seed (42, 424242), 48,000 episodes total. Early failures alone do not eliminate arms whose epsilon remains high. Screening may extend surviving arms to 10,000 episodes each. Two finalists may then extend both training replicas to 24,000 and add training seed 848484 for 24,000 episodes. Eight-arm screening plus two finalists totals at most 264,000 training episodes. Discarded arms retain all checkpoints and results.

Validation selection uses 100 obstacle seeds 300000-300099. The historical 40 seeds (100000-100019 and 200000-200019) are benchmark-only. Two training replicas are compared on identical obstacle courses within each stage. Finalists are ranked first by robustness of median and lower-tail scores across training replicas, then mean and success rate. Isolated maxima are not used for selection. Selection is frozen before examining final test seeds 900000-900199. Every retained screening/final agent and the original baseline will receive final-test evaluation without subsequent tuning on those seeds.

Training obstacle seed ranges are training_seed + episode index, at most 24,000 per training seed. They do not overlap benchmark, validation or test sets. Exploration RNGs also use the respective independent training seeds. Physics, reward accounting and success definition (100 pipes) remain fixed except in the explicit reward arms. Cross-reward comparisons use pipe score; raw training returns are not directly comparable between reward configurations.

The hard wall-time limit is four hours from study.json creation. Training stops ten minutes before this deadline, final evaluations stop two minutes before it, leaving time to save metadata and summarize incomplete work. Checkpoints save at least every 30 seconds and on graceful stop. A STOP file at the study root requests a clean stop. Interrupted pilot continuation is recorded; no repeated or failed run is silently discarded from cost reporting.

Pilot decisions and any additional candidate configuration will be documented with timestamps and reasons before training it. Unused episode budget may be reassigned only within the approved 264,000-episode ceiling and four-hour cap. No agent is promoted or pushed to GitHub automatically.
