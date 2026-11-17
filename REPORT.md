# Concise technical report draft

**Flappy Bird AI â€” Reinforcement Learning**

## Objective and design

The project investigates whether tabular Q-learning can learn to navigate Flappy Bird using a compact numerical description of the world. The application provides human play, autonomous play and a seeded human-versus-AI comparison. The game uses original programmatically drawn graphics, a fixed 60 Hz simulation and independent rendering. This separation permits accelerated training without changing the task's dynamics.

The bird has gravity, a flap impulse and capped downward velocity. Obstacles have a fixed gap and spacing with seeded random opening heights. Collision uses a circular bird against rectangular obstacles, ceiling and floor. A score increments only when the bird safely clears a pipe. The course generator is independent of policy actions, so a fixed seed produces comparable obstacles.

## Learning method

The action set is {DO_NOTHING, FLAP}. The five observations are bird height, vertical velocity, horizontal distance to the next pipe, and signed distances to the opening's upper and lower boundaries. Each observation is normalized using fixed physical ranges and discretized into configurable bins. The resulting five-integer tuple indexes a sparse Q-table containing two action values. This is tabular reinforcement learning, with no neural network or evolutionary optimizer.

Q-learning applies Q(s,a) = Q(s,a) + alpha [r + gamma max Q(s',a') - Q(s,a)]. For true terminal states the future-value term is zero; a time-limit truncation retains it. The default learning rate is 0.18 and discount factor 0.98. Epsilon-greedy exploration begins at 1, decays by 0.9995 per episode and has a floor of 0.01. Default rewards are +0.05 per surviving step, +10 per pipe and -10 on death. A pipe is rewarded once, and finite step and score limits ensure termination.

## Experiment and validation

Training records episode score, return, epsilon, step count, rolling score and wall time. Compressed checkpoints preserve the Q-table, configuration, episode count and exploration generator state. Evaluation uses a separate fixed seed set and disables exploration without mutating the table. Metrics include mean score, best score and success rate, defined as reaching the configured target. CSV files and graphs provide auditable evidence. After 24,000 training episodes (7.18 minutes), the greedy policy averaged 20.45 pipes on 20 benchmark seeds (best 68). On 20 fresh seeds it averaged 18.4 (best 72). Neither set reached the 100-pipe target, so success rate was 0%. The untrained greedy baseline averaged zero. Training seeds were separate from evaluation seeds.

Automated tests cover physics, collision geometry, one-time scoring, termination, Bellman arithmetic, discretization, checkpoint round trips and deterministic evaluation. A headless Pygame smoke test checks rendering separately from environment logic.

## Limitations and interpretation

Discretization loses information; distinct physical situations can share a state. Sparse visitation and the redundant position/distance observations can slow learning. A policy trained on one game configuration may fail after difficulty changes, so checkpoints retain their environment settings. Twenty evaluation seeds provide a reproducible sample, not a guarantee of generalization. Additional training and a broader unseen seed set would strengthen conclusions. Human comparisons should use the same seed, configuration and speed, and account for familiarity gained from repeated courses.
