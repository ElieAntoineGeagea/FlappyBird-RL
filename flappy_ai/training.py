"""Training, durable statistics, and reproducible greedy evaluation."""
import csv
import json
import time
from pathlib import Path
from collections import deque
from .environment import Environment
from .controls import TrainingControl
from .storage import CSVJournal, atomic_json, unique_directory

FIELDS = ['episode', 'seed', 'score', 'reward', 'epsilon', 'rolling_average', 'steps', 'reason', 'training_seconds']


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, path, evaluation=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not rows:
        return
    fig, axes = plt.subplots(1 if evaluation else 3, 1, figsize=(10, 5 if evaluation else 10), squeeze=False)
    try:
        ax = axes[0, 0]
        if evaluation:
            ax.bar([str(r['seed']) for r in rows], [r['score'] for r in rows])
            ax.set_xlabel('Fixed evaluation seed')
            ax.set_ylabel('Pipes passed')
        else:
            x = [r['episode'] for r in rows]
            reward_window = deque(maxlen=100)
            reward_average = []
            for row in rows:
                reward_window.append(row['reward'])
                reward_average.append(sum(reward_window) / len(reward_window))
            ax.plot(x, [r['reward'] for r in rows], alpha=.25, label='Episode reward')
            ax.plot(x, reward_average, label='100-episode average')
            ax.set_ylabel('Total reward')
            ax.legend()
            axes[1, 0].plot(x, [r['score'] for r in rows], alpha=.25, label='Episode score')
            axes[1, 0].plot(x, [r['rolling_average'] for r in rows], label='100-episode average')
            axes[1, 0].set_ylabel('Pipes passed')
            axes[1, 0].legend()
            axes[2, 0].plot(x, [r['epsilon'] for r in rows])
            axes[2, 0].set_ylabel('Epsilon')
            axes[2, 0].set_xlabel('Episode')
        for axis in axes[:, 0]:
            axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def train(agent, episodes, output, render=False, control=None, checkpoint_seconds=60):
    """Always create a new run folder, even when resuming an existing agent."""
    agent.config.validate()
    if type(episodes) is not int or episodes < 1:
        raise ValueError('episodes must be a positive integer')
    if type(checkpoint_seconds) not in (int, float) or not 0 < checkpoint_seconds < float('inf'):
        raise ValueError('checkpoint_seconds must be positive and finite')
    run = unique_directory(output, 'run')
    print(f'Training outputs: {run.resolve()}', flush=True)
    print('P pause/resume, C checkpoint, S stop (Windows terminal); Ctrl+C saves and stops.\n'
          'Control files in this run folder: PAUSE, CHECKPOINT, STOP.', flush=True)
    control = control or TrainingControl(run)
    env = Environment(agent.config)
    recent = deque(maxlen=100)
    rows = []
    view = None
    journal = None
    started = time.perf_counter()
    base_duration = agent.duration
    last_checkpoint = started
    status = 'failed'
    pause_saved = False
    last_poll = started

    def save_checkpoint(named=False):
        nonlocal last_checkpoint
        agent.duration = base_duration + time.perf_counter() - started
        agent.save(run / 'latest.npz')
        if named:
            agent.save(run / f'checkpoint_{agent.episodes}.npz')
        last_checkpoint = time.perf_counter()

    def stop_requested():
        control.poll()
        if control.checkpoint:
            save_checkpoint()
            control.checkpoint = False
            print('Checkpoint saved.', flush=True)
        return control.stop

    def status_file(state):
        atomic_json(run / 'status.json', dict(status=state, episodes=agent.episodes,
                    epsilon=agent.epsilon, training_seconds=agent.duration,
                    completed_rows=len(rows), partial_episode_saved=not env.done))

    try:
        journal = CSVJournal(run / 'training.csv', FIELDS)
        agent.config.save(run / 'config.json')
        control.install()
        save_checkpoint()
        status_file('running')
        if render:
            from .ui import TrainingView
            view = TrainingView(agent.config)
        for _ in range(episodes):
            seed = agent.config.training.seed + agent.episodes
            obs = env.reset(seed)
            total = 0.0
            while not env.done:
                now = time.perf_counter()
                if env.steps % 128 == 0 or now - last_poll >= .1:
                    last_poll = now
                    paused = control.poll()
                    if control.stop:
                        break
                    if control.checkpoint:
                        save_checkpoint()
                        control.checkpoint = False
                        print('Checkpoint saved.', flush=True)
                    while paused and not control.stop:
                        if not pause_saved:
                            save_checkpoint()
                            status_file('paused')
                            print('Paused; checkpoint saved.', flush=True)
                            pause_saved = True
                        if view and not view.update(env, agent, total, recent, on_checkpoint=save_checkpoint, should_stop=stop_requested, external_paused=True):
                            control.stop = True
                            break
                        time.sleep(.05)
                        paused = control.poll()
                        if control.checkpoint:
                            save_checkpoint()
                            control.checkpoint = False
                    if pause_saved and not control.stop:
                        print('Resuming training.', flush=True)
                        status_file('running')
                        pause_saved = False
                if control.stop:
                    break
                if view and env.steps % max(1, view.speed) == 0:
                    if not view.update(env, agent, total, recent, on_checkpoint=save_checkpoint, should_stop=stop_requested):
                        control.stop = True
                        break
                action = agent.act(obs)
                nxt, reward, done, info = env.step(action)
                # Unchanged Bellman update and time-limit bootstrap behavior.
                agent.update(obs, action, reward, nxt, done and info['reason'] != 'time_limit')
                obs = nxt
                total += reward
                if time.perf_counter() - last_checkpoint >= checkpoint_seconds:
                    save_checkpoint()
            if control.stop:
                break
            agent.finish_episode()
            recent.append(env.score)
            agent.duration = base_duration + time.perf_counter() - started
            row = dict(episode=agent.episodes, seed=seed, score=env.score, reward=round(total, 4),
                       epsilon=agent.epsilon, rolling_average=sum(recent) / len(recent),
                       steps=env.steps, reason=env.reason, training_seconds=agent.duration)
            journal.append(row)
            rows.append(row)
            if agent.episodes % 100 == 0 or len(rows) == 1:
                print(f"Episode {agent.episodes} | epsilon {agent.epsilon:.4f} | score {env.score} | "
                      f"reward {total:.2f} | average {row['rolling_average']:.2f}", flush=True)
            if agent.episodes % agent.config.training.checkpoint_every == 0:
                save_checkpoint(named=True)
            status_file('running')
        status = 'stopped' if control.stop else 'completed'
    except KeyboardInterrupt:
        status = 'interrupted'
        print('Interrupted; saving progress.', flush=True)
    finally:
        # Save before plotting; a plotting failure must never lose the learned table.
        try:
            agent.duration = base_duration + time.perf_counter() - started
            agent.save(run / 'agent.npz')
            agent.save(run / 'latest.npz')
            status_file(status)
        finally:
            if journal is not None:
                journal.close()
            control.restore()
            if view is not None:
                view.close()
        if rows:
            try:
                plot_rows(rows, run / 'training.png')
            except Exception as error:
                print(f'Warning: graph export failed ({error}); CSV and checkpoints are saved.', flush=True)
        print(f'Training {status}. Saved agent and statistics: {run.resolve()}', flush=True)
    return rows


def evaluate(agent, seeds, output):
    seeds = list(seeds)
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError('At least one integer evaluation seed is required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('evaluation.csv', 'evaluation_summary.json', 'evaluation.png')):
        raise FileExistsError(f'Evaluation results already exist in {output}; choose a new output directory')
    env = Environment(agent.config)
    rows = []
    for seed in seeds:
        obs = env.reset(seed)
        total = 0
        while not env.done:
            obs, reward, _, _ = env.step(agent.act(obs, explore=False))
            total += reward
        rows.append(dict(seed=seed, score=env.score, reward=total, steps=env.steps,
                         success=int(env.reason == 'target'), reason=env.reason))
    summary = dict(episodes=len(rows), average_score=sum(r['score'] for r in rows) / len(rows),
                   best_score=max(r['score'] for r in rows),
                   success_rate=sum(r['success'] for r in rows) / len(rows),
                   success_definition=f'Reach {agent.config.game.target_score} pipes',
                   training_duration_seconds=agent.duration, seeds=seeds, exploration=False)
    write_csv(output / 'evaluation.csv', rows)
    atomic_json(output / 'evaluation_summary.json', summary)
    plot_rows(rows, output / 'evaluation.png', True)
    return summary
