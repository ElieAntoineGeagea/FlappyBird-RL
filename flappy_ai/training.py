import csv
import json
import time
from pathlib import Path
from collections import deque
from .environment import Environment


def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot_rows(rows, path, evaluation=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2 if not evaluation else 1,1,figsize=(9,6),squeeze=False)
    ax=axes[0,0]
    if evaluation:
        ax.bar([str(r['seed']) for r in rows],[r['score'] for r in rows]); ax.set_xlabel('Fixed evaluation seed')
    else:
        x=[r['episode'] for r in rows]
        ax.plot(x,[r['score'] for r in rows],alpha=.3,label='Episode score')
        ax.plot(x,[r['rolling_average'] for r in rows],label='Last 100 episodes'); ax.legend()
        axes[1,0].plot(x,[r['epsilon'] for r in rows]); axes[1,0].set_ylabel('Epsilon'); axes[1,0].set_xlabel('Episode')
    ax.set_ylabel('Pipes passed'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def train(agent, episodes, output, render=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    env=Environment(agent.config); recent=deque(maxlen=100); rows=[]
    view=None
    if render:
        from .ui import TrainingView
        view=TrainingView(agent.config)
    started=time.perf_counter(); base_duration=agent.duration
    stop=False
    try:
        for _ in range(episodes):
            seed=agent.config.training.seed+agent.episodes
            obs=env.reset(seed); total=0.0
            while not env.done:
                if view and env.steps % max(1,view.speed) == 0:
                    if not view.update(env,agent,total,recent): stop=True; break
                action=agent.act(obs)
                nxt,reward,done,info=env.step(action)
                # A time limit is a truncation: bootstrap its last transition.
                agent.update(obs,action,reward,nxt,done and info['reason']!='time_limit')
                obs=nxt; total+=reward
            if stop: break
            agent.finish_episode(); recent.append(env.score)
            agent.duration=base_duration+time.perf_counter()-started
            row=dict(episode=agent.episodes,seed=seed,score=env.score,reward=round(total,4),epsilon=agent.epsilon,rolling_average=sum(recent)/len(recent),steps=env.steps,reason=env.reason,training_seconds=agent.duration)
            rows.append(row)
            if agent.episodes % 100 == 0 or len(rows)==1:
                print(f"Episode {agent.episodes} | epsilon {agent.epsilon:.4f} | score {env.score} | reward {total:.2f} | average {row['rolling_average']:.2f}",flush=True)
            if agent.episodes % agent.config.training.checkpoint_every==0:
                agent.save(output/f'checkpoint_{agent.episodes}.npz')
    except KeyboardInterrupt:
        print('Training interrupted; saving completed episodes.',flush=True)
    finally:
        agent.duration=base_duration+time.perf_counter()-started
        agent.save(output/'agent.npz')
        # Separate history per invocation preserves earlier runs when resuming.
        if rows:
            stem=f'training_{rows[0]["episode"]}_{rows[-1]["episode"]}'
            write_csv(output/f'{stem}.csv',rows); plot_rows(rows,output/f'{stem}.png')
        if view: view.close()
    return rows


def evaluate(agent, seeds, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    env=Environment(agent.config); rows=[]
    for seed in seeds:
        obs=env.reset(seed); total=0
        while not env.done:
            obs,reward,_,_=env.step(agent.act(obs,explore=False)); total+=reward
        rows.append(dict(seed=seed,score=env.score,reward=total,steps=env.steps,success=int(env.reason=='target'),reason=env.reason))
    if not rows: raise ValueError('At least one evaluation seed required')
    summary=dict(episodes=len(rows),average_score=sum(r['score'] for r in rows)/len(rows),best_score=max(r['score'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows),success_definition=f'Reach {agent.config.game.target_score} pipes',training_duration_seconds=agent.duration,seeds=list(seeds),exploration=False)
    write_csv(output/'evaluation.csv',rows)
    (output/'evaluation_summary.json').write_text(json.dumps(summary,indent=2))
    plot_rows(rows,output/'evaluation.png',True)
    return summary
