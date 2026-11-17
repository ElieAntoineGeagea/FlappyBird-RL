"""Isolated tabular Q-learning study. Never writes the original project."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
from flappy_ai.agent import QAgent
from flappy_ai.config import Config
from flappy_ai.controls import TrainingControl
from flappy_ai.environment import Environment
from flappy_ai.storage import atomic_json
from flappy_ai.training import train, write_csv

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'FlappyBird-RL'
ROOT = HERE / 'experiments' / 'study_20261117_122405'
TRAINING_SEEDS = [42, 424242, 848484]
BENCHMARK = list(range(100000,100020)) + list(range(200000,200020))
VALIDATION = list(range(300000,300100))
TEST = list(range(900000,900200))
CHANGES = {
 'focused': {'bins':[4,8,12,24,24], 'state_ranges':[[0,590],[-6.5,10],[-83,280],[-80,255],[-80,255]]},
 'control': {},
 'intermediate': {'bins':[8,12,16,36,36]},
 'fine': {'bins':[8,16,20,60,60]},
 'gamma': {'gamma':.995},
 'alpha': {'alpha':.08},
 'epsilon': {'epsilon_start':.30,'epsilon_decay':.9997,'epsilon_min':.005},
 'rewards': {'survive':.01,'pipe':10,'death':-15},
 'combined': {'bins':[8,16,20,60,60],'gamma':.995,'alpha':.08,
              'epsilon_start':.30,'epsilon_decay':.9997,'epsilon_min':.005,
              'survive':.01,'pipe':10,'death':-15},
}


def load_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    path = ROOT/'study.json'
    if not path.exists():
        started=time.time()
        plan=dict(started_epoch=started,hard_deadline_epoch=started+4*3600,
                  training_deadline_epoch=started+4*3600-600,episode_ceiling=264000,
                  training_seeds=TRAINING_SEEDS,benchmark=BENCHMARK,validation=VALIDATION,
                  locked_test=TEST,changes=CHANGES,algorithm='Unmodified tabular QAgent',
                  original_main_commit='bbda37d8219373350cf0173fec9291fb45233ba7',
                  pilot_episodes=3000,screening_target=10000,finalist_target=24000)
        atomic_json(path,plan)
        manifest={p.relative_to(ORIGINAL).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (ORIGINAL/'artifacts').rglob('*') if p.is_file()}
        atomic_json(ROOT/'original_hashes.json',manifest)
    return load_json(path)


def check_preserved():
    for path,digest in load_json(ROOT/'original_hashes.json').items():
        if hashlib.sha256((ORIGINAL/path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Original artifact changed: {path}')


def config_for(arm,seed):
    c=Config()
    # Preserve this historical study's 100-pipe task despite new game defaults.
    c.game.target_score=100; c.game.max_steps=18000
    c.training.seed=seed
    for key,value in CHANGES[arm].items():
        section=c.rewards if key in ('survive','pipe','death') else c.training
        setattr(section,key,value)
    return c.validate()


class BudgetControl(TrainingControl):
    def __init__(self,plan):
        super().__init__(ROOT)
        self.deadline=plan['training_deadline_epoch']
    def poll(self):
        result=super().poll()
        if time.time()>=self.deadline:
            self.stop=True
        return result


def model_index():
    return load_json(ROOT/'models.json',{})


def train_job(arm,seed,target):
    plan=prepare(); index=model_index(); key=f'{arm}_s{seed}'
    previous=index.get(key)
    if previous is None:
        recovered=sorted((ROOT/arm/f'seed_{seed}').glob('run_*/agent.npz'))
        if recovered:
            checkpoint=recovered[-1]
            recovered_agent=QAgent.load(checkpoint)
            previous=dict(arm=arm,training_seed=seed,episodes=recovered_agent.episodes,model=str(checkpoint),duration=recovered_agent.duration,states=len(recovered_agent.q),epsilon=recovered_agent.epsilon,runs=[str(checkpoint.parent)],last100_mean=None)
            index[key]=previous
            atomic_json(ROOT/'models.json',index)
    existing=previous['episodes'] if previous else 0
    if existing>=target:
        print(f'SKIP {key}: already {existing}',flush=True); return previous
    completed=sum(entry['episodes'] for entry in index.values())
    allowed=min(target-existing,plan['episode_ceiling']-completed)
    if allowed<=0 or time.time()>=plan['training_deadline_epoch'] or (ROOT/'STOP').exists():
        print('Budget or STOP reached; no further training.',flush=True); return None
    agent=QAgent.load(previous['model']) if previous else QAgent(config_for(arm,seed))
    directory=ROOT/arm/f'seed_{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    agent.config.save(directory/'config.json')
    print(f'TRAIN {key} {existing} -> {existing+allowed}',flush=True)
    with (directory/'console.log').open('a',encoding='utf-8') as log:
        with contextlib.redirect_stdout(log):
            rows=train(agent,allowed,directory,control=BudgetControl(plan),checkpoint_seconds=30)
    runs=sorted(directory.glob('run_*'))
    latest=runs[-1]
    entry=dict(arm=arm,training_seed=seed,episodes=agent.episodes,model=str(latest/'agent.npz'),
               duration=agent.duration,states=len(agent.q),epsilon=agent.epsilon,
               runs=(previous['runs'] if previous else [])+[str(latest)],
               last100_mean=float(np.mean([r['score'] for r in rows[-100:]])) if rows else None)
    index[key]=entry; atomic_json(ROOT/'models.json',index)
    print('DONE',json.dumps(entry),flush=True)
    return entry


def evaluate_agent(path, seeds, destination):
    destination=Path(destination)
    if (destination/'summary.json').exists():
        return load_json(destination/'summary.json')
    destination.mkdir(parents=True,exist_ok=True)
    agent=QAgent.load(path); env=Environment(agent.config)
    q_hash=hashlib.sha256(b''.join(v.tobytes() for v in agent.q.values())).hexdigest()
    rows=[]; deadline=prepare()['hard_deadline_epoch']-120
    for seed in seeds:
        if time.time()>=deadline:
            raise TimeoutError('Evaluation stopped for report reserve')
        obs=env.reset(seed); total=0; unknown=0; samples=0
        while not env.done:
            state=agent.state(obs); was_unknown=state not in agent.q
            unknown+=int(was_unknown); samples+=1
            action=agent.act(obs,explore=False)
            old=obs
            obs,reward,_,info=env.step(action); total+=reward
        if env.reason!='collision': cause=env.reason
        elif env.y-env.g.bird_radius<=0: cause='ceiling'
        elif env.y+env.g.bird_radius>=env.g.ground: cause='floor'
        else:
            pipe=env.next_pipe()
            cause='upper_pipe' if env.y<pipe.center else 'lower_pipe'
        rows.append(dict(seed=seed,score=env.score,reward=total,steps=env.steps,
                         success=int(env.reason=='target'),reason=env.reason,cause=cause,
                         unknown_fraction=unknown/samples,last_state_unknown=int(was_unknown),
                         death_velocity=env.velocity,last_action=action,
                         last_y=old[0],last_v=old[1],last_dx=old[2],last_upper=old[3],last_lower=old[4]))
    assert q_hash==hashlib.sha256(b''.join(v.tobytes() for v in agent.q.values())).hexdigest()
    scores=np.array([r['score'] for r in rows])
    tail_n=max(1,math.ceil(len(rows)*.1))
    summary=dict(episodes=len(rows),mean=float(np.mean(scores)),median=float(np.median(scores)),
                 p10=float(np.quantile(scores,.1)),lower_tail_mean=float(np.mean(np.sort(scores)[:tail_n])),
                 best=int(np.max(scores)),minimum=int(np.min(scores)),success=float(np.mean(scores>=env.g.target_score)),
                 mean_unknown_fraction=float(np.mean([r['unknown_fraction'] for r in rows])),
                 training_seconds=agent.duration,trained_episodes=agent.episodes,states=len(agent.q),
                 causes={cause:sum(r['cause']==cause for r in rows) for cause in set(r['cause'] for r in rows)},
                 model=str(path),seeds=list(seeds))
    write_csv(destination/'episodes.csv',rows); atomic_json(destination/'summary.json',summary)
    print('EVAL',destination.parent.name,destination.name,json.dumps({k:summary[k] for k in ['mean','median','p10','best','success']}),flush=True)
    return summary


def evaluate_entry(entry,stage, test=False):
    key=f"{entry['arm']}_s{entry['training_seed']}"
    base=ROOT/'evaluations'/stage/key
    results={}
    for split,seeds in [('benchmark',BENCHMARK),('validation',VALIDATION)]+([('test',TEST)] if test else []):
        results[split]=evaluate_agent(entry['model'],seeds,base/split)
    return results


def stage_summary(stage):
    rows=[]
    for directory in sorted((ROOT/'evaluations'/stage).glob('*')):
        for split in ('benchmark','validation','test'):
            path=directory/split/'summary.json'
            if path.exists():
                s=load_json(path)
                rows.append(dict(agent=directory.name,split=split,**{k:s[k] for k in ['trained_episodes','mean','median','p10','lower_tail_mean','best','success','states','training_seconds','mean_unknown_fraction']}))
    write_csv(ROOT/f'{stage}_comparison.csv',rows)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['pilot','extend','evaluate','status'])
    parser.add_argument('--arms',nargs='+',choices=list(CHANGES),default=list(CHANGES))
    parser.add_argument('--seeds',nargs='+',type=int,default=TRAINING_SEEDS[:2])
    parser.add_argument('--target',type=int,default=10000)
    parser.add_argument('--stage',default='screening')
    parser.add_argument('--test',action='store_true')
    args=parser.parse_args(); prepare(); check_preserved()
    if args.command=='pilot':
        evaluate_agent(ORIGINAL/'artifacts/agent.npz',BENCHMARK,ROOT/'evaluations'/'pilot'/'baseline'/'benchmark')
        evaluate_agent(ORIGINAL/'artifacts/agent.npz',VALIDATION,ROOT/'evaluations'/'pilot'/'baseline'/'validation')
        for seed in args.seeds:
            for arm in args.arms:
                entry=train_job(arm,seed,3000)
                if entry: evaluate_entry(entry,'pilot')
        stage_summary('pilot')
    elif args.command=='extend':
        for seed in args.seeds:
            for arm in args.arms:
                entry=train_job(arm,seed,args.target)
                if entry: evaluate_entry(entry,args.stage,args.test)
        stage_summary(args.stage)
    elif args.command=='evaluate':
        for entry in model_index().values():
            if entry['arm'] in args.arms and entry['training_seed'] in args.seeds:
                evaluate_entry(entry,args.stage,args.test)
        stage_summary(args.stage)
    else:
        print(json.dumps(dict(plan=prepare(),models=model_index()),indent=2))
    check_preserved()

if __name__=='__main__': main()
