"""Evaluate frozen policies on long games; performs no training or policy edits."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import time
import numpy as np
from flappy_ai.agent import QAgent
from flappy_ai.runtime import ORIGINAL_MODEL,IMPROVED_MODEL,PROJECT_ROOT,episode_config
from flappy_ai.environment import Environment
from flappy_ai.training import evaluate,write_csv
from flappy_ai.storage import unique_directory,atomic_json


def main():
    output=unique_directory('validations','long_game')
    cohorts={'fixed_benchmark':list(range(100000,100020))+list(range(200000,200020)),
             'previous_test':list(range(900000,900200)),
             'fresh_unseen':list(range(1200000,1200200))}
    seeds=[seed for group in cohorts.values() for seed in group]
    all_summary=[];started=time.perf_counter()
    model_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ORIGINAL_MODEL,IMPROVED_MODEL]}
    for name,path in [('original',ORIGINAL_MODEL),('improved',IMPROVED_MODEL)]:
        agent=QAgent.load(path);q_before={s:v.copy() for s,v in agent.q.items()}
        for target in [1000,100]:
            before=time.perf_counter()
            folder=output/name/f'target_{target}'
            summary=evaluate(agent,seeds,folder,target_score=target)
            elapsed=time.perf_counter()-before
            with (folder/'evaluation.csv').open() as f:rows=list(csv.DictReader(f))
            for cohort,group in dict(all=seeds,**cohorts).items():
                included=[r for r in rows if int(r['seed']) in set(group)]
                values=[int(r['score']) for r in included]
                item=dict(agent=name,target=target,cohort=cohort,episodes=len(values),
                          average=statistics.mean(values),median=statistics.median(values),
                          p10=float(np.quantile(values,.1)),best=max(values),minimum=min(values),
                          successes=sum(int(r['success']) for r in included),
                          success_rate=sum(int(r['success']) for r in included)/len(values),
                          collisions=sum(r['reason']=='collision' for r in included),
                          time_limits=sum(r['reason']=='time_limit' for r in included),
                          evaluation_seconds=elapsed)
                all_summary.append(item)
                print(json.dumps(item),flush=True)
            assert agent.config.game.target_score==100
            assert q_before.keys()==agent.q.keys() and all(np.array_equal(v,agent.q[s]) for s,v in q_before.items())
        with (output/name/'target_1000/evaluation.csv').open() as f:long=list(csv.DictReader(f))
        with (output/name/'target_100/evaluation.csv').open() as f:short=list(csv.DictReader(f))
        assert all(min(int(a['score']),100)==int(b['score']) for a,b in zip(long,short))
        env=Environment(episode_config(agent.config,1000));obs=env.reset(2026)
        while not env.done:obs,_,_,_=env.step(agent.act(obs,False))
        atomic_json(output/name/'demo_seed_2026.json',dict(seed=2026,score=env.score,reason=env.reason,steps=env.steps,target=1000))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in model_hashes.items())
    write_csv(output/'comparison.csv',all_summary)
    atomic_json(output/'summary.json',dict(policies={Path(p).relative_to(PROJECT_ROOT).as_posix(): digest for p, digest in model_hashes.items()},cohorts=cohorts,results=all_summary,
        actual_evaluation_seconds=time.perf_counter()-started,training_performed=False,exploration=False,
        short_long_prefixes_match=True))
    manifest=json.loads(Path('.long_game_preservation.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    print('SAVED',output.resolve(),flush=True)

if __name__=='__main__':main()
