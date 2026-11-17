"""Validation-only finalist allocation. The locked test is not read here."""
import json,time
from experiment_runner import ROOT,train_job,evaluate_entry,stage_summary,load_json,check_preserved
from research_analysis import family_ranking,stage_data
from flappy_ai.storage import atomic_json

for seed in [42,424242]:
    entry=train_job('focused',seed,10000)
    if entry: evaluate_entry(entry,'screening10000')
stage_summary('screening10000')
ranking=family_ranking(stage_data('screening10000'))
assert all(row['replicas']==2 for row in ranking)
finalists=[row['arm'] for row in ranking[:2]]
plan=dict(epoch=time.time(),finalists=finalists,validation_ranking=ranking,
          criteria=['worst replica median','worst replica p10','worst replica lower-tail mean','mean replica median','mean score'],
          training_seeds=[42,424242,848484],target_episodes=24000,
          test_not_accessed=True,episode_ceiling=264000)
atomic_json(ROOT/'finalists_plan.json',plan)
print('FINALISTS',json.dumps(plan),flush=True)
decisions=load_json(ROOT/'decisions.json')
decisions.append(dict(epoch=time.time(),stage='focused_pilot',decision='Continue focused to 10000 for equal-budget screening',reason='Focused representation achieves one median of 1 and uses ~3100 states vs ~5000 control at 6000. It is mixed across two seeds, so it is not claimed superior; an equal-budget check is justified by lower memory and comparable early performance.'))
decisions.append(dict(epoch=time.time(),stage='screening10000',decision='Advance two top robust validation configurations to 24000 episodes and a third training seed',finalists=finalists,ranking=ranking,test_not_accessed=True))
atomic_json(ROOT/'decisions.json',decisions)
for seed in [42,424242,848484]:
    for arm in finalists:
        entry=train_job(arm,seed,24000)
        if entry: evaluate_entry(entry,'final_validation')
stage_summary('final_validation')
check_preserved()
print('FINAL_VALIDATION',json.dumps(family_ranking(stage_data('final_validation'))),flush=True)
