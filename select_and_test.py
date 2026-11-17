"""Freeze validation selection first, then evaluate the untouched test courses."""
import hashlib
import json
import shutil
import time
import subprocess
import sys
import numpy as np
from pathlib import Path
from experiment_runner import ROOT,ORIGINAL,BENCHMARK,VALIDATION,TEST,model_index,evaluate_agent,evaluate_entry,stage_summary,load_json,check_preserved
from research_analysis import stage_data,family_ranking,choose_representative
from flappy_ai.agent import QAgent
from flappy_ai.storage import atomic_json


def main():
    index=model_index(); plan=load_json(ROOT/'finalists_plan.json')
    finalist_keys=[f'{arm}_s{seed}' for arm in plan['finalists'] for seed in plan['training_seeds']]
    assert all(index[key]['episodes']==24000 for key in finalist_keys),'Wait for all six complete training runs'
    selection_path=ROOT/'selection.json'
    if not selection_path.exists():
        assert not (ROOT/'evaluations/locked_test').exists(),'Test results must not exist before selection'
        data=stage_data('final_validation'); ranking=family_ranking(data)
        assert all(r['replicas']==3 for r in ranking)
        arm=ranking[0]['arm']
        candidates={key:s for key,s in data.items() if key.startswith(arm+'_s')}
        chosen=choose_representative(candidates)
        baseline=load_json(ROOT/'evaluations/pilot/baseline/validation/summary.json')
        best=candidates[chosen]
        if best['median']<baseline['median'] or best['p10']<baseline['p10'] or best['mean']<baseline['mean']:
            chosen='baseline'; arm='baseline'
        path=ORIGINAL/'artifacts/agent.npz' if chosen=='baseline' else Path(index[chosen]['model'])
        selection=dict(epoch=time.time(),recommended_arm=arm,recommended_key=chosen,
                       recommended_source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       final_validation_ranking=ranking,finalist_keys=finalist_keys,
                       selection_split='validation 300000-300099 only',test_not_accessed=True,
                       representative_rule='Within 10% of the best validation median in the most robust family, prefer p10, lower-tail mean, median and mean; never maximum score',
                       final_test_seeds=TEST)
        atomic_json(selection_path,selection)
    else: selection=load_json(selection_path)
    print('LOCKED_SELECTION',json.dumps(selection),flush=True)
    check_preserved()
    base=ROOT/'evaluations/locked_test/baseline'
    evaluate_agent(ORIGINAL/'artifacts/agent.npz',BENCHMARK,base/'benchmark')
    evaluate_agent(ORIGINAL/'artifacts/agent.npz',VALIDATION,base/'validation')
    evaluate_agent(ORIGINAL/'artifacts/agent.npz',TEST,base/'test')
    for key,entry in index.items():
        evaluate_entry(entry,'locked_test',test=True)
    stage_summary('locked_test')
    source=Path(selection['recommended_source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==selection['source_sha256']
    target=ROOT/'recommended_agent.npz'
    if not target.exists(): shutil.copy2(source,target)
    assert target.read_bytes()==source.read_bytes()
    selected_agent=QAgent.load(target)
    selected_agent.config.save(ROOT/'recommended_config.json')
    # Old main-branch loaders do not know the new optional None-valued range field.
    # Export an identical table with only that unused metadata key omitted.
    if selected_agent.config.training.state_ranges is None:
        compatible=ROOT/'recommended_compatible_agent.npz'
        with np.load(target,allow_pickle=False) as archive:
            states=archive['states'].copy(); values=archive['values'].copy()
            metadata=json.loads(str(archive['metadata']))
        metadata['config']['training'].pop('state_ranges',None)
        atomic_json(ROOT/'recommended_compatible_config.json',metadata['config'])
        if not compatible.exists():
            with compatible.open('xb') as output:
                np.savez_compressed(output,states=states,values=values,metadata=json.dumps(metadata))
        verify_code='import sys,json,hashlib; from flappy_ai.agent import QAgent; a=QAgent.load(sys.argv[1]); print(json.dumps(dict(episodes=a.episodes,states=len(a.q),q_sha256=hashlib.sha256(b"".join(v.tobytes() for v in a.q.values())).hexdigest())))'
        checked=json.loads(subprocess.check_output([sys.executable,'-c',verify_code,str(compatible)],cwd=ORIGINAL,text=True))
        expected=hashlib.sha256(b''.join(v.tobytes() for v in selected_agent.q.values())).hexdigest()
        assert checked['q_sha256']==expected and checked['episodes']==selected_agent.episodes
        atomic_json(ROOT/'compatibility_check.json',dict(original_main_loader_verified=True,**checked,model=str(compatible),metadata_change='Only omitted unused state_ranges=None; Q-table unchanged'))

    check_preserved()
    print('LOCKED TEST COMPLETE. Recommended source:',selection['recommended_source'],flush=True)

if __name__=='__main__':main()
