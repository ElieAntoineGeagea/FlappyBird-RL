import json
from pathlib import Path
import pytest
from flappy_ai import storage
from experiment_runner import config_for, CHANGES, TRAINING_SEEDS, BENCHMARK, VALIDATION, TEST


def test_experiment_configs_are_valid_and_environment_identical():
    control=config_for('control',42)
    for arm in CHANGES:
        c=config_for(arm,42)
        c.validate()
        assert c.game==control.game


def test_seed_sets_are_disjoint():
    evaluated=set(BENCHMARK+VALIDATION+TEST)
    assert len(evaluated)==340
    for seed in TRAINING_SEEDS:
        assert not set(range(seed,seed+24000)) & evaluated


def test_transient_windows_file_lock_retry(tmp_path,monkeypatch):
    source=tmp_path/'source'; target=tmp_path/'target'; source.write_text('new'); target.write_text('old')
    original=storage.os.replace; calls=[]
    def replacement(a,b):
        calls.append(True)
        if len(calls)==1: raise PermissionError('transient lock')
        original(a,b)
    monkeypatch.setattr(storage.os,'replace',replacement)
    storage.replace_with_retry(source,target)
    assert target.read_text()=='new' and len(calls)==2


def test_focused_state_representation_is_bounded_and_serializable(tmp_path):
    import numpy as np
    from flappy_ai.agent import QAgent
    agent=QAgent(config_for('focused',42))
    assert np.prod(agent.config.training.bins) < np.prod(config_for('control',42).training.bins)
    for obs in [(0,-6.5,-83,-80,-80),(590,10,280,255,255),(-100,-20,-1000,-1000,1000)]:
        state=agent.state(obs)
        assert all(0<=index<bins for index,bins in zip(state,agent.config.training.bins))
    p=tmp_path/'focused.npz'; agent.save(p)
    restored=QAgent.load(p)
    assert restored.config==agent.config
    assert restored.state((100,2,40,20,155))==agent.state((100,2,40,20,155))


@pytest.mark.parametrize('ranges',[[[1,0]]*5,[[0,1]]*4,[[0,float('nan')]]*5])
def test_invalid_experimental_ranges_rejected(ranges):
    c=config_for('control',42); c.training.state_ranges=ranges
    with pytest.raises(ValueError): c.validate()


def test_robust_ranking_does_not_reward_an_isolated_maximum():
    from research_analysis import family_ranking
    def row(median,p10,mean):return dict(median=median,p10=p10,mean=mean,lower_tail_mean=p10,success=0)
    data={'unstable_s1':row(100,20,100),'unstable_s2':row(0,0,0),'reliable_s1':row(20,8,25),'reliable_s2':row(22,9,26)}
    assert family_ranking(data)[0]['arm']=='reliable'


def test_paired_bootstrap_constant_improvement():
    import numpy as np
    from research_analysis import paired_bootstrap
    baseline=np.arange(20); candidate=baseline+3
    result=paired_bootstrap(candidate,baseline,repetitions=200)
    assert result['mean_delta']==result['median_delta']==3
    assert np.allclose(result['mean_delta_ci'],[3,3])
    assert result['win_fraction']==1


def test_study_budget_control_stops_at_deadline(tmp_path):
    import time
    from experiment_runner import BudgetControl
    control=BudgetControl({'training_deadline_epoch':time.time()-1})
    control.directory=tmp_path
    control.poll()
    assert control.stop


def test_representative_median_near_tie_prefers_lower_tail():
    from research_analysis import choose_representative
    candidates={'slightly_higher':dict(median=54,p10=3,lower_tail_mean=1,mean=53),
                'safer':dict(median=52,p10=11.8,lower_tail_mean=5,mean=57),
                'too_low':dict(median=40,p10=20,lower_tail_mean=15,mean=45)}
    assert choose_representative(candidates)=='safer'
