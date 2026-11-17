"""Long-game mechanics fixtures are not claims of AI reaching 1000."""
import gc
import hashlib
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
import pytest
from flappy_ai.config import Config
from flappy_ai.agent import QAgent
from flappy_ai.environment import Environment, Pipe
from flappy_ai.runtime import episode_config, ORIGINAL_MODEL, IMPROVED_MODEL
from flappy_ai.ui import Milestones
from flappy_ai.training import evaluate

ROOT=Path(__file__).resolve().parents[1]


def pass_next_pipe_fixture(score, target=1000):
    c=Config(); c.game.target_score=target
    env=Environment(c); env.score=score
    env.pipes[0]=Pipe(env.g.bird_x-env.g.pipe_width-env.g.bird_radius+1,env.y)
    return env


def test_defaults_and_long_runtime_budget():
    assert Config().game.target_score==1000
    original=QAgent.load(ORIGINAL_MODEL)
    runtime=episode_config(original.config,1000)
    assert original.config.game.target_score==100
    assert runtime.game.target_score==1000 and runtime.game.max_steps>80070
    assert runtime.game.max_steps>original.config.game.max_steps
    assert episode_config(original.config,100).game.max_steps==18000
    assert episode_config(original.config,1000,5).game.max_steps==5


@pytest.mark.parametrize('score',[99,100,249,499,749,998])
def test_score_does_not_end_before_target(score):
    env=pass_next_pipe_fixture(score)
    _,reward,done,info=env.step(0)
    assert env.score==score+1 and not done and info['reason']==''
    assert reward==env.config.rewards.pipe+env.config.rewards.survive


def test_full_course_scores_each_of_1000_pipes_once():
    class CenteredCourse(Environment):
        def new_pipe(self, x):
            return Pipe(x, self.g.ground / 2)

    env = CenteredCourse(Config())
    scores = []
    while not env.done:
        previous_score = env.score
        _, _, _, info = env.step(int(env.y >= env.g.ground / 2 and env.velocity >= 0))
        if env.score != previous_score:
            scores.append(env.score)

    assert scores == list(range(1, 1001))
    assert info == {'score': 1000, 'reason': 'target'}
    assert env.steps < env.g.max_steps


def test_1000_target_boundary_and_terminal_guard():
    env=pass_next_pipe_fixture(999)
    _,_,done,info=env.step(0)
    assert done and info=={'score':1000,'reason':'target'}
    with pytest.raises(RuntimeError):env.step(0)


def test_100_target_is_still_available():
    env=pass_next_pipe_fixture(99,100)
    assert env.step(0)[3]['reason']=='target'


def test_collision_at_high_score_is_not_victory():
    env=pass_next_pipe_fixture(999);env.y=0
    _,_,done,info=env.step(0)
    assert done and info['reason']=='collision' and env.score==999


def test_explicit_time_limit_is_not_victory():
    c=Config();c.game.max_steps=1
    assert Environment(c).step(0)[3]['reason']=='time_limit'


def test_milestones_once_and_expiry_and_reset():
    m=Milestones()
    assert m.update(99,0) is None
    for value in [100,250,500,750,1000]:
        assert m.update(value,value*100)==value
        assert m.update(value,value*100+2300) is None
    assert Milestones().update(0,0) is None


def test_pipe_recycling_is_bounded_and_deterministic_for_100000_frames():
    a=Environment(Config());b=Environment(Config());a.reset(77);b.reset(77)
    initial=[p.center for p in a.pipes]
    for frame in range(100000):
        a.advance_pipes();a.recycle_pipes()
        b.advance_pipes();b.recycle_pipes()
        assert len(a.pipes)==len(b.pipes)==3
        if frame%1000==0:assert a.pipes==b.pipes
    assert initial!=[p.center for p in a.pipes]
    assert all(-a.g.pipe_width<=p.x<=a.g.width+2*a.g.pipe_spacing for p in a.pipes)


def test_frozen_policy_actually_continues_beyond_100():
    agent=QAgent.load(IMPROVED_MODEL)
    env=Environment(episode_config(agent.config,1000));obs=env.reset(2026)
    seen_100=False
    while not env.done:
        obs,_,_,_=env.step(agent.act(obs,False))
        if env.score==100:
            assert not env.done
            seen_100=True
    assert seen_100 and env.score>100
    assert env.reason in ('collision','target')
    assert len(env.pipes)==3 and agent.config.game.target_score==100


def test_evaluation_1000_metrics_and_q_immutability(tmp_path):
    agent=QAgent.load(IMPROVED_MODEL);before={key:v.copy() for key,v in agent.q.items()}
    result=evaluate(agent,[2026,1200000],tmp_path,1000)
    assert result['target_score']==1000 and result['max_steps']>80070
    assert result['collisions']+result['victories']+result['time_limits']==2
    assert result['success_rate']==result['victories']/2
    assert result['best_score']>100
    assert before.keys()==agent.q.keys() and all(np.array_equal(v,agent.q[k]) for k,v in before.items())


def test_all_old_results_and_models_preserved():
    manifest=json.loads((ROOT/'.long_game_preservation.json').read_text())
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest for path,digest in manifest.items())
    assert len(list((ROOT/'artifacts').glob('checkpoint_*.npz')))==24


def test_long_pipe_simulation_memory_is_bounded():
    env=Environment(Config());tracemalloc.start()
    try:
        for _ in range(5000):env.advance_pipes();env.recycle_pipes()
        gc.collect();before=tracemalloc.get_traced_memory()[0]
        for _ in range(100000):env.advance_pipes();env.recycle_pipes()
        gc.collect();after=tracemalloc.get_traced_memory()[0]
        assert len(env.pipes)==3 and after-before<100000
    finally:tracemalloc.stop()
