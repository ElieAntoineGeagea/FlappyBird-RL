import json
import numpy as np
import pytest
from flappy_ai.config import Config
from flappy_ai.environment import Environment, Pipe
from flappy_ai.agent import QAgent
from flappy_ai.training import evaluate, train


def test_physics_and_flap():
    e=Environment(Config()); y=e.y
    e.step(0); assert e.y>y and e.velocity>0
    e.step(1); assert e.velocity==pytest.approx(e.g.flap_velocity+e.g.gravity)

@pytest.mark.parametrize('y',[1,589])
def test_world_collisions(y):
    e=Environment(Config()); e.y=y; assert e.collides()


def test_pipe_collision_and_gap():
    e=Environment(Config()); e.pipes=[Pipe(e.g.bird_x,295)]
    e.y=295; assert not e.collides()
    e.y=200; assert e.collides()


def test_pass_reward_once_and_terminal_guard():
    e=Environment(Config()); e.pipes[0]=Pipe(e.g.bird_x-e.g.pipe_width-e.g.bird_radius+1,e.y)
    _,r,_,_=e.step(0); assert e.score==1 and r==e.config.rewards.pipe+e.config.rewards.survive
    e.step(0); assert e.score==1
    e.y=0; e.step(0)
    with pytest.raises(RuntimeError): e.step(0)


def test_seed_and_limit():
    c=Config(); c.game.max_steps=1
    a,b=Environment(c),Environment(c)
    assert a.reset(12)==b.reset(12)
    assert a.pipes==b.pipes
    assert a.step(0)[3]['reason']=='time_limit'


def test_bellman_and_terminal():
    c=Config(); a=QAgent(c); o=Environment(c).observe(); s=a.state(o)
    a.values(s)[:]=[2,4]
    a.update(o,0,3,o,False)
    assert a.values(s)[0]==pytest.approx(2+c.training.alpha*(3+c.training.gamma*4-2))
    a.update(o,1,-10,o,True)
    assert a.values(s)[1]==pytest.approx(4+c.training.alpha*(-10-4))


def test_bins_epsilon_and_greedy():
    a=QAgent(Config()); o=Environment(a.config).observe()
    a.values(a.state(o))[:]=[0,5]; a.epsilon=1
    assert all(a.act(o,False)==1 for _ in range(20))
    s=a.state((-1e9,1e9,-1e9,1e9,-1e9))
    assert all(0<=v<n for v,n in zip(s,a.config.training.bins))
    a.finish_episode(); assert a.epsilon<1


def test_serialization(tmp_path):
    a=QAgent(Config()); o=Environment(a.config).observe(); a.values(a.state(o))[1]=2.3
    a.finish_episode(); a.duration=9.5; p=tmp_path/'agent.npz'; a.save(p); b=QAgent.load(p)
    assert a.config==b.config and a.episodes==b.episodes and a.duration==b.duration
    assert a.act(o,False)==b.act(o,False)
    assert np.array_equal(a.q[a.state(o)],b.q[b.state(o)])
    assert a.rng.random()==b.rng.random()


def test_evaluation_is_deterministic_and_does_not_learn(tmp_path):
    a=QAgent(Config()); before=(len(a.q),a.epsilon,a.episodes)
    x=evaluate(a,[100,101],tmp_path/'a'); y=evaluate(a,[100,101],tmp_path/'b')
    assert x==y and before==(len(a.q),a.epsilon,a.episodes)
    assert (tmp_path/'a'/'evaluation.csv').exists()
    assert x['success_rate']==0


def test_short_training(tmp_path):
    a=QAgent(Config()); rows=train(a,3,tmp_path)
    assert len(rows)==3 and a.episodes==3 and len(a.q)>0
    assert QAgent.load(next(tmp_path.glob('run_*/agent.npz'))).episodes==3


def test_invalid_config():
    c=Config(); c.training.gamma=1
    with pytest.raises(ValueError): c.validate()


def test_target_termination():
    c=Config(); c.game.target_score=1
    e=Environment(c); e.pipes[0]=Pipe(e.g.bird_x-e.g.pipe_width-e.g.bird_radius+1,e.y)
    _,_,done,info=e.step(0)
    assert done and info['reason']=='target' and info['score']==1


def test_different_seeds_change_course():
    a,b=Environment(Config()),Environment(Config())
    a.reset(100); b.reset(101)
    assert [p.center for p in a.pipes] != [p.center for p in b.pipes]


def test_nonfinite_rewards_rejected():
    c=Config(); c.rewards.survive=float('nan')
    with pytest.raises(ValueError): c.validate()
