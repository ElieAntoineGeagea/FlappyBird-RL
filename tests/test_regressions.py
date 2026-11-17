"""Regression checks use temporary agents only; the trained policy is never updated."""
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from flappy_ai.agent import QAgent
from flappy_ai.config import Config
from flappy_ai.controls import TrainingControl
from flappy_ai.storage import CSVJournal, save_comparison
from flappy_ai import training

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('section,name,value', [
    ('game','pipe_width',-1), ('game','pipe_width',0), ('game','pipe_width',float('inf')),
    ('game','pipe_width',10), ('game','width',0), ('game','height',100),
    ('game','bird_radius',-1), ('game','bird_x',0), ('game','pipe_gap',20),
    ('game','pipe_spacing',80), ('game','pipe_speed',0), ('game','gravity',float('nan')),
    ('game','gravity',100), ('game','max_velocity',0), ('game','flap_velocity',1),
    ('game','max_steps',1.5), ('game','target_score',True), ('game','ground','590'),
    ('training','alpha',0), ('training','gamma',1), ('training','epsilon_decay',0),
    ('training','epsilon_start',float('nan')), ('training','epsilon_min',2),
    ('training','bins',[8,12,12,24,True]), ('training','bins',None),
    ('training','episodes',1.5), ('training','checkpoint_every',0),
    ('training','seed',-1), ('rewards','death','bad'), ('rewards','survive',float('inf')),
])
def test_reject_invalid_parameters(section,name,value):
    c=Config(); setattr(getattr(c,section),name,value)
    with pytest.raises(ValueError): c.validate()


def test_config_shape_and_unknown_parameters(tmp_path):
    p=tmp_path/'config.json'
    for content in ['[]','{"game":{"typo":1}}','{"game":false}','{"unknown":{}}','{broken']:
        p.write_text(content)
        with pytest.raises(ValueError): Config.load(p)


def tiny_agent():
    c=Config(); c.game.max_steps=1; c.training.checkpoint_every=1
    return QAgent(c)


def test_incremental_journal_and_run_isolation(tmp_path):
    original=tmp_path/'agent.npz'; original.write_bytes(b'preserved original')
    agent=tiny_agent(); update=agent.update
    observed=[]
    def inspect(*args):
        journal=next(tmp_path.glob('run_*/training.csv'))
        with journal.open() as f: observed.append(len(list(csv.DictReader(f))))
        update(*args)
    agent.update=inspect
    rows=training.train(agent,3,tmp_path)
    assert observed==[0,1,2]  # Rows were readable before the next episode, not only at exit.
    assert len(rows)==3 and original.read_bytes()==b'preserved original'
    training.train(tiny_agent(),1,tmp_path)
    runs=list(tmp_path.glob('run_*')); assert len(runs)==2
    assert sum((run/'agent.npz').exists() for run in runs)==2
    assert (runs[0]/'training.png').exists()


def test_checkpoint_pause_resume_and_cooperative_stop(tmp_path):
    class ScriptedControl(TrainingControl):
        calls=0
        def poll(self):
            self.calls+=1
            if self.calls==1: return True
            if self.calls==2: self.checkpoint=True; return True
            if self.calls==4: self.stop=True
            return False
    control=ScriptedControl(tmp_path)
    rows=training.train(tiny_agent(),5,tmp_path,control=control)
    assert len(rows)==1
    run=next(tmp_path.glob('run_*'))
    assert QAgent.load(run/'agent.npz').episodes==1
    assert (run/'latest.npz').exists()
    assert json.loads((run/'status.json').read_text())['status']=='stopped'


def test_control_files_and_signal_request(tmp_path):
    control=TrainingControl(tmp_path)
    (tmp_path/'PAUSE').touch(); (tmp_path/'CHECKPOINT').touch()
    assert control.poll() and control.checkpoint
    assert not (tmp_path/'CHECKPOINT').exists()
    (tmp_path/'PAUSE').unlink(); assert not control.poll()
    (tmp_path/'STOP').touch(); control.poll(); assert control.stop
    control.stop=False
    previous=signal.getsignal(signal.SIGINT)
    control.install()
    try:
        signal.getsignal(signal.SIGINT)(signal.SIGINT,None)
        assert control.stop
    finally: control.restore()
    assert signal.getsignal(signal.SIGINT)==previous


@pytest.mark.skipif(os.name!='nt',reason='Windows terminal controls')
def test_windows_terminal_keys(monkeypatch,tmp_path):
    import msvcrt
    keys=iter('pcs')
    available=iter([True,True,True,False])
    monkeypatch.setattr(msvcrt,'kbhit',lambda:next(available))
    monkeypatch.setattr(msvcrt,'getwch',lambda:next(keys))
    monkeypatch.setattr(sys,'stdin',type('Terminal',(),{'isatty':lambda self:True})())
    control=TrainingControl(tmp_path); assert control.poll()
    assert control.stop and control.checkpoint


def test_keyboard_interrupt_saves_and_restores_handlers(tmp_path,monkeypatch):
    agent=tiny_agent()
    monkeypatch.setattr(agent,'act',lambda *args,**kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    old=signal.getsignal(signal.SIGINT)
    assert training.train(agent,5,tmp_path)==[]
    run=next(tmp_path.glob('run_*'))
    assert QAgent.load(run/'agent.npz').episodes==0
    assert json.loads((run/'status.json').read_text())['status']=='interrupted'
    assert signal.getsignal(signal.SIGINT)==old


def test_failure_preserves_journal_and_agent(tmp_path,monkeypatch):
    agent=tiny_agent(); original=agent.act
    def fail(*args,**kwargs):
        if agent.episodes==1: raise RuntimeError('simulated unexpected failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(agent,'act',fail)
    with pytest.raises(RuntimeError,match='simulated'): training.train(agent,3,tmp_path)
    run=next(tmp_path.glob('run_*'))
    with (run/'training.csv').open() as f: assert len(list(csv.DictReader(f)))==1
    assert QAgent.load(run/'agent.npz').episodes==1
    assert json.loads((run/'status.json').read_text())['status']=='failed'


def test_plot_failure_does_not_lose_checkpoint(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(training,'plot_rows',lambda *args,**kwargs: (_ for _ in ()).throw(OSError('simulated plotting error')))
    training.train(tiny_agent(),1,tmp_path)
    run=next(tmp_path.glob('run_*'))
    assert QAgent.load(run/'agent.npz').episodes==1
    assert 'graph export failed' in capsys.readouterr().out


def test_reward_score_epsilon_graphs(tmp_path,monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    inspected=[]
    original=Figure.savefig
    def inspect(fig,*args,**kwargs):
        inspected.append([ax.get_ylabel() for ax in fig.axes])
        return original(fig,*args,**kwargs)
    monkeypatch.setattr(Figure,'savefig',inspect)
    row=dict(episode=1,reward=-5,score=0,epsilon=.5,rolling_average=0)
    training.plot_rows([row],tmp_path/'curve.png')
    assert inspected==[['Total reward','Pipes passed','Epsilon']]
    assert not plt.get_fignums()


def test_comparison_unique_and_preserved(tmp_path):
    old=tmp_path/'comparison.csv'; old.write_bytes(b'existing results')
    row=dict(seed=2026,human_score=1,ai_score=17,speed=1)
    first=save_comparison(row,tmp_path); second=save_comparison(row,tmp_path)
    assert first!=second and first.exists() and second.exists()
    assert old.read_bytes()==b'existing results'
    with first.open() as f: assert list(csv.DictReader(f))[0]['ai_score']=='17'


def test_atomic_save_failure_preserves_previous_file(tmp_path,monkeypatch):
    import flappy_ai.agent as module
    path=tmp_path/'agent.npz'; agent=tiny_agent(); agent.save(path); before=path.read_bytes()
    monkeypatch.setattr(module.os,'replace',lambda *args: (_ for _ in ()).throw(OSError('simulated replace failure')))
    with pytest.raises(OSError): agent.save(path)
    assert path.read_bytes()==before and not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('kind',['missing','badshape','nan','out_of_range','duplicate','badmeta'])
def test_corrupt_checkpoint_rejected(kind,tmp_path):
    p=tmp_path/'bad.npz'; tiny_agent().save(p)
    with np.load(p) as archive: data={k:archive[k].copy() for k in archive.files}
    if kind=='missing': data.pop('metadata')
    elif kind=='badshape': data['states']=np.zeros((1,4),dtype=int)
    elif kind=='nan': data['states']=np.zeros((1,5),dtype=int); data['values']=np.array([[0,np.nan]])
    elif kind=='out_of_range': data['states']=np.array([[999,0,0,0,0]]); data['values']=np.zeros((1,2))
    elif kind=='duplicate': data['states']=np.zeros((2,5),dtype=int); data['values']=np.zeros((2,2))
    else:
        meta=json.loads(str(data['metadata'])); meta['episodes']=-1; data['metadata']=json.dumps(meta)
    np.savez(p,**data)
    with pytest.raises(ValueError,match='Invalid agent checkpoint'): QAgent.load(p)


def test_evaluation_refuses_overwrite_and_handles_generator(tmp_path):
    agent=tiny_agent()
    result=training.evaluate(agent,(seed for seed in [1,2]),tmp_path)
    assert result['seeds']==[1,2]
    before=(tmp_path/'evaluation.csv').read_bytes()
    with pytest.raises(FileExistsError): training.evaluate(agent,[1,2],tmp_path)
    assert (tmp_path/'evaluation.csv').read_bytes()==before


def test_original_artifacts_and_policy_preserved(tmp_path):
    manifest=json.loads((ROOT/'.preservation-manifest.json').read_text())
    assert len(list((ROOT/'artifacts').glob('checkpoint_*.npz')))==24
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in manifest.items())
    agent=QAgent.load(ROOT/'artifacts/agent.npz'); assert agent.episodes==24000
    before={s:v.copy() for s,v in agent.q.items()}
    for folder,seeds in [('evaluation',range(100000,100020)),('evaluation_fresh',range(200000,200020))]:
        result=training.evaluate(agent,seeds,tmp_path/folder)
        assert result==json.loads((ROOT/'artifacts'/folder/'evaluation_summary.json').read_text())
    assert all(np.array_equal(value,agent.q[state]) for state,value in before.items())


def test_cli_errors_are_readable():
    for args in [['evaluate','--model','does-not-exist.npz'],['play','--speed','nan'],['train','--episodes','0']]:
        result=subprocess.run([sys.executable,'-m','flappy_ai',*args],cwd=ROOT,capture_output=True,text=True,timeout=15)
        assert result.returncode==2 and 'Error:' in result.stderr and 'Traceback' not in result.stderr


def test_real_headless_stop_file(tmp_path):
    config=tiny_agent().config; config.save(tmp_path/'config.json')
    process=subprocess.Popen([sys.executable,'-m','flappy_ai','train','--config',str(tmp_path/'config.json'),'--episodes','1000000','--output',str(tmp_path/'runs')],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+15
        run=None
        while time.monotonic()<deadline:
            candidates=list((tmp_path/'runs').glob('run_*/training.csv'))
            if candidates:
                try:
                    with candidates[0].open() as f: enough=len(list(csv.DictReader(f)))>=2
                    if enough: run=candidates[0].parent; break
                except OSError: pass
            time.sleep(.02)
        assert run is not None, 'Training failed to produce incremental statistics'
        (run/'STOP').touch()
        _,stderr=process.communicate(timeout=15)
        assert process.returncode==0,stderr
        assert QAgent.load(run/'agent.npz').episodes>=2
        assert json.loads((run/'status.json').read_text())['status']=='stopped'
    finally:
        if process.poll() is None:
            process.kill(); process.wait()


def test_abrupt_process_termination_keeps_completed_statistics(tmp_path):
    config=tiny_agent().config; config.save(tmp_path/'config.json')
    process=subprocess.Popen([sys.executable,'-m','flappy_ai','train','--config',str(tmp_path/'config.json'),'--episodes','1000000','--output',str(tmp_path/'runs')],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+15
        durable=None; run=None
        while time.monotonic()<deadline:
            candidates=list((tmp_path/'runs').glob('run_*/training.csv'))
            if candidates:
                raw=candidates[0].read_bytes()
                complete=raw[:raw.rfind(b'\n')+1]
                if len(complete.splitlines())>=3:
                    durable=complete; run=candidates[0].parent; break
            time.sleep(.02)
        assert durable is not None
        process.kill(); process.wait(timeout=10)
        assert (run/'training.csv').read_bytes().startswith(durable)
        assert QAgent.load(run/'latest.npz').episodes>=0
    finally:
        if process.poll() is None: process.kill(); process.wait()


def test_journal_fsyncs_each_row(tmp_path,monkeypatch):
    from flappy_ai import storage
    calls=[]; original=storage.os.fsync
    def fsync(fd): calls.append(fd); original(fd)
    monkeypatch.setattr(storage.os,'fsync',fsync)
    journal=CSVJournal(tmp_path/'statistics.csv',['episode'])
    try:
        journal.append({'episode':1}); journal.append({'episode':2})
        assert len(calls)==3  # Header and both rows flushed to the operating system.
    finally: journal.close()


@pytest.mark.skipif(os.name!='nt',reason='Windows extended keys')
def test_windows_extended_keys_do_not_pause(monkeypatch,tmp_path):
    import msvcrt
    keys=iter(['\xe0','P'])
    available=iter([True,False])
    monkeypatch.setattr(msvcrt,'kbhit',lambda:next(available))
    monkeypatch.setattr(msvcrt,'getwch',lambda:next(keys))
    monkeypatch.setattr(sys,'stdin',type('Terminal',(),{'isatty':lambda self:True})())
    assert not TrainingControl(tmp_path).poll()
