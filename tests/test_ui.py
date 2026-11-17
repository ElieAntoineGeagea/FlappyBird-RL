"""SDL dummy driver tests exercise rendering and real menu/event transitions."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import pygame
from flappy_ai.config import Config
from flappy_ai.environment import Environment
from flappy_ai.agent import QAgent
from flappy_ai import ui


def test_procedural_render_and_training_stop(tmp_path):
    c=Config(); view=ui.TrainingView(c); env=Environment(c)
    view.scene(env,'Smoke test'); pygame.display.flip()
    pygame.image.save(view.screen,str(tmp_path/'scene.png'))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_s))
    assert view.update(env,QAgent(c),0,[]) is False
    view.close()


def test_menu_comparison_and_restart(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    c=Config(); c.game.max_steps=2
    model=tmp_path/'agent.npz'; QAgent(c).save(model)
    class Clock:
        def tick(self, *args):
            return 100

    current = None
    phases = iter([pygame.K_c, pygame.K_RETURN, pygame.K_r, pygame.K_ESCAPE, pygame.K_q])
    expected = iter(['MENU', 'COMPARE_NEXT', 'OVER', 'HUMAN', 'MENU'])
    next_screen = next(expected)

    class FastCanvas(ui.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.clock = Clock()

        def scene(self, env, label=''):
            nonlocal current
            assert not env.limited
            current = label.split()[0]
            super().scene(env, label)

    frame_budget = iter(range(120))

    def events():
        nonlocal next_screen
        assert next(frame_budget, None) is not None, current
        if current == next_screen:
            next_screen = next(expected, None)
            return [pygame.event.Event(pygame.KEYDOWN, key=next(phases))]
        return []

    monkeypatch.setattr(ui, 'Canvas', FastCanvas)
    monkeypatch.setattr(pygame.event, 'get', events)
    ui.play(c, model)
    assert len(list((tmp_path/'artifacts'/'comparisons').glob('comparison_*.csv'))) == 1



def test_training_pause_resume(monkeypatch):
    c=Config(); view=ui.TrainingView(c)
    batches=iter([[pygame.K_p],[pygame.K_p]])
    monkeypatch.setattr(pygame.event,'get',lambda: [pygame.event.Event(pygame.KEYDOWN,key=k) for k in next(batches,[])])
    assert view.update(Environment(c),QAgent(c),0,[]) is True
    assert view.paused is False
    view.close()


def test_visual_training_checkpoint_and_stop_while_paused(monkeypatch):
    view=ui.TrainingView(Config()); saved=[]
    batches=iter([[pygame.K_p,pygame.K_c],[pygame.K_s]])
    monkeypatch.setattr(pygame.event,'get',lambda: [pygame.event.Event(pygame.KEYDOWN,key=k) for k in next(batches,[])])
    try:
        assert not view.update(Environment(Config()),QAgent(Config()),0,[],on_checkpoint=lambda:saved.append(True))
        assert len(saved)==2 and view.paused and view.notice=='Checkpoint saved.'
    finally: view.close()


def test_visual_pause_obeys_external_stop(monkeypatch):
    view=ui.TrainingView(Config()); view.paused=True
    try:
        assert not view.update(Environment(Config()),QAgent(Config()),0,[],should_stop=lambda:True)
    finally: view.close()


def test_corrupt_model_keeps_human_mode_available(monkeypatch,tmp_path,capsys):
    model=tmp_path/'bad.npz'; model.write_bytes(b'invalid checkpoint')
    batches=iter([[pygame.K_a],[pygame.K_h],[pygame.K_ESCAPE],[pygame.K_q]])
    monkeypatch.setattr(pygame.event,'get',lambda: [pygame.event.Event(pygame.KEYDOWN,key=k) for k in next(batches,[pygame.K_q])])
    ui.play(Config(),str(model))
    assert 'Human mode remains available' in capsys.readouterr().out


def test_human_pause_speed_flap_and_comparison_lock(monkeypatch,tmp_path):
    c=Config(); c.game.max_steps=1000
    model=tmp_path/'agent.npz'; QAgent(c).save(model)
    frames=[]
    class Clock:
        def tick(self,*args): return 34
    original=ui.Canvas
    class ObservedCanvas(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs); self.clock=Clock()
        def scene(self,env,label=''):
            frames.append((env.steps,env.velocity,label))
            super().scene(env,label)
    monkeypatch.setattr(ui,'Canvas',ObservedCanvas)
    batches=iter([[pygame.K_h],[pygame.K_SPACE],[pygame.K_p],[pygame.K_EQUALS],[pygame.K_p],[pygame.K_r],[pygame.K_ESCAPE],[pygame.K_c],[pygame.K_EQUALS],[pygame.K_ESCAPE],[pygame.K_q]])
    monkeypatch.setattr(pygame.event,'get',lambda: [pygame.event.Event(pygame.KEYDOWN,key=k) for k in next(batches,[pygame.K_q])])
    ui.play(c,str(model))
    assert frames[1][1]<0
    assert frames[2][0]==frames[3][0]==frames[1][0]  # Paused despite speed input.
    assert '1.25x' in frames[3][2]
    assert frames[4][0]>frames[3][0]
    assert frames[5][0]<frames[4][0]  # Restart resets simulation steps.
    assert '1.00x' in frames[7][2] and '1.00x' in frames[8][2]


def test_cli_loads_improved_submission_checkpoint(monkeypatch, tmp_path):
    import sys
    from flappy_ai import __main__ as cli

    delivered = []

    def play(config, model, seed, speed):
        delivered.append(QAgent.load(model))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui, 'play', play)
    monkeypatch.setattr(sys, 'argv', ['flappy_ai', 'play'])
    cli.main()

    agent, = delivered
    assert agent.config.game.target_score == 100
    assert agent.config.training.seed == 424242
    assert agent.config.rewards.death == -15
    assert agent.episodes == 24000 and agent.q


def test_removed_shortcuts_leave_menu_and_ai_unchanged(monkeypatch):
    from flappy_ai.__main__ import DEFAULT_MODEL

    frames = []
    panels = []
    loads = []
    real_load = QAgent.load

    def load_checkpoint(path):
        loads.append((path, len(frames)))
        return real_load(path)

    monkeypatch.setattr(QAgent, 'load', load_checkpoint)

    class ObservedCanvas(ui.Canvas):
        def scene(self, env, label=''):
            frames.append((label, env.limited, env.g.target_score, env.g.max_steps))
            super().scene(env, label)

        def panel(self, lines):
            panels.append(lines)
            super().panel(lines)

    batches = iter([[], [pygame.K_t, pygame.K_1, pygame.K_2], [pygame.K_a],
                    [pygame.K_t, pygame.K_1, pygame.K_2], [pygame.K_ESCAPE, pygame.K_q]])
    monkeypatch.setattr(ui, 'Canvas', ObservedCanvas)
    monkeypatch.setattr(pygame.event, 'get', lambda: [
        pygame.event.Event(pygame.KEYDOWN, key=key) for key in next(batches, [])])
    ui.play(Config(), DEFAULT_MODEL)
    assert frames[0] == frames[1] == ('MENU  |  1.00x', False, 100, 18000)
    assert frames[2] == frames[3] == ('AI  |  1.00x', False, 100, 18000)
    assert panels[0] == panels[1]
    assert panels[0][1:5] == ['H — Human flight', 'A — Watch AI', 'C — Human vs AI', 'Q — Quit']
    assert loads == [(DEFAULT_MODEL, 2)]
