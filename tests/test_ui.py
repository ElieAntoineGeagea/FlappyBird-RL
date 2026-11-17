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
        def tick(self,*args): return 34
    original=ui.Canvas
    class FastCanvas(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs); self.clock=Clock()
    monkeypatch.setattr(ui,'Canvas',FastCanvas)
    batches=iter([[pygame.K_h],[pygame.K_r],[pygame.K_ESCAPE],[pygame.K_c],[pygame.K_RETURN],[],[pygame.K_r],[pygame.K_ESCAPE],[pygame.K_a],[pygame.K_ESCAPE],[pygame.K_q]])
    monkeypatch.setattr(pygame.event,'get',lambda: [pygame.event.Event(pygame.KEYDOWN,key=k) for k in next(batches,[pygame.K_q])])
    ui.play(c,str(model))
    assert (tmp_path/'artifacts'/'comparison.csv').exists()


def test_training_pause_resume(monkeypatch):
    c=Config(); view=ui.TrainingView(c)
    batches=iter([[pygame.K_p],[pygame.K_p]])
    monkeypatch.setattr(pygame.event,'get',lambda: [pygame.event.Event(pygame.KEYDOWN,key=k) for k in next(batches,[])])
    assert view.update(Environment(c),QAgent(c),0,[]) is True
    assert view.paused is False
    view.close()
