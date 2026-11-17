"""Original procedural artwork and Pygame interface."""
import math
from pathlib import Path
import pygame
from .environment import Environment
from .agent import QAgent
from .storage import save_comparison

BG=(16,26,47); WHITE=(231,240,250); GOLD=(255,204,77); TEAL=(40,204,168)

class Canvas:
    def __init__(self,config,title='Flappy Bird AI'):
        pygame.init(); self.config=config
        self.screen=pygame.display.set_mode((config.game.width,config.game.height))
        pygame.display.set_caption(title)
        self.font=pygame.font.SysFont('segoeui',20); self.big=pygame.font.SysFont('segoeui',34,bold=True)
        self.clock=pygame.time.Clock()
    def text(self,text,x,y,color=WHITE,big=False):
        self.screen.blit((self.big if big else self.font).render(str(text),True,color),(x,y))
    def scene(self,env,label=''):
        s=self.screen; g=env.g; s.fill((92,181,211))
        for i in range(6):
            x=(i*111-env.steps*.3)%(g.width+130)-70
            pygame.draw.ellipse(s,(201,235,237),(x,95+i%3*34,100,30))
        for i in range(9):
            x=i*65; h=45+(i*23)%70
            pygame.draw.rect(s,(75,151,171),(x,g.ground-h,55,h))
        for p in env.pipes:
            upper=p.center-g.pipe_gap/2; lower=p.center+g.pipe_gap/2
            for y,h in ((0,upper),(lower,g.ground-lower)):
                rect=pygame.Rect(p.x,y,g.pipe_width,h)
                pygame.draw.rect(s,(27,129,102),rect,border_radius=5)
                pygame.draw.rect(s,(49,192,133),(p.x+7,y,g.pipe_width-18,h),border_radius=4)
            for y in (upper-20,lower):
                pygame.draw.rect(s,(22,113,94),(p.x,y,g.pipe_width,20),border_radius=4)
                pygame.draw.line(s,(93,227,155),(p.x,y+3),(p.x+g.pipe_width,y+3),3)
        pygame.draw.rect(s,(224,194,132),(0,g.ground,g.width,g.height-g.ground))
        pygame.draw.rect(s,(62,159,101),(0,g.ground,g.width,8))
        bird=pygame.Surface((54,46),pygame.SRCALPHA)
        pygame.draw.ellipse(bird,GOLD,(9,10,30,27)); pygame.draw.ellipse(bird,(233,149,40),(6,21+int(math.sin(env.steps*.5)*4),20,12))
        pygame.draw.circle(bird,WHITE,(32,17),7); pygame.draw.circle(bird,BG,(34,17),3)
        pygame.draw.polygon(bird,(249,102,58),[(38,23),(51,27),(38,30)])
        bird=pygame.transform.rotate(bird,max(-60,min(25,-env.velocity*5)))
        s.blit(bird,bird.get_rect(center=(g.bird_x,env.y)))
        self.text(label,18,14,BG); self.text(str(env.score),g.width//2-12,42,WHITE,True)
    def panel(self,lines):
        overlay=pygame.Surface(self.screen.get_size(),pygame.SRCALPHA); overlay.fill((10,20,40,215)); self.screen.blit(overlay,(0,0))
        y=120
        for i,line in enumerate(lines):
            self.text(line,28,y,GOLD if i==0 else WHITE,i==0); y+=52 if i==0 else 36
    def close(self): pygame.quit()

class TrainingView(Canvas):
    def __init__(self,config):
        super().__init__(config,'Q-Learning - training'); self.speed=1000; self.paused=False; self.notice=''
    def update(self,env,agent,total,recent,on_checkpoint=None,should_stop=None,external_paused=False):
        def checkpoint():
            if on_checkpoint:
                on_checkpoint()
                self.notice='Checkpoint saved.'
                print(self.notice, flush=True)
        while True:
            if should_stop and should_stop(): return False
            for event in pygame.event.get():
                if event.type==pygame.QUIT: return False
                if event.type==pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE,pygame.K_s): return False
                    if event.key==pygame.K_p:
                        self.paused=not self.paused
                        if self.paused: checkpoint()
                    if event.key==pygame.K_c: checkpoint()
                    if event.key in (pygame.K_EQUALS,pygame.K_KP_PLUS): self.speed=min(10000,self.speed*2)
                    if event.key in (pygame.K_MINUS,pygame.K_KP_MINUS): self.speed=max(1,self.speed//2)
            self.scene(env,'Q-LEARNING / TRAINING')
            self.panel(['Paused' if self.paused or external_paused else 'Training',f'Episode {agent.episodes+1}   score {env.score}',f'Epsilon {agent.epsilon:.4f}',f'Reward {total:.2f}',f'Rolling average {sum(recent)/max(1,len(recent)):.2f}',f'{self.speed} simulation steps / display','P pause  C checkpoint  S stop','+ / - training speed',self.notice])
            pygame.display.flip(); self.clock.tick(60)
            if not self.paused: return True


def play(config, model='artifacts/agent.npz',seed=2026,speed=1.0):
    agent=None
    if Path(model).exists():
        try:
            agent=QAgent.load(model); config=agent.config
        except (OSError, ValueError) as error:
            print(f'Cannot load AI model {model}: {error}. Human mode remains available.', flush=True)
    if agent is None and not Path(model).exists():
        print(f'AI model not found: {model}. Human mode is available; load a valid model for AI/comparison.', flush=True)
    view=Canvas(config); env=Environment(config); env.reset(seed)
    mode='menu'; paused=False; accumulator=0.0; running=True; comparison=False; pending_flap=False
    human_score=None; message='' if agent else 'AI unavailable: check model path.'; comparison_message=''
    while running:
        dt=min(view.clock.tick(60)/1000,.1)
        for event in pygame.event.get():
            if event.type==pygame.QUIT: running=False
            if event.type!=pygame.KEYDOWN: continue
            k=event.key
            if k==pygame.K_ESCAPE:
                mode='menu'; paused=False; comparison=False
            if not comparison and k in (pygame.K_EQUALS,pygame.K_KP_PLUS): speed=min(8,speed+.25)
            if not comparison and k in (pygame.K_MINUS,pygame.K_KP_MINUS): speed=max(.25,speed-.25)
            if mode=='menu':
                if k in (pygame.K_h,pygame.K_a,pygame.K_c):
                    if k!=pygame.K_h and agent is None: message='Train an agent first (see README).'; continue
                    comparison=k==pygame.K_c; mode='human' if k!=pygame.K_a else 'ai'
                    env.reset(seed); accumulator=0; pending_flap=False; human_score=None; message=''
                    if comparison: speed=1.0
                if k==pygame.K_q: running=False
            elif mode in ('human','ai'):
                if k==pygame.K_SPACE and mode=='human': pending_flap=True
                if k==pygame.K_p: paused=not paused; accumulator=0
                if k==pygame.K_r: env.reset(seed); accumulator=0; pending_flap=False
            elif mode=='over' and k==pygame.K_r:
                mode='human' if comparison or human_score is None else 'ai'; env.reset(seed); accumulator=0; pending_flap=False
            elif mode=='compare_next' and k==pygame.K_RETURN:
                mode='ai'; env.reset(seed); accumulator=0; pending_flap=False
        if mode in ('human','ai') and not paused:
            accumulator+=dt*speed
            while accumulator>=1/60 and not env.done:
                action=agent.act(env.observe(),False) if mode=='ai' else int(pending_flap)
                pending_flap=False; env.step(action); accumulator-=1/60
            if env.done:
                if comparison and mode=='human':
                    human_score=env.score; mode='compare_next'
                elif comparison:
                    try:
                        saved=save_comparison(dict(seed=seed,human_score=human_score,ai_score=env.score,speed=speed))
                        comparison_message='Comparison saved (see terminal).'
                        print(f'Comparison saved: {saved.resolve()}', flush=True)
                    except OSError as error:
                        comparison_message='Save failed (see terminal).'
                        print(f'Could not save comparison: {error}', flush=True)
                    mode='over'
                else:
                    human_score=None if mode=='human' else -1; mode='over'
        view.scene(env,f'{mode.upper()}  |  {speed:.2f}x')
        if mode=='menu': view.panel(['Flappy Bird AI','REINFORCEMENT LEARNING','H   Human flight','A   Trained AI','C   Human vs AI (same seed)','Q   Quit',message or 'Space flap | P pause | R restart','+ / - speed | Esc menu'])
        elif paused: view.panel(['Paused','P resume | Esc menu'])
        elif mode=='compare_next': view.panel(['Your flight is complete',f'Human score: {human_score}','Enter: AI flies the same seed'])
        elif mode=='over':
            lines=['Flight complete',f'Score: {env.score}  ({env.reason})']
            if comparison: lines += [f'Human {human_score} vs AI {env.score}',comparison_message]
            lines += ['R restart | Esc menu']; view.panel(lines)
        else: view.text('Space flap  |  P pause  |  Esc menu',16,config.game.height-35,BG)
        pygame.display.flip()
    view.close()
