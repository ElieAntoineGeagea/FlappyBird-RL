"""Original procedural artwork and Pygame interface."""
import math
from pathlib import Path
import pygame
from .environment import Environment
from .agent import QAgent
from .storage import save_comparison
from .runtime import episode_config, ORIGINAL_MODEL, IMPROVED_MODEL

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
        self.text(label,18,14,BG)
        score=self.big.render(f'{env.score:,} / {g.target_score:,}',True,WHITE)
        self.screen.blit(score,score.get_rect(midtop=(g.width//2,48)))
    def panel(self,lines):
        overlay=pygame.Surface(self.screen.get_size(),pygame.SRCALPHA); overlay.fill((10,20,40,215)); self.screen.blit(overlay,(0,0))
        y=85 if len(lines)>10 else 120
        for i,line in enumerate(lines):
            self.text(line,28,y,GOLD if i==0 else WHITE,i==0); y+=48 if i==0 else (30 if len(lines)>10 else 36)
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



class Milestones:
    thresholds = (100, 250, 500, 750, 1000)

    def __init__(self):
        self.previous = 0
        self.current = None
        self.until = 0

    def update(self, score, now_ms):
        crossed = [value for value in self.thresholds if self.previous < value <= score]
        if crossed:
            self.current = crossed[-1]
            self.until = now_ms + 2200
        self.previous = score
        return self.current if now_ms < self.until else None


def play(config, model=None, seed=2026, speed=1.0, selected='improved', *, max_steps=None):
    """Frozen policies share runtime target/step limits; checkpoint bytes stay untouched."""
    paths = {'Original': ORIGINAL_MODEL, 'Improved': IMPROVED_MODEL}
    if model is not None:
        paths['Custom'] = Path(model)
    agents = {}
    for name, path in paths.items():
        try:
            agents[name] = QAgent.load(path)
        except (OSError, ValueError) as error:
            agents[name] = None
            print(f'Cannot load {name} AI model {path}: {error}. Human mode remains available.', flush=True)
    choice = 'Custom' if model is not None else selected.capitalize()
    target = config.game.target_score
    def session():
        source = agents[choice].config if agents[choice] else config
        step_cap = max_steps
        if step_cap is None and target == config.game.target_score:
            step_cap = config.game.max_steps
        return episode_config(source, target, step_cap)

    current = session()
    view = Canvas(current)
    env = Environment(current)
    env.reset(seed)
    mode = 'menu'
    paused = False
    accumulator = 0.0
    running = True
    comparison = False
    pending_flap = False
    human_score = None
    message = ''
    comparison_message = ''
    last_mode = 'human'
    milestones = Milestones()

    def reset_flight():
        nonlocal accumulator, pending_flap, paused, milestones
        env.reset(seed)
        accumulator = 0.0
        pending_flap = False
        paused = False
        milestones = Milestones()

    def change_session():
        nonlocal env
        new_config = session()
        if view.screen.get_size() != (new_config.game.width, new_config.game.height):
            view.screen = pygame.display.set_mode((new_config.game.width, new_config.game.height))
        view.config = new_config
        env = Environment(new_config)
        reset_flight()

    try:
        while running:
            dt = min(view.clock.tick(60)/1000, .1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type != pygame.KEYDOWN:
                    continue
                key = event.key
                if key == pygame.K_ESCAPE:
                    mode = 'menu'; paused = False; comparison = False
                if not comparison and key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(8, speed+.25)
                if not comparison and key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(.25, speed-.25)
                if mode == 'menu':
                    if key in (pygame.K_1, pygame.K_2):
                        choice = 'Original' if key == pygame.K_1 else 'Improved'
                        change_session(); message = ''
                    if key == pygame.K_t:
                        target = 100 if target != 100 else 1000
                        change_session()
                    if key in (pygame.K_h, pygame.K_a, pygame.K_c):
                        if key != pygame.K_h and agents[choice] is None:
                            message = 'Selected AI unavailable; choose 1 or 2.'
                            continue
                        comparison = key == pygame.K_c
                        mode = 'human' if key != pygame.K_a else 'ai'
                        reset_flight(); human_score = None; message = ''
                        if comparison:
                            speed = 1.0
                    if key == pygame.K_q:
                        running = False
                elif mode in ('human', 'ai'):
                    if key == pygame.K_SPACE and mode == 'human':
                        pending_flap = True
                    if key == pygame.K_p:
                        paused = not paused; accumulator = 0
                    if key == pygame.K_r:
                        reset_flight()
                elif mode in ('over', 'victory') and key == pygame.K_r:
                    mode = 'human' if comparison else last_mode
                    reset_flight()
                elif mode == 'compare_next' and key == pygame.K_RETURN:
                    mode = 'ai'
                    reset_flight()
            if mode in ('human', 'ai') and not paused:
                accumulator += dt*speed
                while accumulator >= 1/60 and not env.done:
                    action = agents[choice].act(env.observe(), False) if mode == 'ai' else int(pending_flap)
                    pending_flap = False
                    env.step(action)
                    accumulator -= 1/60
                if env.done:
                    last_mode = mode
                    if comparison and mode == 'human':
                        human_score = env.score
                        mode = 'compare_next'
                    else:
                        if comparison:
                            try:
                                saved = save_comparison(dict(seed=seed, agent=choice, target=target,
                                    human_score=human_score, ai_score=env.score, ai_reason=env.reason, speed=speed))
                                comparison_message = 'Comparison saved (see terminal).'
                                print(f'Comparison saved: {saved.resolve()}', flush=True)
                            except OSError as error:
                                comparison_message = 'Save failed (see terminal).'
                                print(f'Could not save comparison: {error}', flush=True)
                        mode = 'victory' if env.reason == 'target' else 'over'
            view.scene(env, f'{mode.upper()} | {choice} | {speed:.2f}x')
            milestone = milestones.update(env.score, pygame.time.get_ticks())
            if milestone is not None and mode in ('human','ai','victory'):
                text = view.font.render(f'Milestone: {milestone:,} pipes!', True, GOLD)
                rect = text.get_rect(center=(env.g.width//2, 112))
                pygame.draw.rect(view.screen, BG, rect.inflate(24,12), border_radius=8)
                view.screen.blit(text,rect)
            if mode == 'menu':
                view.panel(['Flappy Bird AI',f'Agent: {choice} | Target: {target:,}',
                    '1   Original Q-Learning agent','2   Improved Q-Learning agent',
                    'H   Human flight','A   Watch selected AI','C   Human vs selected AI',
                    'T   Toggle 100 / 1,000 target','Q   Quit',
                    message or 'Space flap | P pause | R restart','+ / - speed | Esc menu'])
            elif paused:
                view.panel(['Paused',f'{choice} | {env.score:,} / {target:,}','P resume | R restart | Esc menu'])
            elif mode == 'compare_next':
                view.panel(['Your flight is complete',f'Human: {human_score:,} / {target:,}',
                            f'End: {env.reason}','Enter: AI flies the same seed'])
            elif mode in ('over','victory'):
                title = 'Victory!' if mode == 'victory' else ('Time limit reached' if env.reason == 'time_limit' else 'Flight complete')
                lines = [title, f'{choice} | {env.score:,} / {target:,}',
                         'Target reached!' if mode == 'victory' else f'End: {env.reason}']
                if comparison:
                    lines += [f'Human {human_score:,} vs AI {env.score:,}', comparison_message]
                lines += ['R restart | Esc menu']
                view.panel(lines)
            else:
                view.text('Space flap | P pause | Esc menu',16,env.g.height-35,BG)
            pygame.display.flip()
    finally:
        view.close()
