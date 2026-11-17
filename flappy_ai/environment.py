"""Deterministic fixed-timestep environment; no rendering dependency."""
from dataclasses import dataclass
import random

@dataclass
class Pipe:
    x: float
    center: float
    passed: bool = False

class Environment:
    def __init__(self, config, *, limited=True):
        self.config = config
        self.limited = limited
        self.g = config.game
        self.reset(0)

    def reset(self, seed=0):
        self.rng = random.Random(int(seed))
        self.y = self.g.ground / 2
        self.velocity = 0.0
        self.steps = self.score = 0
        self.done = False
        self.reason = ''
        self.pipes = [self.new_pipe(self.g.width + i * self.g.pipe_spacing) for i in range(3)]
        return self.observe()

    def new_pipe(self, x):
        margin = self.g.pipe_gap / 2 + 40
        return Pipe(x, self.rng.uniform(margin, self.g.ground - margin))

    def next_pipe(self):
        return next(p for p in self.pipes if p.x + self.g.pipe_width >= self.g.bird_x - self.g.bird_radius)

    def observe(self):
        p = self.next_pipe()
        upper, lower = p.center - self.g.pipe_gap / 2, p.center + self.g.pipe_gap / 2
        return (self.y, self.velocity, p.x - self.g.bird_x, self.y - upper, lower - self.y)

    def collides(self):
        g = self.g
        if self.y - g.bird_radius <= 0 or self.y + g.bird_radius >= g.ground:
            return True
        for p in self.pipes:
            for top, bottom in ((0, p.center-g.pipe_gap/2), (p.center+g.pipe_gap/2, g.ground)):
                nx = max(p.x, min(g.bird_x, p.x+g.pipe_width))
                ny = max(top, min(self.y, bottom))
                if (g.bird_x-nx)**2 + (self.y-ny)**2 <= g.bird_radius**2:
                    return True
        return False

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset before stepping a finished episode')
        if action not in (0, 1):
            raise ValueError('Actions: 0=DO_NOTHING, 1=FLAP')
        g, r = self.g, self.config.rewards
        if action:
            self.velocity = g.flap_velocity
        self.velocity = min(g.max_velocity, self.velocity + g.gravity)
        self.y += self.velocity
        self.steps += 1
        for p in self.pipes:
            p.x -= g.pipe_speed
        reward = r.survive
        if self.collides():
            self.done, self.reason = True, 'collision'
            reward = r.death
        else:
            for p in self.pipes:
                if not p.passed and p.x + g.pipe_width < g.bird_x - g.bird_radius:
                    p.passed = True
                    self.score += 1
                    reward += r.pipe
            if self.limited and (self.score >= g.target_score or self.steps >= g.max_steps):
                self.done = True
                self.reason = 'target' if self.score >= g.target_score else 'time_limit'
        while self.pipes[0].x + g.pipe_width < 0:
            self.pipes.pop(0)
            self.pipes.append(self.new_pipe(self.pipes[-1].x + g.pipe_spacing))
        return self.observe(), reward, self.done, {'score': self.score, 'reason': self.reason}
