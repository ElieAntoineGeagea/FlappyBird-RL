"""Sparse tabular Q learning with normalized, configurable observation bins."""
import json
import os
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .config import Config, GameConfig, RewardConfig, TrainingConfig

class QAgent:
    def __init__(self, config, seed=None):
        self.config = config
        self.q = {}
        self.epsilon = config.training.epsilon_start
        self.episodes = 0
        self.duration = 0.0
        self.rng = np.random.default_rng(config.training.seed if seed is None else seed)

    def state(self, observation):
        g = self.config.game
        ranges = [(0,g.ground), (g.flap_velocity,g.max_velocity), (-g.pipe_width-g.bird_radius,g.width), (-g.ground,g.ground), (-g.ground,g.ground)]
        return tuple(max(0, min(n-1, int((v-lo)/(hi-lo)*n))) for v,(lo,hi),n in zip(observation,ranges,self.config.training.bins))

    def values(self, state):
        return self.q.setdefault(state, np.zeros(2, dtype=np.float64))

    def act(self, observation, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        values = self.q.get(self.state(observation))
        return int(np.argmax(values)) if values is not None else 0

    def update(self, observation, action, reward, next_observation, terminal):
        values = self.values(self.state(observation))
        target = reward if terminal else reward + self.config.training.gamma * float(np.max(self.values(self.state(next_observation))))
        values[action] += self.config.training.alpha * (target-values[action])

    def finish_episode(self):
        self.episodes += 1
        self.epsilon = max(self.config.training.epsilon_min, self.epsilon*self.config.training.epsilon_decay)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(config=asdict(self.config), epsilon=self.epsilon, episodes=self.episodes, duration=self.duration, rng=self.rng.bit_generator.state, version=1)
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, states=np.array(list(self.q),dtype=np.int32).reshape(-1,5), values=np.array(list(self.q.values()),dtype=float).reshape(-1,2), metadata=json.dumps(meta))
        os.replace(temp,path)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata']))
            if m['version'] != 1: raise ValueError('Unsupported checkpoint version')
            c=m['config']
            agent=cls(Config(GameConfig(**c['game']),RewardConfig(**c['rewards']),TrainingConfig(**c['training'])).validate())
            if d['states'].shape != (len(d['values']),5) or d['values'].shape[1:] != (2,): raise ValueError('Invalid Q-table')
            agent.q={tuple(map(int,s)):v.copy() for s,v in zip(d['states'],d['values'])}
            agent.epsilon=m['epsilon']; agent.episodes=m['episodes']; agent.duration=m['duration']
            agent.rng.bit_generator.state=m['rng']
            return agent
