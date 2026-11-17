"""Sparse tabular Q learning with normalized, configurable observation bins."""
import json
import os
import tempfile
import zipfile
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .config import Config, GameConfig, RewardConfig, TrainingConfig
from .storage import replace_with_retry

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
        ranges = self.config.training.state_ranges or [(0,g.ground), (g.flap_velocity,g.max_velocity), (-g.pipe_width-g.bird_radius,g.width), (-g.ground,g.ground), (-g.ground,g.ground)]
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
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, suffix='.tmp', delete=False) as f:
                temp = Path(f.name)
                np.savez_compressed(f, states=np.array(list(self.q), dtype=np.int32).reshape(-1, 5),
                                    values=np.array(list(self.q.values()), dtype=float).reshape(-1, 2),
                                    metadata=json.dumps(meta))
                f.flush()
                os.fsync(f.fileno())
            replace_with_retry(temp, path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        try:
            with np.load(path, allow_pickle=False) as d:
                m = json.loads(str(d['metadata']))
                if m['version'] != 1:
                    raise ValueError('Unsupported checkpoint version')
                c = m['config']
                agent = cls(Config(GameConfig(**c['game']), RewardConfig(**c['rewards']),
                                   TrainingConfig(**c['training'])).validate())
                states, values = d['states'], d['values']
                if states.ndim != 2 or states.shape[1] != 5 or values.shape != (len(states), 2):
                    raise ValueError('Invalid Q-table shape')
                if not np.issubdtype(states.dtype, np.integer) or not np.isfinite(values).all():
                    raise ValueError('Q-table must contain integer states and finite values')
                if np.any(states < 0) or np.any(states >= np.array(agent.config.training.bins)):
                    raise ValueError('Q-table state is outside configured bins')
                agent.q = {tuple(map(int, s)): v.astype(float).copy() for s, v in zip(states, values)}
                if len(agent.q) != len(states):
                    raise ValueError('Duplicate Q-table states')
                if type(m['episodes']) is not int or m['episodes'] < 0:
                    raise ValueError('Invalid completed episode count')
                if not (np.isfinite(m['epsilon']) and 0 <= m['epsilon'] <= 1
                        and np.isfinite(m['duration']) and m['duration'] >= 0):
                    raise ValueError('Invalid checkpoint epsilon or duration')
                agent.epsilon = m['epsilon']
                agent.episodes = m['episodes']
                agent.duration = m['duration']
                agent.rng.bit_generator.state = m['rng']
                return agent
        except (KeyError, TypeError, IndexError, AttributeError, ValueError, EOFError, zipfile.BadZipFile) as error:
            raise ValueError(f'Invalid agent checkpoint {path}: {error}') from error
