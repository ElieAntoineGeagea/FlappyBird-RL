from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path

@dataclass
class GameConfig:
    width: int = 480
    height: int = 640
    ground: int = 590
    bird_x: float = 115
    bird_radius: float = 13
    gravity: float = 0.42
    flap_velocity: float = -6.5
    max_velocity: float = 10
    pipe_speed: float = 3
    pipe_width: float = 70
    pipe_gap: float = 175
    pipe_spacing: float = 240
    max_steps: int = 100000
    target_score: int = 1000

@dataclass
class RewardConfig:
    survive: float = 0.05
    pipe: float = 10
    death: float = -10

@dataclass
class TrainingConfig:
    alpha: float = 0.18
    gamma: float = 0.98
    epsilon_start: float = 1
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995
    bins: list = field(default_factory=lambda: [8, 12, 12, 24, 24])
    state_ranges: list | None = None
    episodes: int = 12000
    checkpoint_every: int = 1000
    seed: int = 42

@dataclass
class Config:
    game: GameConfig = field(default_factory=GameConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self):
        g, t = self.game, self.training

        def number(name, value, integer=False):
            valid_type = type(value) is int if integer else type(value) in (int, float)
            if not valid_type or (type(value) is float and not math.isfinite(value)):
                raise ValueError(f'{name} must be a finite {"integer" if integer else "number"}')

        for name, value in asdict(g).items():
            number(f'game.{name}', value, name in ('width', 'height', 'ground', 'max_steps', 'target_score'))
        for name, value in asdict(self.rewards).items():
            number(f'rewards.{name}', value)
        for name in ('alpha', 'gamma', 'epsilon_start', 'epsilon_min', 'epsilon_decay'):
            number(f'training.{name}', getattr(t, name))
        for name in ('episodes', 'checkpoint_every', 'seed'):
            number(f'training.{name}', getattr(t, name), True)
        if not (0 < g.width <= 8192 and 100 < g.ground < g.height <= 8192):
            raise ValueError('World dimensions must be positive and at most 8192, with height > ground > 100')
        if not (g.bird_radius > 0 and g.bird_radius < g.bird_x < g.width - g.bird_radius):
            raise ValueError('Bird radius must be positive and bird_x must keep it inside the world')
        if not (20 <= g.pipe_width < g.width):
            raise ValueError('pipe_width must be at least 20 and less than world width')
        if not (2 * g.bird_radius < g.pipe_gap < g.ground - 80):
            raise ValueError('pipe_gap must fit the bird and leave 40-pixel margins')
        if g.pipe_spacing <= g.pipe_width + 2 * g.bird_radius:
            raise ValueError('pipe_spacing must exceed pipe_width + twice bird_radius')
        if not (0 < g.gravity <= g.bird_radius and -g.bird_radius <= g.flap_velocity < 0
                and 0 < g.max_velocity <= g.bird_radius and 0 < g.pipe_speed <= g.bird_radius):
            raise ValueError('Physics values must have valid directions and steps no larger than bird_radius')
        if g.max_steps < 1 or g.target_score < 1:
            raise ValueError('max_steps and target_score must be positive integers')
        if not (0 < t.alpha <= 1 and 0 <= t.gamma < 1):
            raise ValueError('alpha must be in (0, 1] and gamma in [0, 1)')
        if not (0 <= t.epsilon_min <= t.epsilon_start <= 1 and 0 < t.epsilon_decay <= 1):
            raise ValueError('Invalid exploration range or epsilon_decay')
        if not isinstance(t.bins, (list, tuple)) or len(t.bins) != 5 or any(type(b) is not int or not 2 <= b <= 2147483647 for b in t.bins):
            raise ValueError('bins must contain five integers between 2 and 2147483647')
        if t.state_ranges is not None:
            if not isinstance(t.state_ranges, list) or len(t.state_ranges) != 5:
                raise ValueError('state_ranges must contain five [low, high] pairs')
            for pair in t.state_ranges:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError('Invalid state range pair')
                for value in pair:
                    number('state range', value)
                if not pair[0] < pair[1]:
                    raise ValueError('State range low must be below high')
        if t.episodes < 1 or t.checkpoint_every < 1 or t.seed < 0:
            raise ValueError('Training counts must be positive integers and seed must be nonnegative')
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path=None):
        if not path:
            return cls().validate()
        try:
            d = json.loads(Path(path).read_text(encoding='utf-8-sig'))
            if not isinstance(d, dict) or set(d) - {'game', 'rewards', 'training'}:
                raise ValueError('Expected game, rewards and training configuration sections')
            return cls(GameConfig(**d.get('game', {})), RewardConfig(**d.get('rewards', {})), TrainingConfig(**d.get('training', {}))).validate()
        except (TypeError, AttributeError, json.JSONDecodeError) as error:
            raise ValueError(f'Invalid configuration {path}: {error}') from error
