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
    max_steps: int = 18000
    target_score: int = 100

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
        if not (0 < g.pipe_gap < g.ground - 80 and g.pipe_spacing > g.pipe_width and g.pipe_speed > 0 and g.gravity > 0 and g.flap_velocity < 0):
            raise ValueError('Invalid pipe geometry or physics')
        if not (g.max_steps > 0 and g.target_score > 0 and 0 < g.bird_radius < g.pipe_gap / 2 and 0 < g.bird_x < g.width and g.height > g.ground > 100):
            raise ValueError('Invalid world dimensions or episode limits')
        if not (0 < t.alpha <= 1 and 0 <= t.gamma < 1 and 0 <= t.epsilon_min <= t.epsilon_start <= 1 and 0 < t.epsilon_decay <= 1):
            raise ValueError('Invalid learning parameters')
        if len(t.bins) != 5 or any(not isinstance(b, int) or b < 2 for b in t.bins) or t.episodes < 1 or t.checkpoint_every < 1:
            raise ValueError('Invalid bins or training counts')
        if not (0 < g.max_velocity <= g.bird_radius and abs(g.flap_velocity) <= g.bird_radius and g.pipe_speed <= g.bird_radius):
            raise ValueError('Physics steps must be small enough for reliable collisions')
        if not all(math.isfinite(x) for x in asdict(self.rewards).values()):
            raise ValueError('Rewards must be finite')
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path=None):
        if not path:
            return cls().validate()
        d = json.loads(Path(path).read_text())
        return cls(GameConfig(**d.get('game', {})), RewardConfig(**d.get('rewards', {})), TrainingConfig(**d.get('training', {}))).validate()
