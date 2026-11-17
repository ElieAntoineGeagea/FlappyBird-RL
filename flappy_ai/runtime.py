"""Runtime episode limits are independent of immutable saved policy configuration."""
from copy import deepcopy
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MODEL = PROJECT_ROOT / 'artifacts' / 'agent.npz'
IMPROVED_MODEL = PROJECT_ROOT / 'experiments' / 'study_20261117_122405' / 'recommended_compatible_agent.npz'


def episode_config(saved_config, target_score=None, max_steps=None):
    config = deepcopy(saved_config)
    if target_score is not None:
        if type(target_score) is not int or target_score < 1:
            raise ValueError('target_score must be a positive integer')
        config.game.target_score = target_score
    if max_steps is not None:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        config.game.max_steps = max_steps
    elif target_score is not None:
        g = config.game
        # First trailing-edge clearance plus subsequent pipe spacings and a margin.
        needed = math.ceil((g.width + g.pipe_width - g.bird_x + g.bird_radius
                            + (g.target_score - 1) * g.pipe_spacing) / g.pipe_speed) + 2
        config.game.max_steps = max(g.max_steps, needed + math.ceil(2*g.pipe_spacing/g.pipe_speed))
    return config.validate()
