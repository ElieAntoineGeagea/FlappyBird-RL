"""Windows-friendly command line entry point."""
import argparse
import json
from pathlib import Path
from .config import Config
from .agent import QAgent
from .training import train, evaluate
from .storage import unique_directory


DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / 'experiments' / 'study_20261117_122405' / 'recommended_compatible_agent.npz'
)


def main():
    parser = argparse.ArgumentParser(description='Flappy Bird AI - tabular Q-Learning')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('play')
    p.add_argument('--model', default=DEFAULT_MODEL, help='Saved policy (default: improved submission agent)')
    p.add_argument('--config')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--speed', type=float, default=1)
    p = sub.add_parser('train')
    p.add_argument('--config')
    p.add_argument('--resume')
    p.add_argument('--episodes', type=int)
    p.add_argument('--output', default='artifacts')
    p.add_argument('--render', action='store_true')
    p.add_argument('--checkpoint-seconds', type=float, default=60,
                   help='Maximum wall seconds between automatic latest checkpoints')
    p = sub.add_parser('evaluate')
    p.add_argument('--model', default=DEFAULT_MODEL, help='Saved policy (default: improved submission agent)')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(100000, 100020)))
    p.add_argument('--output', default='artifacts/evaluation')
    p = sub.add_parser('plot', help='Regenerate reward, score and epsilon graphs from an existing CSV')
    p.add_argument('--csv', required=True)
    p.add_argument('--output', default='artifacts/plots')
    args = parser.parse_args()
    try:
        if args.command == 'play':
            from .ui import play
            if not 0.25 <= args.speed <= 8:
                raise ValueError('Speed must be between 0.25 and 8')
            play(Config.load(args.config), args.model, args.seed, args.speed)
        elif args.command == 'train':
            if args.resume and args.config:
                raise ValueError('Resume uses checkpoint configuration; omit --config')
            agent = QAgent.load(args.resume) if args.resume else QAgent(Config.load(args.config))
            count = args.episodes if args.episodes is not None else agent.config.training.episodes
            train(agent, count, args.output, args.render, checkpoint_seconds=args.checkpoint_seconds)
        elif args.command == 'evaluate':
            agent = QAgent.load(args.model)
            output = unique_directory(args.output, 'evaluation')
            print(json.dumps(evaluate(agent, args.seeds, output), indent=2))
            print(f'Evaluation saved: {output.resolve()}')
        else:
            import csv
            import math
            from .training import plot_rows
            with open(args.csv, newline='', encoding='utf-8-sig') as f:
                rows = []
                for row in csv.DictReader(f):
                    converted = {k: float(row[k]) for k in ('episode', 'score', 'reward', 'epsilon', 'rolling_average')}
                    if not all(math.isfinite(value) for value in converted.values()):
                        raise ValueError('Statistics must contain finite numbers')
                    rows.append(converted)
            if not rows:
                raise ValueError('Training CSV contains no completed episodes')
            output = unique_directory(args.output, 'plot') / 'training.png'
            plot_rows(rows, output)
            print(f'Training graph saved: {output.resolve()}')
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(2, f'Error: {error}\nCheck file paths, configuration values, and output folder permissions.\n')


if __name__ == '__main__':
    main()
