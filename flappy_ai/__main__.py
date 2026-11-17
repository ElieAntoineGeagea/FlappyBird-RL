import argparse
import json
from .config import Config
from .agent import QAgent
from .training import train, evaluate


def main():
    parser=argparse.ArgumentParser(description='Flappy Bird AI - tabular Q-Learning')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('play'); p.add_argument('--model',default='artifacts/agent.npz'); p.add_argument('--config'); p.add_argument('--seed',type=int,default=2026); p.add_argument('--speed',type=float,default=1)
    p=sub.add_parser('train'); p.add_argument('--config'); p.add_argument('--resume'); p.add_argument('--episodes',type=int); p.add_argument('--output',default='artifacts'); p.add_argument('--render',action='store_true')
    p=sub.add_parser('evaluate'); p.add_argument('--model',default='artifacts/agent.npz'); p.add_argument('--seeds',type=int,nargs='+',default=list(range(100000,100020))); p.add_argument('--output',default='artifacts/evaluation')
    args=parser.parse_args()
    if args.command=='play':
        from .ui import play
        if not 0.25<=args.speed<=8: parser.error('Speed must be between 0.25 and 8')
        play(Config.load(args.config),args.model,args.seed,args.speed)
    elif args.command=='train':
        if args.resume and args.config: parser.error('Resume uses checkpoint configuration; omit --config')
        agent=QAgent.load(args.resume) if args.resume else QAgent(Config.load(args.config))
        count=args.episodes if args.episodes is not None else agent.config.training.episodes
        if count<1: parser.error('Episodes must be positive')
        train(agent,count,args.output,args.render)
    else:
        print(json.dumps(evaluate(QAgent.load(args.model),args.seeds,args.output),indent=2))

if __name__=='__main__': main()
