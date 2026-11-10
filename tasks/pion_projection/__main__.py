import argparse
from .common import DEFAULT_OUTPUT


def main():
    parser = argparse.ArgumentParser(description='Fixed S16T16 pion projection pilot')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    commands = parser.add_subparsers(dest='command', required=True)
    measure = commands.add_parser('measure')
    measure.add_argument('--split', choices=['development', 'validation'], required=True)
    measure.add_argument('--limit', type=int)
    commands.add_parser('develop')
    commands.add_parser('evaluate')
    commands.add_parser('plot')
    candidate = commands.add_parser('candidate')
    candidate.add_argument('--definition', required=True)
    candidate.add_argument('--split', choices=['development', 'validation'], required=True)
    candidate.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'measure':
        from .measure import measure
        measure(args.output, args.split, args.limit)
    elif args.command == 'candidate':
        from .candidate import run_candidate
        run_candidate(args.output, args.definition, args.split, args.verify_only)
    elif args.command == 'plot':
        from .plots import plot
        plot(args.output)
    else:
        from .experiment import develop, evaluate
        (develop if args.command == 'develop' else evaluate)(args.output)


if __name__ == '__main__':
    main()
