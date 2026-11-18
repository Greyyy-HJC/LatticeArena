import argparse

from .common import DEFAULT_OUTPUT


def main():
    parser = argparse.ArgumentParser(description='Static potential from smeared spatial Wilson lines')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    commands = parser.add_subparsers(dest='command', required=True)
    measure = commands.add_parser('measure')
    measure.add_argument('--split', choices=['development', 'validation'], required=True)
    measure.add_argument('--limit', type=int)
    commands.add_parser('develop')
    commands.add_parser('evaluate')
    commands.add_parser('plot')
    args = parser.parse_args()
    if args.command == 'measure':
        from .measure import measure
        measure(args.output, args.split, args.limit)
    elif args.command == 'develop':
        from .experiment import develop
        develop(args.output)
    elif args.command == 'evaluate':
        from .experiment import evaluate
        evaluate(args.output)
    else:
        from .plots import plot
        plot(args.output)


if __name__ == '__main__':
    main()
