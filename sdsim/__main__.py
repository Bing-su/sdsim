import argparse
import json
from pathlib import Path

from .sdsim import similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("targets", type=Path, nargs="+")
    parser.add_argument("-j", "--json", type=Path, default=None)
    parser.add_argument("-s", "--seed", type=int, default=114514)
    parser.add_argument("-nv", "--no-verbose", action="store_false")
    args = parser.parse_args()

    result = similarity(
        base=args.base,
        targets=args.targets,
        seed=args.seed,
        verbose=args.no_verbose,
    )

    if args.json is not None:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
