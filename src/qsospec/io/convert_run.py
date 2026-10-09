"""CLI: python -m qsospec.io.convert_run SOURCE DESTINATION."""

import argparse
import json
from .conversion import convert_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--model-storage", choices=["parameters", "arrays"], default="parameters")
    args = parser.parse_args()
    store = convert_run(args.source, args.destination, model_storage=args.model_storage)
    print(json.dumps(store.manifest["conversion"], indent=2))


if __name__ == "__main__":
    main()
