import argparse, os
from .config import cfg
from . import ingest, features, labels, train

def run_all():
    ingest.main()
    features.main()
    labels.main()
    train.main()

def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline")
    parser.add_argument("--stage", choices=["all","ingest","features","labels","train","score"], default="all")
    args = parser.parse_args()

    if args.stage == "all":
        run_all()
    elif args.stage == "ingest":
        ingest.main()
    elif args.stage == "features":
        features.main()
    elif args.stage == "labels":
        labels.main()
    elif args.stage == "train":
        train.main()
    elif args.stage == "score":
        train.score_full_series()

if __name__ == "__main__":
    main()
