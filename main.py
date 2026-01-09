from rll.taxi import run as run_taxi, train_with_hyperparameters as train_taxi
from rll.mountain_car_sarsa import (
    train_with_hyperparameters as train_mountain_car_sarsa,
)
import argparse


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("project", type=str, help="project name")
    argparser.add_argument("trainOrRun", type=str, help="train or run the project")
    argparser.add_argument(
        "model", type=str, nargs="?", default=None, help="model path"
    )

    args = argparser.parse_args()

    if args.project == "taxi":
        if args.trainOrRun == "train":
            train_taxi()
        elif args.trainOrRun == "run":
            if not args.model:
                argparser.error("model is required")
            run_taxi(args.model)

    elif args.project == "mountain_car":
        if args.trainOrRun == "train":
            train_mountain_car_sarsa()


if __name__ == "__main__":
    main()
