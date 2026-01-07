from rll.taxi import run as run_taxi, train_with_hyperparameters as train_taxi
import argparse


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("project", type=str, help="project name")
    argparser.add_argument("trainOrRun", type=str, help="train or run the project")
    argparser.add_argument("model", type=str, help="model path")

    args = argparser.parse_args()

    if args.project == "taxi":
        if args.trainOrRun == "train":
            train_taxi()
        elif args.trainOrRun == "run":
            run_taxi(args.model)


if __name__ == "__main__":
    main()
