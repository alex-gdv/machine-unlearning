import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json

def make_retain_forget_json(dir, frac):
    train = pd.read_json(f"{dir}/train.json").transpose()
    retain, forget = train_test_split(train, test_size=frac, stratify=train["decade"])

    with open(f"{dir}/retain.json", "w+") as retain_fp:
        json.dump(retain.to_dict(orient="index"), retain_fp)

    with open(f"{dir}/forget.json", "w+") as forget_fp:
        json.dump(forget.to_dict(orient="index"), forget_fp)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="directory containing JSON file with training labels")
    parser.add_argument("--frac", required=False, default=0.1, help="fraction of train set to be forgotten")
    args = parser.parse_args()

    assert os.path.isdir(args.dir), "directory does not exist"

    make_retain_forget_json(
        dir=args.dir,
        frac=args.frac
    )