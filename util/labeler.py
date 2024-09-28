from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import os


def make_label_json(source_dir, meta_dir):
    image_paths = os.listdir(source_dir)
    
    labels = pd.DataFrame()
    labels["path"] = [f"{source_dir}/{image_path}" for image_path in image_paths]
    labels["label"] = [int(image_path.split("_")[0]) for image_path in image_paths]
    labels = labels[labels["label"] < 100]

    labels["decade"] = labels["label"] // 10

    labels.set_index("path", inplace=True)
    
    # train = 0.7, test = 0.2, val = 0.1 
    train, test_val = train_test_split(labels, train_size=0.7, stratify=labels["decade"])
    test, val = train_test_split(test_val, train_size=2/3, stratify=test_val["decade"])

    with open(f"{meta_dir}/train.json", "w+") as train_fp:
        json.dump(train.to_dict(orient="index"), train_fp) 

    with open(f"{meta_dir}/test.json", "w+") as test_fp:
        json.dump(test.to_dict(orient="index"), test_fp)    

    with open(f"{meta_dir}/val.json", "w+") as val_fp:
        json.dump(test.to_dict(orient="index"), val_fp)    
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="directory containing images to label")
    parser.add_argument("--meta_dir", required=True, help="directory to output label JSON files for test, train and val")
    args = parser.parse_args()

    assert os.path.isdir(args.source_dir), "source directory does not exist"
    assert os.path.isdir(args.meta_dir), "meta directory does not exist"

    make_label_json(
        source_dir=args.source_dir,
        meta_dir=args.meta_dir
    )
