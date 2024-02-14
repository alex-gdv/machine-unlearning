from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import os


def make_label_json(source_dir, meta_dir):
    image_paths = os.listdir(source_dir)
    
    labels = pd.DataFrame()
    labels["paths"] = [f"{source_dir}/{image_path}" for image_path in image_paths]
    labels["label"] = [int(image_path.split("_")[0]) for image_path in image_paths]

    labels.groupby(["label"]).count().to_csv("group_count.csv")
    
    # train_labels, test_validation_labels = train_test_split(labels, train_size=0.7, stratify=labels["label"])
    # print(train_labels)


    # with open(f"{}", "w+") as outfile_fp:
    #     json.dump(image_labels, outfile_fp)    
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="directory containing images to label")
    parser.add_argument("--meta_dir", required=True, help="directort to output label JSON files for test, train and val")
    args = parser.parse_args()

    assert os.path.isdir(args.source_dir), "source directory does not exist"
    assert os.path.isdir(args.meta_dir), "meta directory does not exist"

    make_label_json(
        source_dir=args.source_dir,
        meta_dir=args.meta_dir
    )