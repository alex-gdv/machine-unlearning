from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
import argparse
import torch
import os

from model.dataset import UTKFaceRegression
from model.model import ResNet50Regression


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=False)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--encoding", default="regression", choices=["ordinal", "regression"])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test