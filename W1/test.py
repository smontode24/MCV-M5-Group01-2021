from models import available_models
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os

import sys
sys.path.append("/home/group01/M5_Group01/W1/utils")
from loaders import get_training_datasets, get_testing_dataset
from evaluation import multi_acc
from utils import polynomial_decay


NUM_WORKERS = 4
PRINT_INTERVAL = 10
# early_stopping
EA_PATIENCE = 30
NUM_EPOCHS = 200


def load_checkpoint(conf):
    path = f'/home/group01/checkpoints_W5_pytorch/{conf.experiment}.pth'
    if not os.path.exists(path):
        print("[ERROR] Checkpoint for this experiment does NOT exist. Aborting...")
        raise Exception()

    #model = available_models[conf.model]()
    checkpoint = torch.load(path)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])
    return model


def test(model, conf, device):
    print("\n---------> TEST EXECUTION <---------")

    # ----- DATASET LOADING ------
    test_dataset = get_testing_dataset(conf.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=NUM_WORKERS)

    criterion = nn.CrossEntropyLoss()

    # ----- TESTING -----
    with torch.no_grad():
        model.eval()
        preds_te, truth_te = [], []
        test_loss = 0.0
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()

            truth_te.append(labels)
            preds_te.append(outputs)

            del inputs
            del labels

    # end of epoch
    test_loss /= len(test_loader)
    test_acc = multi_acc(torch.cat(preds_te, 0), torch.cat(truth_te, 0))
    print(f"[TEST] loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")


    print("---------> END <---------")
    return


def check_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--experiment", # Experiment name of the model
      type=str,
      default="test",
      help="experiment name",
  )

  parser.add_argument(
      "--model", # Model name
      type=str,
      default="random_net",
      help="model name",
  )

  parser.add_argument(
      "--batch_size", # Batch size
      type=int,
      default=16,
      help="batch size",
  )

  parser.add_argument(
      "--img_size", # 
      type=int,
      default=299,
      help="img size",
  )

  parser.add_argument(
      "--verbose", # 
      type=float,
      default=0,
      help="level of verbosity (0, 1)",
  )

  opt = parser.parse_args()
  return opt


if __name__ == "__main__":
    conf = check_args()
    print(conf)

    print(f"Is CUDA available?\n{torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("[WARNING] GPU not found. Running on CPU...")

    # we train the model
    model = load_checkpoint(conf).to(device)

    print(sum(p.numel() for p in model.parameters())
)

    # we test it
    test(model, conf, device)

    print(f"\nEnded: {conf.experiment}")