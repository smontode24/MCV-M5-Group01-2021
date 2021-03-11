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


def save_checkpoint(model, optimizer, conf, replace=True):
    checkpoint = {'model': available_models[conf.model](),
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict()}

    path = f'/home/group01/checkpoints_W5_pytorch/{conf.experiment}.pth'
    if not replace and os.path.exists(path):
        print("[ERROR] Checkpoint for this experiment already exists. Aborting...")
        raise Exception()

    if conf.verbose == 1:
        print(f"[SAVED] Checkpoint saved at '{path}'")
    torch.save(checkpoint, path)


def train(conf, device):
    print("\n---------> TRAIN EXECUTION <---------")

    model = available_models[conf.model]().to(device)

    # ----- DATASET LOADING ------
    tr_dataset, val_dataset = get_training_datasets(conf.img_size)
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=NUM_WORKERS)

    # ----- LOSS AND OPTIMIZER ------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=conf.initial_learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                lr_lambda=lambda step: polynomial_decay(step, init_learning_rate=conf.initial_learning_rate, total_epochs=NUM_EPOCHS))

    # ----- TRAINING -----
    losses, val_accs, tr_accs = [], [], []
    patience, best_val_loss = 0, float('inf')
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        model.train()
        tr_loss = 0.0
        preds_tr, truth_tr = [], []
        for i, data in enumerate(tr_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            tr_loss += loss.item()
            if conf.verbose == 2 and i % PRINT_INTERVAL == 0:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.6f' %
                    (epoch + 1, i + 1, tr_loss))

            truth_tr.append(labels)
            preds_tr.append(outputs)

            del inputs
            del labels

        # end of the epoch training
        # lr update
        scheduler.step()

        # VALIDATION STEP
        with torch.no_grad():
            model.eval()
            preds_val, truth_val = [], []
            val_loss = 0.0
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

                truth_val.append(labels)
                preds_val.append(outputs)

                del inputs
                del labels

        # end of epoch
        val_loss /= len(val_loader)
        tr_loss /= len(tr_loader)
        val_acc = multi_acc(torch.cat(preds_val, 0), torch.cat(truth_val, 0))
        tr_acc = multi_acc(torch.cat(preds_tr, 0), torch.cat(truth_tr, 0))
        
        losses.append(tr_loss)
        val_accs.append(val_acc)
        tr_accs.append(tr_acc)

        # patience update
        if best_val_loss <= val_loss:
            patience += 1
        else: # restart patience counter because loss was decreased
            patience = 0
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, conf)

        print(f"[{epoch+1:03d}/{patience:02d}] loss: {tr_loss:.4f}, accuracy: {tr_acc:.4f}, val_loss: {val_loss:.4f} ({best_val_loss:.4f}), val_accuracy: {val_acc:.4f}")

        # early stopping check
        if patience >= EA_PATIENCE:
            print(f"[END] Early stopping applied after {patience} epochs without decreasing validation loss.")
            break


    print("---------> END <---------")

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
      "--early_stop", # 
      type=int,
      default=1,
      help="early stop",
  )

  parser.add_argument(
      "--tr_da", # 
      type=int,
      default=1,
      help="data augmentaiton",
  )

  parser.add_argument(
      "--img_size", # 
      type=int,
      default=299,
      help="img size",
  )

  parser.add_argument(
      "--initial_learning_rate", # 
      type=float,
      default=0.001,
      help="initial learning rate",
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
    model = train(conf, device)

    # we test it
    test(model, conf, device)

    print(f"\nEnded: {conf.experiment}")