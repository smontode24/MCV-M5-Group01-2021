from configurations_german import *
from models import obtain_model_cfg
from detectron2.config import get_cfg
import argparse
import os 
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation import COCOEvaluator
from random import sample
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import cv2
import torch
from loaders import *
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main(parser):
    # Reproducible results
    torch.manual_seed(0)

    # load dataset
    register_test_dataset_coco()

    print("Dataset loaded...")

    cfg = get_cfg()
    # Basic config
    cfg = basic_configuration(cfg)
    # Modify configuration to use certain model
    #cfg.INPUT.MASK_FORMAT = "bitmask" Enable if encoding image with rle
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    cfg.SOLVER.IMS_PER_BATCH = 16

    cfg.DATASETS.TRAIN = ("ds_test",)
    cfg.DATASETS.TEST = ("ds_test", )

    # load dataset
    dataset_dicts = read_full_ds()

    classes_to_show = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"] 
    classes_to_show = ["skis", "snowboard", "car", "bus", "train", "truck", "boat", "banana", "apple", "sandwich", "orange"]
    classes_to_show = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
    classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    array = np.zeros((len(classes), len(classes)))
    for img_anno in dataset_dicts:
        for anno in img_anno["annotations"]:
            for anno2 in img_anno["annotations"]:
                #if classes_80[anno["category_id"]] not in classes or classes_80[anno2["category_id"]] not in classes:
                #    continue
                array[anno["category_id"], anno2["category_id"]] += 1
    # slice
    slice_cl = [class_to_idx[cla] for cla in classes_to_show]
    array = array[slice_cl, :]
    array = array[:, slice_cl]

    for i in range(len(classes_to_show)):
        array[i][i] = 0
    max_val = np.max(array)
    for i in range(len(classes_to_show)):
        array[i][i] = max_val


    fig, ax = plt.subplots(1,1)

    img = ax.imshow(array, cmap='hot')
    ax.plot((-0.5,len(classes_to_show)-0.5),(-0.5, len(classes_to_show)-0.5), color='black')
    fig.colorbar(img)
    
    ax.set_xticks(list(range(len(classes_to_show))))
    ax.set_xticklabels(classes_to_show)
    ax.set_yticks(list(range(len(classes_to_show))))
    ax.set_yticklabels(classes_to_show)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    fig.tight_layout()
    plt.savefig("coocurrence.png")


def check_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    main(parser)
