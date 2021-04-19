

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

    class_to_find = "snowboard"
    for img_anno in dataset_dicts:
        for anno in img_anno["annotations"]:
            if anno["category_id"] == class_to_idx[class_to_find]:
                print(img_anno["file_name"])
                print(img_anno["height"])
                print(img_anno["width"])
                print(anno["bbox"])

def check_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    main(parser)
