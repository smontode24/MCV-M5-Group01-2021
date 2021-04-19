from configurations import *
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

"""
Generates images with certain data augmentations.
"""

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
    cfg = obtain_model_cfg(cfg, parser.model_name)

    #cfg.INPUT.MASK_FORMAT = "bitmask" Enable if encoding image with rle
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    cfg.DATALOADER.NUM_WORKERS = parser.workers
    cfg.SOLVER.IMS_PER_BATCH = 16

    cfg.DATASETS.TRAIN = ("ds_test",)
    cfg.DATASETS.TEST = ("ds_test", )

    # load dataset
    dataset_dicts = read_full_ds()
    ds = sample(dataset_dicts, 10)

    print("Dataset loaded...")

    cfg = get_cfg()
    # Basic config
    cfg = basic_configuration(cfg)
    # Modify configuration to use certain model
    cfg = obtain_model_cfg(cfg, parser.model_name)

    # SPECIFIC TO KITTI
    cfg.DATASETS.TRAIN = ("ds_test",)
    #cfg.DATASETS.VAL = ("kitti_val",)
    cfg.DATASETS.TEST = ("ds_test", )#"kitti_val", "kitti_train")

    # TRAINING PARAMS
    cfg.MODEL.BACKBONE.FREEZE_AT = parser.freeze_at # Where to freeze backbone layers to finetune
    cfg.DATALOADER.NUM_WORKERS = parser.workers
    cfg.SOLVER.IMS_PER_BATCH = parser.batch_size
    #cfg.SOLVER.MAX_ITER = parser.max_steps
    cfg.SOLVER.WARMUP_ITERS = 0 #0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    #cfg.SOLVER.BASE_LR = parser.lr
    #cfg.SOLVER.STEPS = [8000, 13000] # quick test

    # CHECKPOINTS AND EVAL
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    #print("checkpoint path:", cfg.MODEL.WEIGHTS)
    #trainer = CustomDefaultTrainer(cfg)
    
    #trainer = register_validation_loss_hook(cfg, trainer)
    #trainer.build_hooks()

    # Load...
    #trainer.resume_or_load(resume=False) # if true => load last checkpoint if available (and start training from there)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    print("Checkpoint loaded...")
    for d in range(len(ds)):
        im_path = ds[d]["file_name"]

        n_idx = class_to_idx[parser.class_]
        contains_class = False
        for anno in ds[d]["annotations"]:
            if anno["category_id"] == n_idx:
                contains_class = True
                break
        
        contains_class = True
        if contains_class:
            im = cv2.imread(im_path)
            outputs = predictor(im)

            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(f"/home/group01/W5_imgs_taskD/test_{d}_nomod.png", out.get_image()[:, :, ::-1])

    for d in range(len(ds)):
        im_path = ds[d]["file_name"]

        n_idx = class_to_idx[parser.class_]
        contains_class = False
        for anno in ds[d]["annotations"]:
            if anno["category_id"] == n_idx:
                contains_class = True
                break
        
        if contains_class:
            for map_str_f in ["style_transfer_whole", "style_transfer_part"]:
                im = cv2.imread(im_path)
                im = dict_map_f[map_str_f](im, ds[d], None, [n_idx])
                outputs = predictor(im)

                v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imwrite(f"/home/group01/W5_imgs_taskD/{map_str_f}_test_{d}_mod.png", out.get_image()[:, :, ::-1])
        

def check_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=4,
        help="batch_size",
    )

    parser.add_argument(
        "--workers", 
        type=int,
        default=4,
        help="batch_size",
    )

    parser.add_argument(
        "--model_name", 
        type=str,
        default="cityscapes_r50fpn",
        help="model name",
    )

    parser.add_argument(
        "--freeze_at", 
        type=int,
        default=2,
        help="layers to freeze in backbone",
    )

    parser.add_argument(
        "--map_f", 
        type=str,
        default="crop_box",
    )

    parser.add_argument(
        "--class_", 
        type=str,
    )

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    main(parser)
