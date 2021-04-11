from configurations import *
from models import obtain_model_cfg
from detectron2.config import get_cfg
import argparse
import os 
from loaders import register_test_dataset, get_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation import COCOEvaluator
from new_heads import *
from random import sample
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import cv2
import torch

def main(parser):
    # Reproducible results
    torch.manual_seed(0)

    # load dataset
    subset = "test"
    register_test_dataset()
    dataset_dicts = get_dataset("test")
    ds = sample(dataset_dicts, 10)

    print("Dataset loaded...")

    cfg = get_cfg()
    # Basic config
    cfg = basic_configuration(cfg)
    # Modify configuration to use certain model
    cfg = obtain_model_cfg(cfg, parser.model_name)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_val_model.pth")

    # SPECIFIC TO KITTI
    cfg.DATASETS.TRAIN = ("ds_"+subset,)
    #cfg.DATASETS.VAL = ("kitti_val",)
    cfg.DATASETS.TEST = ("ds_"+subset, )#"kitti_val", "kitti_train")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # for kitti

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
    cfg.MODEL.WEIGHTS = "/home/group01/W4_random_search/cityscapes_mots_r50fpn/34/best_val_model.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)

    print("Checkpoint loaded...")
    #training = COCOEvaluator("kitti_train", tasks=["bbox",], distributed=True, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_kitti_train"))
    #validation = COCOEvaluator("kitti_val", tasks=["bbox",], distributed=True, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_kitti_val"))
    for d in range(len(ds)):
        im_path = ds[d]["file_name"]
        print(im_path)
        im = cv2.imread(im_path)
        outputs = predictor(im)

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(f"test_{d}.png", out.get_image()[:, :, ::-1])
        print("Testing finished...")
        #trainer.test()


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

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    main(parser)
