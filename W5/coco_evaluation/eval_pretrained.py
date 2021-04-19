from configurations import *
from models import obtain_model_cfg
from detectron2.config import get_cfg
import argparse
import os 
from loaders import register_test_dataset_coco
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation import COCOEvaluator
from new_heads import *

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
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.OUTPUT_DIR = f"/home/group01/W4_detectron2/pretrained"
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # SPECIFIC TO KITTI
    cfg.DATASETS.TRAIN = ("ds_test",)
    #cfg.DATASETS.VAL = ("kitti_val",)
    cfg.DATASETS.TEST = ("ds_test", )

    # TRAINING PARAMS
    cfg.DATALOADER.NUM_WORKERS = parser.workers
    cfg.SOLVER.IMS_PER_BATCH = parser.batch_size
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    trainer = CustomDefaultTrainer(cfg)

    # Load...
    trainer.resume_or_load(resume=True) # if true => load last checkpoint if available (and start training from there)

    print("Checkpoint loaded...")
    testing = COCOEvaluator("ds_test", tasks=["bbox","segm"], distributed=True, output_dir=os.path.join(cfg.OUTPUT_DIR, "test_output"))
    #testing.params.catIds = [2, 0]
    evals = [testing, ]
    DefaultTrainer.test(cfg, trainer.model, evaluators=evals)
    print("Testing finished...")

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
        default="resnet_fpn",
        help="model name",
    )

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    main(parser)
