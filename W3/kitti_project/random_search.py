from configurations import *
from models import obtain_model_cfg
from detectron2.config import get_cfg
import argparse
import os 
from loaders import register_dataset
from detectron2.engine.hooks import PeriodicCheckpointer
from detectron2.engine import DefaultTrainer
from new_heads import *
from detectron2.utils.logger import setup_logger
from random import sample 

def main(parser):
    # Reproducible results
    torch.manual_seed(0)

    cfg = get_cfg()
    # Basic config
    cfg = basic_configuration(cfg)

    # load dataset
    register_dataset()
    print("Dataset loaded...")
    # Modify configuration to use certain model
    cfg = obtain_model_cfg(cfg, parser.model_name)
    if parser.balanced_weights:
        if parser.focal_loss:
            cfg = modify_head_focalloss_class(cfg)
        else:    
            cfg = modify_head_balanced_weight_class(cfg)

    # PARAMS OF THE RANDOM SEARCH
    cfg.SOLVER.BASE_LR = sample((0.001, 0.005), 1)[0]
    cfg.SOLVER.NESTEROV = sample((False, True), 1)[0]
    cfg.SOLVER.MOMENTUM = sample((0.8, 0.9), 1)[0]
    cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = sample((2,5,10), 1)[0]
    cfg.SOLVER.LR_SCHEDULER_NAME, cfg.SOLVER.WARMUP_ITERS = sample((("WarmupMultiStepLR", 1000), 
                                                                    ("WarmupCosineLR", 1000), 
                                                                    ("WarmupMultiStepLR", 0)), 1)[0]
    parser.experiment_name += f"_LR={cfg.SOLVER.BASE_LR}_nesterov={cfg.SOLVER.NESTEROV}_momentum={cfg.SOLVER.MOMENTUM}_LRsch={cfg.SOLVER.LR_SCHEDULER_NAME}_WUiters={cfg.SOLVER.WARMUP_ITERS}"
    if "retinanet" in parser.model_name:
        parser.experiment_name += f"_flgamma={cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA}"
    print(f"Parameters for RANDOM SEARCH:\n{parser.experiment_name}")

    cfg.OUTPUT_DIR = f"/home/group01/W3_random_search/{parser.experiment_name}"
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # SPECIFIC TO KITTI
    cfg.DATASETS.TRAIN = ("ds_train",)
    cfg.DATASETS.VAL = ("ds_val",)
    cfg.DATASETS.TEST = ("ds_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # for new ds

    # TRAINING PARAMS
    cfg.MODEL.BACKBONE.FREEZE_AT = parser.freeze_at # Where to freeze backbone layers to finetune
    cfg.DATALOADER.NUM_WORKERS = parser.workers
    cfg.SOLVER.IMS_PER_BATCH = parser.batch_size
    cfg.SOLVER.MAX_ITER = parser.max_steps
    #cfg.SOLVER.WARMUP_ITERS = 0 #0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    #cfg.SOLVER.BASE_LR = parser.lr
    #cfg.SOLVER.STEPS = [30000, 40000] # quick test

    # CHECKPOINTS AND EVAL
    cfg.TEST.EVAL_PERIOD = 5000
    cfg.SOLVER.CHECKPOINT_PERIOD = 2500
    setup_logger(output=cfg.OUTPUT_DIR, name="train")

    #cfg = prepare_dirs_experiment(cfg, parser.experiment_name, split=0)
    if parser.use_da:
        trainer = DataAugmTrainer(cfg)
    else:
        trainer = CustomDefaultTrainer(cfg)
    
    trainer = register_validation_loss_hook(cfg, trainer)
    trainer.build_hooks()
    trainer.build_writers()

    #checkpointer = PeriodicCheckpointer()
    #trainer.register_hooks([checkpointer])

    # Train...
    trainer.resume_or_load(resume=parser.resume) # if true => load last checkpoint if available (and start training from there)
    print("Trainer loaded and hooks registered...")
    trainer.train()
    print("Training finished...")
    #trainer.test()


def check_args():
    parser = argparse.ArgumentParser()
                                                                                                                                    
    #TODO: option to select one model or another
    parser.add_argument(
        "--experiment_name", 
        type=str,
        default="test",
        help="experiment name",
    )

    parser.add_argument(
        "--resume", 
        action="store_true",
        default=False,
        help="if model will be resumed from last checkpoint if exists",
    )

    parser.add_argument(
        "--use_da", 
        action="store_true",
        default=False,
        help="use data augm",
    )

    parser.add_argument(
        "--balanced_weights", 
        action="store_true",
        default=False,
        help="balance class imbalance",
    )

    parser.add_argument(
        "--focal_loss", 
        action="store_true",
        default=False,
        help="focal loss",
    )

    parser.add_argument(
        "--lr", 
        type=float,
        default=0.001,
        help="learning rate",
    )

    parser.add_argument(
        "--batch_size", 
        type=int,
        default=4,
        help="batch_size",
    )

    parser.add_argument(
        "--workers", 
        type=int,
        default=10,
        help="batch_size",
    )

    parser.add_argument(
        "--model_name", 
        type=str,
        default="resnet_fpn",
        help="model name",
    )

    parser.add_argument(
        "--freeze_at", 
        type=int,
        default=0,
        help="layers to freeze in backbone",
    )

    parser.add_argument(
        "--max_steps", 
        type=int,
        default=15000,
        help="max steps to train",
    )

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    main(parser)