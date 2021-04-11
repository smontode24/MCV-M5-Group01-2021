from configurations import *
from models import obtain_model_cfg
from detectron2.config import get_cfg
import argparse
import os 
from loaders import register_test_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation import COCOEvaluator
from new_heads import *

def main(parser):
    # Reproducible results
    torch.manual_seed(0)

    # load dataset
    subset = "val" if parser.val else "test"
    register_test_dataset()

    print("Dataset loaded...")

    cfg = get_cfg()
    
    # Basic config
    cfg = basic_configuration(cfg)
    
    # Modify configuration to use certain model
    cfg = obtain_model_cfg(cfg, parser.model_name)
    
    if parser.balanced_weights:
        if parser.focal_loss:
            cfg = modify_head_focalloss_class(cfg)
        else:    
            cfg = modify_head_balanced_weight_class(cfg)
    
    cfg.OUTPUT_DIR = f"/home/group01/W3_random_search/{parser.experiment_name}"
    print("analyzing:", cfg.OUTPUT_DIR)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_val_model.pth")

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
    trainer = CustomDefaultTrainer(cfg)
    
    #trainer = register_validation_loss_hook(cfg, trainer)
    #trainer.build_hooks()

    # Load...
    trainer.resume_or_load(resume=True) # if true => load last checkpoint if available (and start training from there)

    print("Checkpoint loaded...")
    #training = COCOEvaluator("kitti_train", tasks=["bbox",], distributed=True, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_kitti_train"))
    #validation = COCOEvaluator("kitti_val", tasks=["bbox",], distributed=True, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_kitti_val"))
    testing = COCOEvaluator("ds_"+subset, tasks=["bbox",], distributed=True, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_ds_"+subset))
    evals = [testing, ] #[testing, validation, training]
    DefaultTrainer.test(cfg, trainer.model, evaluators=evals)
    print("Testing finished...")
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
        "--freeze_at", 
        type=int,
        default=0,
        help="layers to freeze in backbone",
    )

    parser.add_argument(
        "--val", 
        action="store_true",
        default=False,
        help="evaluate only validation",
    )

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    main(parser)
