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
import pandas as pd

def main(parser, exp_name, EXP_ID, output_name):
    # Reproducible results
    torch.manual_seed(0)

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

    # PARAMS OF THE RANDOM SEARCH
    cfg.SOLVER.BASE_LR = sample((0.0001, 0.0005, 0.001), 1)[0]
    cfg.SOLVER.NESTEROV = sample((False, True), 1)[0]
    cfg.SOLVER.MOMENTUM = sample((0.8, 0.9), 1)[0]
    cfg.SOLVER.LR_SCHEDULER_NAME, cfg.SOLVER.WARMUP_ITERS = sample((("WarmupMultiStepLR", 1000), 
                                                                    ("WarmupCosineLR", 1000), 
                                                                    ("WarmupMultiStepLR", 0)), 1)[0]
    # Mask RCNN specific parameters
    BG_IOU_THRESHOLD = sample((0.2,0.3,0.4,0.5), 1)[0]
    FG_IOU_THRESHOLD = sample((0.5,0.6,0.7,0.8), 1)[0]
    cfg.MODEL.RPN.IOU_THRESHOLDS = [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = sample((64,128,256,512), 1)[0] #Number of top scoring precomputed proposals to keep for training
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = sample((1000,2000,3000,4000), 1)[0] #Number of top scoring RPN proposals to keep
    #cfg.MODEL.ANCHOR_GENERATOR.NAME = sample(("DefaultAnchorGenerator", "RotatedAnchorGenerator"), 1)[0]
    random_search_params = [EXP_ID, cfg.SOLVER.BASE_LR, cfg.SOLVER.NESTEROV, cfg.SOLVER.MOMENTUM, cfg.SOLVER.LR_SCHEDULER_NAME, cfg.SOLVER.WARMUP_ITERS, 
                            BG_IOU_THRESHOLD, FG_IOU_THRESHOLD, 
                            cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.ANCHOR_GENERATOR.NAME]

    print(f"Parameters for RANDOM SEARCH {exp_name} - {EXP_ID}:\n{random_search_params}")

    cfg.OUTPUT_DIR = f"/home/group01/W4_random_search/{exp_name}/{EXP_ID}"
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    cfg.INPUT.MASK_FORMAT = "bitmask"
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

    # CHECKPOINTS AND EVAL
    #cfg.TEST.EVAL_PERIOD = 0
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

    # Train...
    trainer.resume_or_load(resume=parser.resume) # if true => load last checkpoint if available (and start training from there)
    trainer.train()
    print("Training finished... Testing now!")

    # Testing
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_val_model.pth")
    subset="val"
    cfg.DATASETS.TRAIN = ("ds_"+subset,)
    cfg.DATASETS.TEST = ("ds_"+subset, )
    trainer.resume_or_load(resume=False) # if true => load last checkpoint if available (and start training from there)
    testing = COCOEvaluator("ds_"+subset, tasks=["bbox", "segm"], distributed=True, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_ds_"+subset))
    evals = [testing, ] #[testing, validation, training]
    results = DefaultTrainer.test(cfg, trainer.model, evaluators=evals)
    results = list(results.items())
    bbox, segm = results[0][1], results[1][1]
    #print(bbox)
    #print(segm)
    print("Testing finished...")
    
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AP-car", "AP-person"]
    bboxes_keys = ["bbox_"+k for k in keys]
    segm_keys = ["segm_"+k for k in keys]
    columns = ["ID", "lr", "nesterov", "momentum", "lr_scheduler", "warmup", "bg_iou", "fg_iou", "batch_size_per_image", "post_nms", "anchor_generator"] + bboxes_keys + segm_keys
    random_search_params += [bbox[key] for key in keys]
    random_search_params += [segm[key] for key in keys]
    if not os.path.exists(output_name):
        pd.DataFrame(data=[random_search_params, ], columns=columns).to_csv(output_name, index=False)
    else:
        df = pd.read_csv(output_name)
        df.loc[len(df.index)] = random_search_params
        df.to_csv(output_name, index=False)
    return


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

    parser.add_argument(
        "--motschallenge", 
        type=int,
        default=1,
        help="include motschallenge. 0 = no, 1 = yes",
    )

    parser.add_argument(
        "--rs_iters", 
        type=int,
        default=20,
        help="random search iterations",
    )
    # motschallenge

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    # load dataset
    register_dataset(use_mots_challenge=parser.motschallenge==1)
    exp_name = parser.experiment_name
    print("Dataset loaded...")
    for i in range(parser.rs_iters):
        main(parser, exp_name, i, f"rs_{exp_name}.csv")