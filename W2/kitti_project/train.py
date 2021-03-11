from configurations import *
from detectron2.config import get_cfg

def main(parser):
    cfg = get_cfg()

    # Basic config
    cfg = basic_configuration(cfg)

    # Add by parameter learning rate, etc
    cfg.MODEL.BACKBONE.FREEZE_AT = parser.freeze_at # Where to freeze backbone layers to finetune
    cfg.DATALOADER.NUM_WORKERS = parser.workers #
    cfg.SOLVER.IMS_PER_BATCH = parser.batch_size
    cfg.SOLVER.BASE_LR = cfg.lr  # pick a good LR
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Modify configuration to use certain model
    cfg = modify_config_resnet_fpn(cfg)
    cfg = prepare_dirs_experiment(cfg, parser.experiment_name)

    trainer = DataAugmTrainer(cfg)
    trainer = register_validation_loss_hook(trainer)

    # Train...
    trainer.resume_or_load(resume=False)
    trainer.train()

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
        default=2,
        help="batch_size",
    )

    parser.add_argument(
        "--freeze_at", 
        type=int,
        default=0,
        help="layers to freeze in backbone",
    )

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    main(parser)