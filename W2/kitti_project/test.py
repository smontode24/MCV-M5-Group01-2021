from configurations import *
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

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
    cfg = obtain_model_cfg(cfg, parser.model_name)
    cfg = prepare_dirs_experiment(cfg, parser.experiment_name)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, cfg.experiment_name, cfg.checkpoint_name)  # path to the model we just trained

    trainer = DataAugmTrainer(cfg)

    # Test
    evaluator = COCOEvaluator("", ("bbox", "segm"), False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "")
    res = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(res)

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
        "--model_name", 
        type=str,
        default="resnet_fpn",
        help="model name",
    )

    parser.add_argument(
        "--checkpoint_name", 
        type=str,
        default="",
        help="name of checkpoint",
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