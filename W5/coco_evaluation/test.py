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
from loaders import *

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
    cfg.OUTPUT_DIR = f"/home/group01/W5_experiments/{parser.experiment_name}"
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    cfg.DATALOADER.NUM_WORKERS = 0 #parser.workers
    cfg.SOLVER.IMS_PER_BATCH = 16

    setup_logger(output=cfg.OUTPUT_DIR, name="test_"+parser.experiment_name)
    cfg.DATASETS.TRAIN = ("ds_test",)
    cfg.DATASETS.TEST = ("ds_test", )

    if parser.model_name == "coco_r50frcnn":
        tasks = ["bbox"]
    else:
        tasks = ["bbox", "segm"]

    if parser.comp_init:
        trainer = CustomDefaultTrainer(cfg, {}, [0], parser.map_f)
        trainer.resume_or_load(resume=True)
        testing = COCOEvaluator("ds_test", tasks=tasks, distributed=True, output_dir=os.path.join(cfg.OUTPUT_DIR, "test_"+parser.experiment_name))
        evals = [testing, ]
        res = trainer.test(cfg, trainer.model, evaluators=evals)
        
        pd.Series(res["bbox"]).to_csv(os.path.join(cfg.OUTPUT_DIR, "bbox_coco_all.csv"))
        if "segm" in tasks:
            pd.Series(res["segm"]).to_csv(os.path.join(cfg.OUTPUT_DIR, "segm_coco_all.csv"))

    """ results_classes = {"bbox": {}, "segm": {}}
    for cl_n in range(parser.minc, parser.maxc):
        name_class = classes[cl_n]
        data_k = "AP-"+name_class

        trainer = CustomDefaultTrainer(cfg, {}, [cl_n], parser.map_f)
        trainer.resume_or_load(resume=True)

        testing = COCOEvaluator("ds_test", tasks=["bbox","segm"], distributed=True, output_dir=os.path.join(cfg.OUTPUT_DIR, "test_"+parser.experiment_name))
        evals = [testing, ]
        res = trainer.test(cfg, trainer.model, evaluators=evals)

        results_classes["bbox"][data_k] = res["bbox"][data_k]
        if "segm" in tasks:
            results_classes["segm"][data_k] = res["segm"][data_k]

        print("AP [", name_class, "] - Bbox:", results_classes["bbox"][data_k])
        if "segm" in tasks:
            print("Segm:", results_classes["segm"][data_k]) """
        
    """ pd.Series(results_classes["bbox"]).to_csv(os.path.join(cfg.OUTPUT_DIR, parser.map_f+"_bbox_coco_experiment_"+str(parser.minc)+".csv"))
    if "segm" in tasks:
        pd.Series(results_classes["segm"]).to_csv(os.path.join(cfg.OUTPUT_DIR, parser.map_f+"_segm_coco_experiment_"+str(parser.minc)+".csv")) """

    #print("Result:", results_classes)
    print("Testing finished...")

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
        "--minc", 
        type=int,
        help="min class",
    )

    parser.add_argument(
        "--maxc", 
        type=int,
        help="max class",
    )

    parser.add_argument(
        "--comp_init", 
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--map_f", 
        type=str,
        default="crop_box",
    )

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    main(parser)
