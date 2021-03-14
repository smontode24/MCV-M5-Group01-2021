from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.data import transforms as T
import torch
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator
import os 

class CustomDefaultTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None
        It is not implemented by default.
        """
        return COCOEvaluator("kitti_val", tasks=["bbox",], distributed=True, output_dir=cfg.OUTPUT_DIR)


class DataAugmTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None
        It is not implemented by default.
        """
        return COCOEvaluator("kitti_val", tasks=["bbox",], distributed=True, output_dir=cfg.OUTPUT_DIR)

    # Data augmentation
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg,
            mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                T.RandomBrightness(0.5, 2),
                T.RandomContrast(0.5, 2),
                T.RandomSaturation(0.5, 2),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomApply(T.RandomExtent([0.1,0.1], [0.1, 0.1]), prob=0.5),
            ]))


class ValidationLoss(HookBase):
    # Hook for validation loss logging
    def __init__(self, cfg, val_steps_in_log=50, val_frequency=200):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))
        self.step_num = 0
        self.val_frequency = val_frequency # Log every val_frequency steps
        self.val_steps_in_log = val_steps_in_log # How many validation steps to do each time
        
    def after_step(self):
        self.step_num += 1
        
        if self.step_num % self.val_frequency == 0:
            with torch.no_grad():
                losses = 0
                for i in range(self.val_steps_in_log):
                    data = next(self._loader)

                    loss_dict = self.trainer.model(data)
                    
                    losses += sum(loss_dict.values())
                    
                    assert torch.isfinite(losses).all(), loss_dict
                losses = losses/self.val_steps_in_log
                loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                  comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                  **loss_dict_reduced)

            self.step_num = 0
        

def basic_configuration(cfg):
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    return cfg

def register_validation_loss_hook(cfg, trainer):
    val_loss = ValidationLoss(cfg)  
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    return trainer

# NOT USED
def prepare_dirs_experiment(cfg, experiment_name, split):
    experiment_path = os.path.join(cfg.OUTPUT_DIR, experiment_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(experiment_path, exist_ok=True)
    setup_logger(output=experiment_path, name=f"cv_split_{split}")

    cfg["OUTPUT_DIR"] = experiment_path
    return cfg