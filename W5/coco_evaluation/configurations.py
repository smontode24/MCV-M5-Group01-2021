from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.data import transforms as T
import torch
import copy
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator
from detectron2.config import configurable
from typing import List, Optional, Union
import numpy as np 
import os 
from detectron2.config import CfgNode
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import detection_utils as utils
import cv2
from pycocotools import mask as masktools
import json
from scipy.ndimage.interpolation import shift
from numpy.random import choice
from style_transfer import StyleTransferer

model_style = StyleTransferer("/home/group01/bg_taskD/style1/eleph_skin.png") # StyleTransferer("/home/group01/tmp/cuadro.png") #  

def identity_function(image, dataset_dict, coco_full_dict_info, classes, debug_only=True):
    return image

def obtain_mask(ann, h, w):
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = masktools.frPyObjects(segm, h, w)
        rle = masktools.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = masktools.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']

    m = masktools.decode(rle) == 1
    return m

def new_object(image, dataset_dict, coco_full_dict_info, classes, debug_only=False):
    anno = None
    for a in dataset_dict["annotations"]:
        if a["category_id"] not in classes: # we replace one object by another (not the class we target)
            anno = a
            break

    if anno is None:
        return image

    x, y, w, h = anno["bbox"]

    # snowboard
    #to_add = cv2.imread("/home/group01/data/val2017/000000563349.jpg")
    #bbox = [359, 300, 31, 140]
    to_add = cv2.imread("/home/group01/data/val2017/000000074860.jpg")
    bbox = [115, 304, 129, 33]
    m_x = int(0.4*bbox[2])
    m_y = int(0.4*bbox[3])
    bbox = [bbox[0] - m_x, bbox[1] - m_y, int(bbox[2]*1.8), int(bbox[3]*1.8)]
    new_img = image.copy()
    x, y, w, h = int(x), int(y), int(w), int(h)
    new_img[y:y+bbox[3], x:x+bbox[2]] = to_add[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    #new_img[max(0, y):min(to_add.shape[0], y+h), max(0, x):min(to_add.shape[1], x+w)] = to_add[max(0, y):min(to_add.shape[0], y+h), max(0, x):min(to_add.shape[1], x+w)]
    
    #dataset_dict["annotations"].append({
    #            "category_id": anno["category_id"],
    #            "bbox": displaced_bbox,
    #            "segmentation": shifted_mask, #pycocotools.mask.encode(np.asarray(mask.astype("uint8"), order="F")),
    #            "bbox_mode": anno["bbox_mode"]
    #        })
    print(f"Added: {bbox}")
    return new_img


def move_existent(image, dataset_dict, coco_full_dict_info, classes, debug_only=False):
    anno = None
    for a in dataset_dict["annotations"]:
        if a["category_id"] in classes:
            anno = a
            break

    if anno is None:
        return image

    mask = obtain_mask(anno, image.shape[0], image.shape[1])
    x, y, w, h = anno["bbox"]
    d = (-1, 1)

    new_img = np.copy(image)
    displacement = (0.5 * np.array((choice(d)*h, choice(d)*w))).astype(int)
    displaced_bbox = (x + displacement[1], y + displacement[0], w, h)
    shifted_mask = shift(mask, displacement, order=0, cval=False)
    unshifted_mask = shift(shifted_mask, -displacement, order=0, cval=False)
    new_img[shifted_mask] = image[unshifted_mask]
    
    dataset_dict["annotations"].append({
                "category_id": anno["category_id"],
                "bbox": displaced_bbox,
                "segmentation": shifted_mask, #pycocotools.mask.encode(np.asarray(mask.astype("uint8"), order="F")),
                "bbox_mode": anno["bbox_mode"]
            })
    return new_img

def crop_noise_function(image, dataset_dict, coco_full_dict_info, classes, debug_only=False):
    bboxes = []
    for anno in dataset_dict["annotations"]:
        if anno["category_id"] in classes:
            bboxes.append(anno["bbox"])
    
    if len(bboxes) > 0:
        new_img = np.random.randint(0, 255, image.shape).astype(np.uint8)
        for bbox in bboxes:
            new_img[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] = \
                                image[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
        image = new_img

    return image

def crop_mask_function(image, dataset_dict, coco_full_dict_info, classes, debug_only=True):
    #print("using mask crop")
    
    masks = []
    for anno in dataset_dict["annotations"]:
        if anno["category_id"] in classes:
            masks.append(obtain_mask(anno, image.shape[0], image.shape[1]))
    
    if len(masks) > 0:
        new_img = np.zeros_like(image)
        for mask in masks:
            new_img[mask] = image[mask]
        
        image = new_img
    
    return image

def crop_mask_noise_function(image, dataset_dict, coco_full_dict_info, classes, debug_only=False):
    masks = []
    for anno in dataset_dict["annotations"]:
        if anno["category_id"] in classes:
            masks.append(obtain_mask(anno, image.shape[0], image.shape[1]))
    
    if len(masks) > 0:
        new_img = np.random.randint(0, 255, image.shape).astype(np.uint8)
        for mask in masks:
            new_img[mask] = image[mask]
        
        image = new_img
    
    return image

def crop_mask_bg_random(image, dataset_dict, coco_full_dict_info, classes, debug_only=False, choice=1):
    bg_options = ["/home/group01/bg_taskD/water.jpeg", "/home/group01/bg_taskD/sky.jpeg", "/home/group01/bg_taskD/volcano.jpeg"]
    bg = bg_options[choice]
    bg = cv2.imread(bg)[:,:,[2,1,0]]

    masks = []
    for anno in dataset_dict["annotations"]:
        if anno["category_id"] in classes:
            masks.append(obtain_mask(anno, image.shape[0], image.shape[1]))
    
    if len(masks) > 0:
        if image.shape[0] < bg.shape[0] and image.shape[1] < bg.shape[1]:
            new_img = bg[:image.shape[0], :image.shape[1]]
        else:
            new_img = cv2.resize(bg, (image.shape[0], image.shape[1]))
        
        for mask in masks:
            new_img[mask] = image[mask]
        image = new_img
    
    return image

def transfer_style_whole(image, dataset_dict, coco_full_dict_info, classes, debug_only=False, choice=1):
    img = image.copy()
    try:
        masks = []
        for anno in dataset_dict["annotations"]:
            masks.append(obtain_mask(anno, image.shape[0], image.shape[1]))
        
        im_shape = image.shape
        image = model_style(image)
        image = cv2.resize(image, (im_shape[1], im_shape[0]))
        
        if len(masks) > 0:
            t0 = time()

        return image
    except:
        return img

def transfer_style_part(image, dataset_dict, coco_full_dict_info, classes, debug_only=False, choice=1):
    img = image.copy()
    try:    
        masks = []
        for anno in dataset_dict["annotations"]:
            masks.append(obtain_mask(anno, image.shape[0], image.shape[1]))
        
        im_shape = image.shape
        new_image = model_style(image)
        new_image = cv2.resize(new_image, (im_shape[1], im_shape[0]))
        
        if len(masks) > 0:
            from time import time
            t0 = time()
            for mask in masks:
                image[mask] = new_image[mask]
        
        return image
    except: 
        return img


def basic_configuration(cfg):
    cfg.USE_DA = False # custom
    cfg.VAL_PERIOD = 0 # custom
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    return cfg

dict_map_f = {
    "none": identity_function,
    "move_existent": move_existent,
    "crop_box": crop_function,
    "crop_mask": crop_mask_function,
    "crop_box_noise": crop_noise_function,
    "crop_mask_noise": crop_mask_noise_function,
    "crop_mask_bg_random": crop_mask_bg_random,
    "style_transfer_whole": transfer_style_whole,
    "style_transfer_part": transfer_style_part
}

class CustomDefaultTrainer(DefaultTrainer):
    def __init__(self, cfg, coco_dict_info, classes_to_modify, map_f_name="none"):
        super().__init__(cfg)
        # Full coco dictionary 
        self.coco_dict_info = coco_dict_info 
        # Classes that will be used in the modification of the data
        # e.g., if classes_to_modify=[0] with the crop function -> Crops classes of classes_to_modify in the image
        self.classes_to_modify = classes_to_modify
        self.f_aug = dict_map_f[map_f_name]

        cfg["classes_to_modify"] = self.classes_to_modify
        cfg["map_f_name"] = map_f_name
        cfg["coco_dict_info"] = self.coco_dict_info

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        print("before classes modify:", cfg["classes_to_modify"])
        return build_detection_test_loader(cfg, dataset_name, mapper=MapperClassCrop(cfg, False, \
                        classes_to_modify=cfg["classes_to_modify"], coco_info=cfg["coco_dict_info"], f_aug=dict_map_f[cfg["map_f_name"]]))

class MapperClassCrop:
    @configurable
    def __init__(
        self,
        is_train: bool,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        coco_info: dict = {"test": None},
        f_aug = None,
        classes_to_modify= None,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.coco_info = coco_info
        self.classes_to_modify = classes_to_modify
        self.recompute_boxes = recompute_boxes
        self.instance_mask_format = instance_mask_format
        self.is_train = is_train
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.f_aug = f_aug

        json_anno_path = "/home/group01/data/val2017_fix.anno"
        with open(json_anno_path, "r") as f:
            data = f.read()
        json_anno = json.loads(data)
        self.coco_info = {key: json_anno["anno_info"][key] for key in list(json_anno["anno_info"].keys())}

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, *args, **kwargs):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "classes_to_modify": cfg["classes_to_modify"], 
            "coco_info": cfg["coco_dict_info"], 
            "f_aug": cfg["map_f_name"]
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"])

        image = dict_map_f[self.f_aug](image, dataset_dict, self.coco_info, self.classes_to_modify)

        aug_input = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        
        try:
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        except:
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(np.zeros((1920, 1080, 3)).transpose(2,0,1))).float()

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict