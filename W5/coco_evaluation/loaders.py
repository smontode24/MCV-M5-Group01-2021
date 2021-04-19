import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.boxes import BoxMode
import cv2
import pandas as pd
import pycocotools.mask as masktools
import pycocotools
import numpy as np
from imantics import Polygons, Mask
import json

classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "hair brush", "toothbrush"]
classes_80 = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

class_to_idx = {classes_80[i]:i for i in range(len(classes_80))}

def get_bbox_from_mask(mask):
    locations_y = np.where(np.max(mask,axis=1)==1)
    locations_x = np.where(np.max(mask,axis=0)==1)
    bbox = [np.min(locations_x), np.min(locations_y), np.max(locations_x), np.max(locations_y)]
    return bbox

def segm_to_cocomask(labelMap):
    labelMask = np.expand_dims(labelMap, axis=2)
    labelMask = labelMask.astype('uint8')
    labelMask = np.asfortranarray(labelMask)
    Rs = pycocotools.mask.encode(labelMask)
    return Rs

def read_full_ds():
    dataset_dicts = []

    name_dst = "/home/group01/data/val2017_test1"
    imgs_path = "/home/group01/data/val2017"
    json_anno_path = "/home/group01/data/val2017_fix.anno"
    with open(json_anno_path, "r") as f:
        data = f.read()
    json_anno = json.loads(data)
    reduced_json_anno = {key: json_anno["anno_info"][key] for key in list(json_anno["anno_info"].keys())} # [:500]

    from pycocotools.coco import COCO
    coco_api = COCO(json_anno_path)
    cat_ids = sorted(coco_api.getCatIds())
    id_map = {v: i for i, v in enumerate(cat_ids)}

    for img_id_detectron, img_id in enumerate(reduced_json_anno.keys()):
    
        img_path = os.path.join(imgs_path, str(img_id).zfill(12)+".jpg")
        #img = cv2.imread(img_path)
        #img_path = os.path.join(imgs_path, str(name_dst).zfill(12)+".jpg")
        
        record = {"annotations": []}
        record["file_name"] = img_path
        record["image_id"] = img_id_detectron
        record["height"] = json_anno["img_info"][img_id]["height"]
        record["width"] = json_anno["img_info"][img_id]["width"]

        for annotation in json_anno["anno_info"][img_id]:
            
            #px, py = annotation["segmentation"][0], annotation["segmentation"][1]
            #poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            #poly = [p for x in poly for p in x]
            segm = annotation["segmentation"]
            if isinstance(segm, dict):
                if isinstance(segm["counts"], list):
                    # convert to compressed RLE
                    segm = masktools.frPyObjects(segm, *segm["size"])
            else:
                # filter out invalid polygons (< 3 points)
                segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                if len(segm) == 0:
                    num_instances_without_valid_segmentation += 1
                    continue  # ignore this instance
            bbox = annotation["bbox"]

            record["annotations"].append({
                "category_id": id_map[annotation["category_id"]],
                "bbox": bbox,
                "segmentation": segm, 
                "bbox_mode": BoxMode.XYWH_ABS
            })

        dataset_dicts.append(record)

    return dataset_dicts

def register_test_dataset_coco():
    for subset in ["test"]:
        DatasetCatalog.register(f"ds_{subset}", lambda subset=subset: read_full_ds())
        print(f"Successfully registered 'ds_{subset}'!")
        MetadataCatalog.get(f"ds_{subset}").thing_classes = classes_80

if __name__ == "__main__":
    get_dataset("test")

