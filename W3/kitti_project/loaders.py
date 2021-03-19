import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.boxes import BoxMode
import cv2
import pandas as pd
import pycocotools.mask as masktools
import numpy as np

classes = ['car', 'person']

kitti_mots_splits = {
    "train": [0, 1, 3, 4, 5, 9, 11, 12, 15],
    "val": [17, 19, 20],
    "test": [2, 6, 7, 8, 10, 13, 14, 16, 18]
}

mots_challenge_splits = {
    "train": [2, 5, 9],
    "val": [11]
}

def get_kitti_mots(subset, dataset_dicts):
    return read_full_ds(subset, dataset_dicts, "/home/group01/mcv/datasets/KITTI-MOTS", "training", "image_02", kitti_mots_splits, False)

def get_mots_challenge(subset, dataset_dicts):
    if subset == "test":
        return dataset_dicts

    return read_full_ds(subset, dataset_dicts, "/home/group01/mcv/datasets/MOTSChallenge", "train", "images", mots_challenge_splits, True)
    
def get_bbox_from_mask(mask):
    locations_y = np.where(np.max(mask,axis=1)==1)
    locations_x = np.where(np.max(mask,axis=0)==1)
    bbox = [np.min(locations_x), np.min(locations_y), np.max(locations_x), np.max(locations_y)]
    return bbox

def read_full_ds(subset, dataset_dicts, root, tr_img_folder, img_name_folder, split_seqs, is_kitti_mots, is_coco=False):
    img_id = -1
    for sequence in split_seqs[subset]:
        if not is_kitti_mots:
            labels_path = os.path.join(root, "instances_txt", str(sequence).zfill(4)+".txt")
            extension = ".png"
        else:
            labels_path = os.path.join(root, tr_img_folder, "instances_txt", str(sequence).zfill(4)+".txt")
            extension = ".jpg"
        anno_seq = pd.read_csv(labels_path, sep=" ", header=None)

        last_time_frame = -1
        record = {"annotations": []}
        tot = len(anno_seq)
        """ if is_coco:
            tot = 1000 """

        for i in range(tot):
            time_frame, id_, class_id, h, w, rle = anno_seq.iloc[i, :]

            if time_frame != last_time_frame:
                last_time_frame = time_frame
                img_id += 1                
                if len(record["annotations"]) > 0:
                    dataset_dicts.append(record)
                    record = {"annotations": []}
                
                record["file_name"] = os.path.join(root, tr_img_folder, img_name_folder, str(sequence).zfill(4), str(time_frame).zfill(6)+extension)
                height, width = h, w #cv2.imread(record["file_name"]).shape[:2]
                record["image_id"] = img_id
                record["height"] = height
                record["width"] = width

                mask = masktools.decode({
                    "size": (height, width),
                    "counts": rle.encode("utf-8")
                })
                bbox = get_bbox_from_mask(mask)

                label = "bg"
                if class_id == 1:
                    label = "car"
                elif class_id == 2:
                    label = "person"
                
                if label != "bg":
                    if not is_coco:
                        class_id = class_id-1
                    else:
                        if class_id == 1:
                            class_id = 2
                        elif class_id == 2:
                            class_id = 0

                    record["annotations"].append({
                        "category_id": class_id, #-1,
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "label": label
                    })

            else:
                record["file_name"] = os.path.join(root, tr_img_folder, img_name_folder, str(sequence).zfill(4), str(time_frame).zfill(6)+extension)
                height, width = h, w

                mask = masktools.decode({
                    "size": (height, width),
                    "counts": rle.encode("utf-8")
                })
                bbox = get_bbox_from_mask(mask)

                label = "bg"
                if class_id == 1:
                    label = "car"
                elif class_id == 2:
                    label = "person"
                
                if label != "bg":
                    if not is_coco:
                        class_id = class_id-1
                    else:
                        if class_id == 1:
                            class_id = 2
                        elif class_id == 2:
                            class_id = 0
                            
                    record["annotations"].append({
                        "category_id": class_id, 
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "label": label
                    })

        if len(record["annotations"]) > 0:
            dataset_dicts.append(record)

    return dataset_dicts

def get_dataset(subset):
    dataset_dicts = []
    dataset_dicts = get_kitti_mots(subset, dataset_dicts)
    dataset_dicts = get_mots_challenge(subset, dataset_dicts)
    print("dataset length:",len(dataset_dicts))
    return dataset_dicts


def get_cocofriendly_test_dataset(subset):
    dataset_dicts = []
    return read_full_ds(subset, dataset_dicts, "/home/group01/mcv/datasets/KITTI-MOTS", "training", "image_02", kitti_mots_splits, False, is_coco=True)

def register_test_dataset_coco():
    for subset in ["test"]:
        DatasetCatalog.register(f"ds_{subset}", lambda subset=subset: get_cocofriendly_test_dataset(subset))
        print(f"Successfully registered 'ds_{subset}'!")
        MetadataCatalog.get(f"ds_{subset}").thing_classes = ["person", "bicycle", "car"]

def register_test_dataset():
    for subset in ["val", "test"]:
        DatasetCatalog.register(f"ds_{subset}", lambda subset=subset: get_dataset(subset))
        print(f"Successfully registered 'ds_{subset}'!")
        MetadataCatalog.get(f"ds_{subset}").thing_classes = classes

def register_dataset():
    for subset in ["train", "val", "test"]:
        DatasetCatalog.register(f"ds_{subset}", lambda subset=subset: get_dataset(subset))
        print(f"Successfully registered 'ds_{subset}'!")
        MetadataCatalog.get(f"ds_{subset}").thing_classes = classes

if __name__ == "__main__":
    get_dataset("test")