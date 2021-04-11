import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.boxes import BoxMode
import cv2
import pandas as pd
import pycocotools.mask as masktools
import pycocotools
import numpy as np
from imantics import Polygons, Mask

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

import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    return area, segmentations

def segm_to_cocomask(labelMap):
    labelMask = np.expand_dims(labelMap, axis=2)
    labelMask = labelMask.astype('uint8')
    labelMask = np.asfortranarray(labelMask)
    Rs = pycocotools.mask.encode(labelMask)
    return Rs

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
                        "category_id": class_id,
                        "bbox": bbox,
                        "segmentation": pycocotools.mask.encode(np.asarray(mask.astype("uint8"), order="F")),
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
                        "segmentation": pycocotools.mask.encode(np.asarray(mask.astype("uint8"), order="F")),
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "label": label
                    })

        if len(record["annotations"]) > 0:
            dataset_dicts.append(record)

    return dataset_dicts

def get_dataset(subset, use_mots_challenge=True):
    dataset_dicts = []
    dataset_dicts = get_kitti_mots(subset, dataset_dicts)
    if use_mots_challenge:
        dataset_dicts = get_mots_challenge(subset, dataset_dicts)
    print("dataset length:", len(dataset_dicts))
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

def register_dataset(use_mots_challenge=True):
    for subset in ["train", "val", "test"]:
        DatasetCatalog.register(f"ds_{subset}", lambda subset=subset: get_dataset(subset, use_mots_challenge))
        print(f"Successfully registered 'ds_{subset}'!")
        MetadataCatalog.get(f"ds_{subset}").thing_classes = classes

if __name__ == "__main__":
    get_dataset("test")
