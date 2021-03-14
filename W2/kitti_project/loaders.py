import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.boxes import BoxMode
import cv2

classes = ['car', 'van', 'truck', 'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc']

def get_class_id(cls):
    return classes.index(cls.lower())

def parse_kitti_object(line):
    line = line.replace("\n", "").split(" ")
    label = line[0]
    if label.lower() == "dontcare":
        return None
    truncated, occluded, alpha = float(line[1]), int(line[2]), float(line[3])
    category_id = get_class_id(label)
    bbox_mode = BoxMode.XYXY_ABS
    bbox = [float(v) for v in line[4:8]]
    dim = [float(v) for v in line[8:11]]
    loc = [float(v) for v in line[11:14]]
    rotation_y = float(line[14])
    return {
        "category_id": category_id,
        "bbox": bbox,
        "bbox_mode": bbox_mode,
        "label": label,
        "truncated": truncated, "occluded": occluded, "alpha": alpha,
        "dimensions": dim, "location": loc, "rotation_y": rotation_y, 
    }

def get_kitti_dicts(subset):
    root = "/home/group01/mcv/datasets/KITTI"
    txt_file = os.path.join(root, f"{subset}_kitti.txt")
    with open(txt_file, "r") as f:
        lines = f.readlines()

    dataset_dicts = []
    for idx, annotation_filename in enumerate(lines):
        annotation_filename = annotation_filename.replace("\n", "")
        record = {}
        
        filename = os.path.join(root, "data_object_image_2/training/image_2", annotation_filename.replace("txt", "png"))
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annotation_filepath = os.path.join(root, "training/label_2", annotation_filename)
        with open(annotation_filepath, "r") as f:
            objects = f.readlines()

        objs = []
        for obj_line in objects:
            obj = parse_kitti_object(obj_line)
            if obj is not None:
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    

    return dataset_dicts


def register_kitti_dataset():
    for subset in ["train", "val", "test"]:
        DatasetCatalog.register(f"kitti_{subset}", lambda subset=subset: get_kitti_dicts(subset))
        print(f"Successfully registered 'kitti_{subset}'!")
        MetadataCatalog.get(f"kitti_{subset}").thing_classes = classes