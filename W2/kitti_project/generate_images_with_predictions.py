import torch
import numpy as np
import pandas as pd
import argparse
import os
from loaders import register_kitti_dataset, get_kitti_dicts
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import cv2
from tqdm import tqdm
from detectron2.structures import Instances, Boxes

"""
MODEL=faster_balanced_longer
python generate_images_with_predictions.py --output ~/W2_detectron2/${MODEL}/eval_kitti_test/${MODEL}_test_images_0_5 --file ~/W2_detectron2/${MODEL}/eval_kitti_test/instances_predictions.pth --th 0.5

"""

#classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

def main(args):
    register_kitti_dataset()

    data = torch.load(os.path.join(args.file))
    output_path = os.path.join(args.output)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    dataset_dicts = get_kitti_dicts("test")
    metadata = MetadataCatalog.get("kitti_test")

    for d in tqdm(range(len(dataset_dicts))):   
        d = dataset_dicts[d]
        
        bboxes, scores, classes = [], [], []
        for inst in data[d["image_id"]]["instances"]:
            if inst["score"] < args.th:
                continue
            b = inst["bbox"]
            bbox = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            bboxes.append(bbox)
            scores.append(inst["score"])
            classes.append(inst["category_id"])
        {"pred_boxes": bboxes, "scores": scores, "pred_classes": classes}

        im = cv2.imread(d["file_name"])
        instances = Instances(im.shape[:2])
        instances.pred_boxes = Boxes(torch.FloatTensor(bboxes))
        instances.scores = torch.FloatTensor(scores)
        instances.pred_classes = torch.IntTensor(classes)
        
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(instances)
        cv2.imwrite(os.path.join(output_path, d["file_name"].split("/")[-1]), out.get_image()[:, :, ::-1])

    return


def check_args():
    parser = argparse.ArgumentParser()
                                                   
    parser.add_argument(
        "--output", 
        type=str,
        required=True,
        help="folder where to store images",
    )
    parser.add_argument(
        "--file", 
        type=str,
        required=True,
        help="pth file containing the model output",
    )
    parser.add_argument(
        "--th", 
        type=float,
        default=0.5,
        help="detection threshold",
    )    

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    main(parser)