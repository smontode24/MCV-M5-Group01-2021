import torch
import numpy as np
import pandas as pd
import argparse
import os

# EXAMPLE:
"""
MODEL=faster_balanced_longer
python prepare_evaluator_data.py --folder ~/W2_detectron2/${MODEL}/eval_kitti_test/ --file ~/test_kitti_full.txt
~/mcv/datasets/KITTI/devkit_kitti_txt/cpp/evaluate_object_txt ~/W2_detectron2/${MODEL}/eval_kitti_test/kitti_format/ ~/test_kitti_full.txt

"""

classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

def main(args):
    data = torch.load(os.path.join(args.folder, "instances_predictions.pth"))
    output_path = os.path.join(args.folder, "kitti_format/data")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(args.file, "r") as f:
        image_names = f.readlines()
        for image_name, img_data in zip(image_names, data):
            with open(os.path.join(output_path, image_name.replace("\n", "").split("/")[-1]), "w+") as out:
                for inst in img_data["instances"]:
                    cl = classes[inst["category_id"]]
                    bbox = inst["bbox"]
                    out.write(f"{cl} {-1} {-1} {-10} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[0] + bbox[2]:.2f} {bbox[1] + bbox[3]:.2f} {-1} {-1} {-1} {-1000} {-1000} {-1000} {-10} {inst['score']:.2f}\n")
                

                #str, &trash, &trash, &d.box.alpha, &d.box.x1, &d.box.y1,
                #    &d.box.x2, &d.box.y2, &d.h, &d.w, &d.l, &d.t1, &d.t2, &d.t3,
                #    &d.ry, &d.thresh)==16
                #break
            #break

    return


def check_args():
    parser = argparse.ArgumentParser()
                                                                                                                                    
    #TODO: option to select one model or another
    parser.add_argument(
        "--folder", 
        type=str,
        required=True,
        help="folder containing 'instances_predictions.pth'",
    )
    parser.add_argument(
        "--file", 
        type=str,
        required=True,
        help="file containing the GT list of the subset to evaluate",
    )

    return parser.parse_args()

if __name__ == "__main__":
    parser = check_args()
    main(parser)