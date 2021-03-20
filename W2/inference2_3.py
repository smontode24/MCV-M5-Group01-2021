"""

███████╗███████╗████████╗██╗   ██╗██████╗     ████████╗██╗  ██╗███████╗    ███████╗███╗   ██╗██╗   ██╗██╗██████╗  ██████╗ ███╗   ██╗███╗   ███╗███████╗███╗   ██╗████████╗
██╔════╝██╔════╝╚══██╔══╝██║   ██║██╔══██╗    ╚══██╔══╝██║  ██║██╔════╝    ██╔════╝████╗  ██║██║   ██║██║██╔══██╗██╔═══██╗████╗  ██║████╗ ████║██╔════╝████╗  ██║╚══██╔══╝
███████╗█████╗     ██║   ██║   ██║██████╔╝       ██║   ███████║█████╗      █████╗  ██╔██╗ ██║██║   ██║██║██████╔╝██║   ██║██╔██╗ ██║██╔████╔██║█████╗  ██╔██╗ ██║   ██║
╚════██║██╔══╝     ██║   ██║   ██║██╔═══╝        ██║   ██╔══██║██╔══╝      ██╔══╝  ██║╚██╗██║╚██╗ ██╔╝██║██╔══██╗██║   ██║██║╚██╗██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║
███████║███████╗   ██║   ╚██████╔╝██║            ██║   ██║  ██║███████╗    ███████╗██║ ╚████║ ╚████╔╝ ██║██║  ██║╚██████╔╝██║ ╚████║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║
╚══════╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝            ╚═╝   ╚═╝  ╚═╝╚══════╝    ╚══════╝╚═╝  ╚═══╝  ╚═══╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝

"""

## With this dependencies, would be feasible to run the program in Google Collab

# CUDA DRIVERS in Ubuntu 20.04 (WSL2) - https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0

# !pip install pyyaml==5.1
# !pip install torch==1.7.1 torchvision==0.8.2
# !pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
# !pip install opencv

"""
██╗███╗   ███╗██████╗  ██████╗ ██████╗ ████████╗███████╗
██║████╗ ████║██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝
██║██╔████╔██║██████╔╝██║   ██║██████╔╝   ██║   ███████╗
██║██║╚██╔╝██║██╔═══╝ ██║   ██║██╔══██╗   ██║   ╚════██║
██║██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║   ██║   ███████║
╚═╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝
"""

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import glob
import os
from fnmatch import fnmatch
import time

"""
██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗ ███████╗
██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗██╔════╝
███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝███████╗
██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║
██║  ██║███████╗███████╗██║     ███████╗██║  ██║███████║
╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝
                                                        
"""

"""
When converting: outputs["instances"].to("cpu") to string, the expected result is:

{'instances': Instances(num_instances=1, image_height=256, image_width=256, fields=[pred_boxes: Boxes(tensor([[167.9608, 182.4473, 184.4191, 195.6117]], device='cuda:0')), scores: tensor([0.7329], device='cuda:0'), pred_classes: tensor([8], device='cuda:0')])}
"""
# POST: When receiving the output, it makes a string conversions to obtain a simplified string and then use it in CSV
def output_to_csv_line(output):
    # Get full string
    text = str(output)
    # delete endlines
    text = text.replace('\n','').replace('\r','')
    # delete first part of the string, till num of instances
    nInstances1 = text[text.find("=")+1:]
    # should be now: 1, image_height=256, image_width=256, fields=[pred_boxes: Boxes(tensor([[111.0964, 140.7526, 130.6315, 158.4013]])), scores: tensor([0.8170]), pred_classes: tensor([2])])
    nInstances2= nInstances1[:nInstances1.find(',')]
    # now we focus on looking other parts and repeat the logic till the end
    # probability:
    prob1 = nInstances1[nInstances1.find(' tensor(')+8:]
    prob2 = prob1[:prob1.find(']')+1]
    # classes:
    classes1 = prob1[prob1.find('pred_classes: tensor(')+21:]
    classes2 = classes1[:classes1.find(']')+1]
    """ #Debug
    print("Text = ", text)
    print("Ninstances = ", nInstances2)
    print("Precission = ", prob2)
    print("Classes = ", classes2)
    """
    final_result = nInstances2 + ";" + prob2 + ";" + classes2
    return final_result

# POST: In this 2nd version of the output to csv we use just scores, separated by whitespaces
def output_to_csv_line_only_scores(output):
    # Get full string
    text = str(output)
    # delete endlines
    text = text.replace('\n','').replace('\r','')
    # delete first part of the string, till num of instances
    nInstances1 = text[text.find("=")+1:]
    # should be now: 1, image_height=256, image_width=256, fields=[pred_boxes: Boxes(tensor([[111.0964, 140.7526, 130.6315, 158.4013]])), scores: tensor([0.8170]), pred_classes: tensor([2])])
    nInstances2= nInstances1[:nInstances1.find(',')]
    # now we focus on looking other parts and repeat the logic till the end
    # probability:
    prob1 = nInstances1[nInstances1.find(' tensor(')+9:]
    prob2 = prob1[:prob1.find(']')]
    # classes:
    classes1 = prob1[prob1.find('pred_classes: tensor(')+21:]
    classes2 = classes1[:classes1.find(']')+1]

    '''
    print("Text = ", text)
    print("Ninstances = ", nInstances2)
    print("Precission = ", prob2)
    print("Classes = ", classes2)

    prob2 = prob2.replace(" ","")
    prob2 = prob2.replace(","," ")
    print("RESULT: ", prob2)
    '''
    final_result = prob2
    return final_result


# POST: Given a PATH and a file pattern (e.g: *jpg) it creates you a list of image_paths
def list_images_from_path(path, pattern):
    im_paths = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern):
                im_paths.append(os.path.join(path, name))
    print( len(im_paths), " have been found in " + path + " with a given extension: "+ pattern)
    return im_paths


# POST: Given tot ticks, it shows some statits
def calculate_performance(t1, t2, nImages):
    print ("TIME ELAPSED:\t", t2-t1)
    print ("AVG time for img:\t", (t2-t1)/nImages)


# POST: Encapsulating the generation
def generate_predictor(threshold, model):
    cfg = get_cfg()  # get a fresh new config
    cfg.merge_from_file(model_zoo.get_config_file(model))
    # AIXO ES IMPORTANT! Si baixes el threshold, et surtiran més deteccions
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    predictor = DefaultPredictor(cfg)

    # print ("A model of type: " + str(model) + " with Threshold: " + threshold + " will be used")

    return cfg, predictor

# IN TYPE 1: We use the first output format
def do_experiments_type1(cfg, predictor, train_images):
    results = []
    for im_path in train_images:
        im = cv2.imread(im_path)
        outputs = predictor(im)
        # Just if you need visualize it.....
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])
        results.append(output_to_csv_line(outputs))

    return results

# IN TYPE 2: We use the second output format, focused on weights with panda formats
def do_experiments_type2(cfg, predictor, train_images):
    results = []
    i = 0
    for im_path in train_images:
        im = cv2.imread(im_path)
        outputs = predictor(im)
        i+=1
        # Just if you need visualize it.....
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])
        results.append(output_to_csv_line_only_scores(outputs))

    return results

def write_results(path, results):
    # opening the csv file in 'w+' mode
    file = open(path, 'w+')
    for line in results:
        file.write(line + '\n')
    file.close()

"""
███████╗██╗  ██╗██████╗ ███████╗██████╗ ██╗███╗   ███╗███████╗███╗   ██╗████████╗███████╗
██╔════╝╚██╗██╔╝██╔══██╗██╔════╝██╔══██╗██║████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔════╝
█████╗   ╚███╔╝ ██████╔╝█████╗  ██████╔╝██║██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ███████╗
██╔══╝   ██╔██╗ ██╔═══╝ ██╔══╝  ██╔══██╗██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ╚════██║
███████╗██╔╝ ██╗██║     ███████╗██║  ██║██║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ███████║
╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝
                                                                                         
"""

# We like to compare two models. To do that, these are some Hypothesis we would like to check:

# Hypothesis 1: Which one of both identifies more correct elements? (considering same threshold, accuracy)
# Hypothesis 2: Which one of both identifies more type of elements? (wider recognition)
# Hypothesis 3: Is any more prone to errors? (false positive & false negatives)
# Hypothesis 4: Difference between using 1x ; 3x; Pyramids ; DC ?

# We are using MIT dataset, the same as in M3

PATH = "/home/mcv/m5/datasets/MIT_split/train/highway"
PATTERN = "*.jpg"

# Get images
images = list_images_from_path(PATH, PATTERN)

# Models to analize
FASTER_RCNN = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
RETINANET = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
FASTER_RCNN_3 = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
RETINANET_3 = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

results = []

## 1: RetinaNet @ 0.25
cfg, predictor = generate_predictor(0.25,RETINANET)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
print(results)
t1 = time.time()
write_results('experiment1.csv',results)

calculate_performance(t0,t1, len(images))

## 2: RetinaNet @ 0.5
cfg, predictor = generate_predictor(0.5,RETINANET)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment2.csv',results)
calculate_performance(t0,t1, len(images))

## 3: RetinaNet @ 0.7
cfg, predictor = generate_predictor(0.7,RETINANET)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment3.csv',results)
calculate_performance(t0,t1, len(images))

## 4: FASTER_RCNN @ 0.25
cfg, predictor = generate_predictor(0.25,FASTER_RCNN)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment4.csv',results)
calculate_performance(t0,t1, len(images))

## 5: FASTER_RCNN @ 0.5
cfg, predictor = generate_predictor(0.5,FASTER_RCNN)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment5.csv',results)
calculate_performance(t0,t1, len(images))

## 5: FASTER_RCNN @ 0.7
cfg, predictor = generate_predictor(0.7,FASTER_RCNN)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment6.csv',results)
calculate_performance(t0,t1, len(images))


## 1: RetinaNet @ 0.25
cfg, predictor = generate_predictor(0.25,RETINANET_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment11.csv',results)
calculate_performance(t0,t1, len(images))

## 2: RetinaNet @ 0.5
cfg, predictor = generate_predictor(0.5,RETINANET_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment12.csv',results)
calculate_performance(t0,t1, len(images))

## 3: RetinaNet @ 0.7
cfg, predictor = generate_predictor(0.7,RETINANET_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment13.csv',results)
calculate_performance(t0,t1, len(images))

## 4: FASTER_RCNN @ 0.25
cfg, predictor = generate_predictor(0.25,FASTER_RCNN_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment14.csv',results)
calculate_performance(t0,t1, len(images))

## 5: FASTER_RCNN @ 0.5
cfg, predictor = generate_predictor(0.5,FASTER_RCNN_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment15.csv',results)
calculate_performance(t0,t1, len(images))

## 5: FASTER_RCNN @ 0.7
cfg, predictor = generate_predictor(0.7,FASTER_RCNN_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment16.csv',results)
calculate_performance(t0,t1, len(images))


# Get images  [Whole Dataset]
images = list_images_from_path("/home/mcv/m5/datasets/MIT_split/train", PATTERN)

## 1: RetinaNet @ 0.25
cfg, predictor = generate_predictor(0.25,RETINANET_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment21.csv',results)
calculate_performance(t0,t1, len(images))

## 2: RetinaNet @ 0.5
cfg, predictor = generate_predictor(0.5,RETINANET_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment22.csv',results)
calculate_performance(t0,t1, len(images))

## 3: RetinaNet @ 0.7
cfg, predictor = generate_predictor(0.7,RETINANET_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment23.csv',results)
calculate_performance(t0,t1, len(images))

## 4: FASTER_RCNN @ 0.25
cfg, predictor = generate_predictor(0.25,FASTER_RCNN_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment24.csv',results)
calculate_performance(t0,t1, len(images))

## 5: FASTER_RCNN @ 0.5
cfg, predictor = generate_predictor(0.5,FASTER_RCNN_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment25.csv',results)
calculate_performance(t0,t1, len(images))

## 5: FASTER_RCNN @ 0.7
cfg, predictor = generate_predictor(0.7,FASTER_RCNN_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment26.csv',results)
calculate_performance(t0,t1, len(images))


### GERMAN EXPERIMENTS & OUTPUT
### WORKING ON THE SCORES DISTRIBUTION
## ALL THRESHOLD = 0

# The Whole Dataset
PATH = "/home/mcv/m5/datasets/MIT_split/"
PATTERN = "*.jpg"

# Get images
images = list_images_from_path(PATH, PATTERN)

# Models to analize
FASTER_1 = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
RETINANET_1 = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"

FASTER_101 = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
RETINANET_101 = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

FASTER_XXX = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"

FASTER_1_3x = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
RETINANET_1_3x = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"


# G1 = FASTER_1
cfg, predictor = generate_predictor(0, FASTER_1)
t0 = time.time()
results = do_experiments_type2(cfg, predictor, images)
t1 = time.time()
write_results('Faster_1.csv',results)
calculate_performance(t0,t1, len(images))

# G2 = RETINANET_1
cfg, predictor = generate_predictor(0, RETINANET_1)
t0 = time.time()
results = do_experiments_type2(cfg, predictor, images)
t1 = time.time()
write_results('Retinanet_1.csv',results)
calculate_performance(t0,t1, len(images))

# G3 = FASTER_101
cfg, predictor = generate_predictor(0, FASTER_101)
t0 = time.time()
results = do_experiments_type2(cfg, predictor, images)
t1 = time.time()
write_results('Faster_101.csv',results)
calculate_performance(t0,t1, len(images))

# G4 = RETINANET_101
cfg, predictor = generate_predictor(0, RETINANET_101)
t0 = time.time()
results = do_experiments_type2(cfg, predictor, images)
t1 = time.time()
write_results('Retinanet_101.csv',results)
calculate_performance(t0,t1, len(images))

# G5 = FASTER_XXX
cfg, predictor = generate_predictor(0, FASTER_XXX)
t0 = time.time()
results = do_experiments_type2(cfg, predictor, images)
t1 = time.time()
write_results('Faster_XXX.csv',results)
calculate_performance(t0,t1, len(images))


# G6 = FASTER_1 x3
cfg, predictor = generate_predictor(0, FASTER_1_3x)
t0 = time.time()
results = do_experiments_type2(cfg, predictor, images)
t1 = time.time()
write_results('Faster_1_3x.csv',results)
calculate_performance(t0,t1, len(images))

# G7 = RETINANET_1 x3
cfg, predictor = generate_predictor(0, RETINANET_1_3x)
t0 = time.time()
results = do_experiments_type2(cfg, predictor, images)
t1 = time.time()
write_results('Retinanet_1_3x.csv',results)
calculate_performance(t0,t1, len(images))


print("Hey, I'm done here!")

