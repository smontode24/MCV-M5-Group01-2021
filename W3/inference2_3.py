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


# Personal NOTES:
# Use object pre-trained object detection
# models in inference on KITTI-MOTS

##
# How to register COCO Fromat Datasets
# https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#register-a-coco-format-dataset


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

import kitti_project.loaders as loaders

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
import io
import pandas as pd
import re
import pickle as pkl

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
    '''
    prob2 = prob2.replace(" ","")
    prob2 = prob2.replace(","," ")
    print("RESULT: ", prob2)

    final_result = prob2
    return final_result

def write_to_file(data, path):
    with open(path, 'w') as f:
        for i in data:
            print(i, file=f)

# DATA
# num_objects = first column
# num_scores = second col
# class_detected = third col
#
# total_detected = calculated. How many class detected? (int)
#
# num_scores_below_33 = calculated  - Total below 33
# class_below_33 = calculated
# class_ratio_below_33 = from the total class detected, which are the more "under determined"
# class_ratio_below_33_not_in_other_thresholds =
#      .... Maybe artifacts of the scene. Below 33 and not in any other class classification.
#
# num_scores_below_66 = calculated  - Total below 66
# class_below_66 = calculated. Between 33 - 66%
# class_ratio_below_66 = from the total class detected, which are in the range 33-66%
# class_ratio_below_66_not_in_other_thresholds =
#      .... Between 33-66.... gray area. This one should be defined based on results
#
# num_scores_upper_66 = calculated  - Total with more than 66
# class_upper_66 = Between 66 - 100%
# class_ratio_upper_66 = from the total class detected, which are in the range 66-100%
# class_ratio_upper_66_not_in_other_thresholds =
#      .... More than 66%.... If we found something here,
#      we should consider that as a true positive match (unless a human tell the opposite)
# index =
#      .... We will have our classes divided in 3 categories
#      .... And then, based on how many classes detected on each category
#      .... we can create an index to show how "good" is our detection

# POST: Get
def get_attributes(line):
    text = output_to_csv_line(line)
    t1 = text.split(';')
    print ("T1 is: ", t1)
    num_objects = text


# POST: Given a line used in the output of Detectron, it modifies memory and append the data
def manage_data(memory, line):
    row = get_attributes(line)
    memory.append(row)
    return memory


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

# POST: It manages the input of a loaded annotations file from .txt
def manage_annotations(input):
    output = []
    for dict in input:      # 2642 elements using img_00002
        frame_related = dict["image_id"]
        for annot in dict["annotations"]:  #
            type = annot["category_id"]
            bbox = annot["bbox"]

            output.append([frame_related,bbox,type])

    #print ("Final annotations have been processed")
    #print ("With a size of: " + str(len(load)))

    return output

# POST: It manages the input of a loaded annotations file from .txt
def manage_predictions(input):
    output = []

    ## input should be: [id_frame, prediction_from_detectron]
    for index, line in input:
        type = line["instances"].get("pred_classes")
        bbox = line["instances"].get("pred_boxes")
        scores = line["instances"].get("scores")

        types = []
        bboxes = []
        score_list = []

        # Treat each type of data

        text_type = str(type)                       # Noisy text: tensor at beginning + cuda at the end
        t10 = re.sub(r"[\n\t\s]*", "", text_type)   # Removing all spaces. Equivalent to ''.join(text_type)
        t11 = t10.replace("tensor([", "")           # Deleting first part
        t12 = t11.replace("],device=\'cuda:0\')", "")   # Deleting last part
        regex_pattern = '\d{1,3}(?:,\d{3})*'  # REGEX: 1 to 3 digits; followed by 0 or more the non capture group comma followed by 3 digits - https://www.reddit.com/r/regex/comments/90g73v/identifying_comma_separated_numbers/
        for match in re.finditer(regex_pattern, t12):
            sGroup = match.group()
            types.append(sGroup)

        text_bbox = str(bbox)
        b10 = re.sub(r"[\n\t\s]*", "", text_bbox)   # Removing all spaces. Equivalent to ''.join(text_type)
        b11 = b10.replace("tensor([", "")           # Removing just the first "brackets" of tensor([
        regex_pattern = '\[(.*?)\]'                 # and taking content between brackets of the rest
        for match in re.finditer(regex_pattern, b11):
            sGroup = match.group()
            bboxes.append(sGroup)

        text_scores = str(scores)                   # Very similar to first case, types.
        s10 = re.sub(r"[\n\t\s]*", "", text_scores)
        s11 = s10.replace("tensor([", "")
        s12 = s11.replace("],device=\'cuda:0\')", "")
        regex_pattern = '[+-]?([0-9]*[.])?[0-9]+'  # REGEX: https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers
        for match in re.finditer(regex_pattern, s12):
            sGroup = match.group()
            score_list.append(sGroup)

        ## ASSERT All list should have the same length
        if (len(score_list)!=len(bboxes) or len(score_list)!=len(types)):       #score_list is the most buggy candidate. Check same size among others
            raise AssertionError("Some differences have been found in regex. Please, check the patterns")

        for index2 in range(len(types)):
            # frame_id = Retrieved by index of first loop; [bbox]; score; type
            row = [index, bboxes[index2], score_list[index2], types[index2]]
            output.append(row)

    return output

test_bbox1 = [10, 20, 30, 40]
test_bbox2 = [20, 20, 30, 40]


def compare_both (annotations, predictions):
    #area1 = int(bbox1[3])*int(bbox1[4])         #INT. Check, as maybe floats can be required in the future
    #area2 = int(bbox2[3])*int(bbox2[4])

    write_to_file(annotations, "anot_out.txt")
    write_to_file(predictions, "pred_out.txt")

    # max_fram_num = max(int(anottations[0])
    # num_frames = np.argmax(anottations)
    maximum = max(annotations, key=lambda x: x[0])

    '''
    index1 = 0
    index2 = 0

    for i in range(0,len(anottations)):
        for j in range(i, len(predictions)):
            if j != i:
                break
            else:
                j = index2
                j = predictions[j][0]  #getting id_frame
    '''


    print('hey')




kitti_mots_splits = {
    "train": [0, 1, 3, 4, 5, 9, 11, 12, 15],
    "val": [17, 19, 20],
    "test": [2, 6, 7, 8, 10, 13, 14, 16, 18]
}


# IN TYPE 1: We use the first output format
def do_experiments_type1(cfg, predictor, train_images):
    '''
    results = []
    predictions = []
    for index,im_path in enumerate(train_images):
        im = cv2.imread(im_path)
        outputs = predictor(im)
        row = [index, outputs]
        #print (index)
        predictions.append(row)

    # we strip the predictions in order to have in a similar format to the annotations file
    predictions = manage_predictions(predictions)

    ## loading dataset annotations
    dataset_dicts = []
    # dataset_dicts = loaders.get_kitti_mots('test', dataset_dicts)             ## OPTIONAL REPRESENTATION
    # Load the dataset and read it
    anottations_dataset = loaders.read_full_ds('test', dataset_dicts, "/home/group01/mcv/datasets/KITTI-MOTS", "training", "image_02", kitti_mots_splits, False)
    # convert it to same format as predictions
    anottations_dataset = manage_annotations(anottations_dataset)

    # SAVING:
    f = open('anotattions.pckl', 'wb')
    pickle.dump(anottations_dataset, f)
    f.close()

    # SAVING:
    f = open('predictions.pckl', 'wb')
    pickle.dump(predictions, f)
    f.close()
    '''

    anottations_dataset = pkl.load(open('anotattions.pckl', 'rb'))
    predictions = pkl.load(open('predictions.pckl', 'rb'))

    compare_both(anottations_dataset, predictions)

    print("hey22")
    # Just if you need visualize it.....
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    print ("Hey, we have here our V: ", v)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    print("Hey, we have here our out: ", out)
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

PATH = "/home/mcv/datasets/KITTI-MOTS/training/image_02/0002/"
PATTERN = "*.png"

# Get images
images = list_images_from_path(PATH, PATTERN)

# Models to analize
FASTER_RCNN = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
RETINANET = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
FASTER_RCNN_3 = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
RETINANET_3 = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

results = []

###
#  FIRST 6 experiments: [0.25, 0.50, 0.75] combined on RetinaNET x1
#
## 1: RetinaNet @ 0.25
cfg, predictor = generate_predictor(0.25,RETINANET)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment1.csv',results)
calculate_performance(t0,t1, len(images))
'''
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

## 6: FASTER_RCNN @ 0.7
cfg, predictor = generate_predictor(0.7,FASTER_RCNN)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment6.csv',results)
calculate_performance(t0,t1, len(images))

##
# SECOND 6 experiments: [0.25, 0.50, 0.75] combined on RetinaNET x3
# Yes, x3 format
#

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
'''

## 6: FASTER_RCNN @ 0.7
cfg, predictor = generate_predictor(0.7,FASTER_RCNN_3)
t0 = time.time()
results = do_experiments_type1(cfg, predictor, images)
t1 = time.time()
write_results('experiment16.csv',results)
calculate_performance(t0,t1, len(images))


# Get images  [Whole Dataset]
# Now, instead of using just a selection, we took the whole dataset
images = list_images_from_path("/home/mcv/m5/datasets/MIT_split/train", PATTERN)

'''

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

'''
print("Hey, I'm done here!")

