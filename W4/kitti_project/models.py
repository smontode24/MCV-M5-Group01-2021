from detectron2 import model_zoo

def obtain_model_cfg(cfg, model_name, pretrained_weights=True):
    if pretrained_weights:
        print("Loading model pretrained weights...")
    else:
        print("Training from scratch...")

    if model_name == "coco_r50fpn": 
        return modify_config_cocorn50fpn(cfg, pretrained_weights)
    elif model_name == "coco_r50c4": 
        return modify_config_cocorn50c4(cfg, pretrained_weights)
    elif model_name == "coco_r50dc5": 
        return modify_config_cocorn50dc5(cfg, pretrained_weights)
    elif model_name == "cityscapes_r50fpn":
        return modify_config_cityscapesrn50fpn(cfg, pretrained_weights)
    elif model_name == "coco_r101fpn":
        return modify_config_cocorn101fpn(cfg, pretrained_weights) 
    elif model_name == "lvisr50":
        return modify_config_lvis50(cfg, pretrained_weights)
    else:
        raise Exception("Architecture/Model not found")

def modify_config_cocorn50fpn(cfg, pretrained_weights):
    yaml = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_cocorn50c4(cfg, pretrained_weights):
    yaml = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_cocorn50dc5(cfg, pretrained_weights):
    yaml = "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_cityscapesrn50fpn(cfg, pretrained_weights):
    yaml = "Cityscapes/mask_rcnn_R_50_FPN.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_cocorn101fpn(cfg, pretrained_weights):
    yaml = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_resnet50_conv4(cfg, pretrained_weights):
    yaml = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_resnet50_dilated_conv(cfg, pretrained_weights):
    yaml = "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_default_rcnn(cfg, pretrained_weights):
    yaml = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_retinanet101(cfg, pretrained_weights):
    yaml = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_retinanet50(cfg, pretrained_weights):
    yaml = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_lvis50(cfg, pretrained_weights):
    yaml = "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg