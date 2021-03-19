from detectron2 import model_zoo

def obtain_model_cfg(cfg, model_name, pretrained_weights=True):
    if pretrained_weights:
        print("Loading model pretrained weights...")
    else:
        print("Training from scratch...")
    
    if model_name == "resnet_dropblock":
        return modify_config_resnet50_dropblock_fpn(cfg, pretrained_weights)
    elif model_name == "resnet50_fpn":
        return modify_config_resnet50_fpn(cfg, pretrained_weights)
    elif model_name == "resnet_dilated_conv":
        return modify_config_resnet50_dilated_conv(cfg, pretrained_weights)
    elif model_name == "resnet_conv4":
        return modify_config_resnet50_conv4(cfg, pretrained_weights)
    elif model_name == "faster_rcnn":
        return modify_config_default_rcnn(cfg, pretrained_weights)
    elif model_name == "resnet101_fpn":
        return modify_config_resnet101_fpn(cfg, pretrained_weights)
    elif model_name == "retinanet50":
        return modify_config_retinanet50(cfg, pretrained_weights)
    elif model_name == "retinanet101":
        return modify_config_retinanet101(cfg, pretrained_weights)
        

def modify_config_resnet50_fpn(cfg, pretrained_weights):
    yaml = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_resnet101_fpn(cfg, pretrained_weights):
    yaml = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    if pretrained_weights:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_resnet50_dropblock_fpn(cfg, pretrained_weights):
    from new_registered_models import build_resnet_fpn_backbone_dropblock
    cfg = modify_config_resnet50_fpn(cfg, pretrained_weights)
    cfg["MODEL"]["BACKBONE"]["NAME"] = "build_resnet_fpn_backbone_dropblock"
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