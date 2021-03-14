from detectron2 import model_zoo

def obtain_model_cfg(cfg, model_name):
    if model_name == "resnet_dropblock":
        return modify_config_resnet50_dropblock_fpn(cfg)
    elif model_name == "resnet50_fpn":
        return modify_config_resnet50_fpn(cfg)
    elif model_name == "resnet_dilated_conv":
        return modify_config_resnet50_dilated_conv(cfg)
    elif model_name == "faster_rcnn":
        return modify_config_default_rcnn(cfg)
    elif model_name == "resnet101_fpn":
        return modify_config_resnet101_fpn(cfg)

def modify_config_resnet50_fpn(cfg):
    yaml = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_resnet101_fpn(cfg):
    yaml = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_resnet50_dropblock_fpn(cfg):
    from new_registered_models import build_resnet_fpn_backbone_dropblock
    cfg = modify_config_resnet50_fpn(cfg)
    cfg["MODEL"]["BACKBONE"]["NAME"] = "build_resnet_fpn_backbone_dropblock"
    return cfg

def modify_config_resnet50_dilated_conv(cfg):
    yaml = "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg

def modify_config_default_rcnn(cfg):
    yaml = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(yaml))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo
    return cfg