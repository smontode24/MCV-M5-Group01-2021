from detectron2 import model_zoo

def obtain_model_cfg(cfg, model_name):
    if model_name == "resnet_dropblock":
        return modify_config_resnet_dropblock_fpn(cfg)
    elif model_name == "resnet_fpn":
        return modify_config_resnet_fpn(cfg)
    elif model_name == "resnet_deform_conv":
        return modify_config_resnet_deform_conv(cfg)

def modify_config_resnet_fpn(cfg):
    cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
    return cfg

def modify_config_resnet_dropblock_fpn(cfg):
    from new_registered_models import build_resnet_fpn_backbone_dropblock
    cfg = modify_config_resnet_fpn(cfg)
    cfg["MODEL"]["BACKBONE"]["NAME"] = "build_resnet_fpn_backbone_dropblock"
    return cfg

def modify_config_resnet_deform_conv(cfg):
    cfg.merge_from_file(model_zoo.get_config_file("Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml")  # Let training initialize from model zoo
    return cfg