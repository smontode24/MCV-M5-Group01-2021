import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.registry import Registry

from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head

from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from typing import Dict, List, Tuple, Union
from fvcore.nn import giou_loss, smooth_l1_loss


from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, FastRCNNOutputLayers 

# Set weights depending on imbalance
WEIGHTS = [1.8/10, 17.69/10, 46.95/10, 11.68/10, 229.22/10, 31.88/10, 101.42/10, 53.42/10, 1]
classes = ['car', 'van', 'truck', 'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc']
# car = 1.8
# pedestrian = 11.68
# tram = 101.42
# van = 17.69
# person_sitting = 229.22
# misc = 53.42
# truck = 46.95
# cycliust = 31.88
# dontcare = 4.65 <- pasar de estos supongo

class FastRCNNOutputsWeighted(FastRCNNOutputs):
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            weights = torch.tensor(WEIGHTS).float().to("cuda:"+str(self.pred_class_logits.get_device()))
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean", weight=weights)

class FastRCNNOutputLayersWeighted(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:
    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions
        losses = FastRCNNOutputsWeighted(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsWeighted(StandardROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.
    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayersWeighted(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

def modify_head_balanced_weight_class(cfg):
    cfg["MODEL"]["ROI_HEADS"]["NAME"] = "StandardROIHeadsWeighted"
    return cfg