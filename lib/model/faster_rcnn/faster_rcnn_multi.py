import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

# from model.roi_layers import ROIAlign, ROIPool
from torchvision.ops import RoIAlign

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes0, classes1, class_agnostic, use_share_regress=False):
        super(_fasterRCNN, self).__init__()
        self.classes0 = classes0
        self.classes1 = classes1
        self.n_classes0 = len(classes0)
        self.n_classes1 = len(classes1)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn0 = _RPN(self.dout_base_model)
        self.RCNN_rpn1 = _RPN(self.dout_base_model)
        self.RCNN_proposal_target0 = _ProposalTargetLayer(self.n_classes0)
        self.RCNN_proposal_target1 = _ProposalTargetLayer(self.n_classes1)

        self.use_share_regress = use_share_regress

        # pytorch 0.4
        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        # pytorch 1.0
        # self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        # self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

        # torchvision.ops
        # self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), spatial_scale=1.0/16.0, sampling_ratio=0)
        if self.use_share_regress:
            self.RCNN_share_regress = nn.Linear(2048, 1)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, flow_id):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        # gt_boxes is to used to check bbox overlap
        # rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        if self.training:
            if flow_id == 0:
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn0(base_feat, im_info, gt_boxes[:, :, :5], num_boxes)
            elif flow_id == 1:
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn1(base_feat, im_info, gt_boxes[:, :, :5], num_boxes)
        else:
            if flow_id == 0:
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn0(base_feat, im_info, gt_boxes, num_boxes)
            elif flow_id == 1:
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn1(base_feat, im_info, gt_boxes, num_boxes)

        # print('RCNN_rpn.rois: ', rois[:, :10, :])
        # gt_boxes.shape [1, 30, 5]
        # gt_share.shape [1, 5]

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            # gt_boxes' last column is class index
            if flow_id == 0:
                roi_data = self.RCNN_proposal_target0(rois, gt_boxes, num_boxes)
            elif flow_id == 1:
                roi_data = self.RCNN_proposal_target1(rois, gt_boxes, num_boxes)

            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, rois_share = roi_data

            # # print('im_data: ', im_data)
            # print('gt_boxes: ', gt_boxes[:, :num_boxes.item(), :])
            # print('num_boxes: ', num_boxes.item())
            # print('rois: ', rois[:, :num_boxes.item(), :])
            # print('rois_target: ', rois_target[:, :num_boxes.item(), :])
            # # print('\n\n')
            # pdb.set_trace()

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            # pdb.set_trace()
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
        else:
            raise AssertionError('selected cfg.POOLING_MODE is unsupported: ', cfg.POOLING_MODE)

        # print('base_feat: ', base_feat)
        # print('rois: ', rois[:, :10, :])
        # print('pooled_feat: ', pooled_feat)
        # pdb.set_trace()

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute regression score
        # self.RCNN_share_regress = nn.Linear(2048, 1)
        if flow_id == 0:
            if self.use_share_regress:
                share_score_pred = self.RCNN_share_regress(pooled_feat)
                # share_pred = F.sigmoid(share_score_pred)  # deprecated
                share_pred = torch.sigmoid(share_score_pred)
                share_pred = share_pred.squeeze(1)

            bbox_pred = self.RCNN_bbox_pred0(pooled_feat)
            cls_score = self.RCNN_cls_score0(pooled_feat)

        elif flow_id == 1:
            bbox_pred = self.RCNN_bbox_pred1(pooled_feat)
            cls_score = self.RCNN_cls_score1(pooled_feat)

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        share_loss = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            # choose rois_lable, batch, rois_label
            if flow_id == 0 and self.use_share_regress:
                fg_idx = rois_label != 0
                share_loss = F.mse_loss(share_pred[fg_idx], rois_share[0, fg_idx]*0.01)

            # # pdb.set_trace()
            # print('rcnn_loss_cls: ', RCNN_loss_cls.item())
            # print('rcnn_loss_cls.cls_score: ', cls_score[:20, :])
            # print('rcnn_loss_cls.rois_label: ', rois_label[:20])

            # print('rcnn_loss_bbox: ', RCNN_loss_bbox.item())
            # print('rcnn_loss_bbox.rois_target: ', rois_target[:20, :])
            # print('rcnn_loss_bbox.bbox_pred: ', bbox_pred[:20, :])
            # print('\n\n')

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        if flow_id == 0 and self.use_share_regress:
            share_pred = share_pred.view(batch_size, rois.size(1), -1)
        else:
            share_pred = 0

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, share_pred, share_loss

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn0.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn0.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn0.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score0, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred0, 0, 0.001, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn1.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn1.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn1.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred1, 0, 0.001, cfg.TRAIN.TRUNCATED)

        if self.use_share_regress:
            normal_init(self.RCNN_share_regress, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
