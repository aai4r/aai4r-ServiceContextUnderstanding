# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

class vgg16(_fasterRCNN):
  def __init__(self, classes, use_pretrained, freeze_base=False, pretrained_path=None, class_agnostic=False):
    self.dout_base_model = 512
    self.pretrained = use_pretrained
    self.model_path = pretrained_path  # 'data/pretrained_model/vgg16_caffe.pth'
    self.class_agnostic = class_agnostic
    self.freeze_base = freeze_base

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" % (self.model_path))
        old_state_dict = torch.load(self.model_path)
        new_state_dict = vgg.state_dict()  # all things are copied, key and value of target net

        # change pretrained-model-name to match with the FRCN
        for key, value in old_state_dict.items():  # ex: encoder.conv1. weight, .. projector.0.weight
          if 'encoder' in key:
            new_key = key.replace('encoder.', '')
          else:
            new_key = key  # this could be 'conv1' or 'projector'

          if new_key in new_state_dict:
            new_state_dict[new_key] = value
          else:
            print('\t[%s] key is ignored because encoder is not included' % key)

        # pdb.set_trace()
        vgg.load_state_dict({k: v for k, v in new_state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # freeze
    if self.freeze_base:
      print("Freeze all base layers: %d" % len(self.RCNN_base))
      for i in range(len(self.RCNN_base)):
        for p in self.RCNN_base[i].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

