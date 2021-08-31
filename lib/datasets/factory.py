# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}

from datasets.OpenImageSimpleCategory import OpenImageSimpleCategory
from datasets.CloudTableThings import CloudTableThings
from datasets.CloudTableThingsFineClass import CloudTableThingsFineClass
from datasets.CloudStatus import CloudStatus

import numpy as np

for split in ['train', 'validation', 'test']:
  name = 'OpenImageSimpleCategory_{}'.format(split)
  __sets[name] = (lambda split=split : OpenImageSimpleCategory(split))

for split in ['train', 'val', 'trainval']:
  name = 'CloudTableThings_{}'.format(split)
  __sets[name] = (lambda split=split : CloudTableThings(split))

for split in ['train', 'val', 'trainval']:
  name = 'CloudTableThingsFineClass_{}'.format(split)
  __sets[name] = (lambda split=split : CloudTableThingsFineClass(split))

for split in ['train', 'val', 'trainval']:
  name = 'CloudStatus_{}'.format(split)
  __sets[name] = (lambda split=split : CloudStatus(split))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
