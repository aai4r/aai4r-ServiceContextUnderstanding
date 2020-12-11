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
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.pascal_voc_cyclewater import pascal_voc_cyclewater
from datasets.pascal_voc_cycleclipart import pascal_voc_cycleclipart
from datasets.sim10k import sim10k
from datasets.water import water
from datasets.clipart import clipart
from datasets.sim10k_cycle import sim10k_cycle
from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.cityscape_10class import cityscape_10class
from datasets.foggy_cityscape import foggy_cityscape
from datasets.kitti_car import kitti_car
from datasets.bdd100k import bdd100k
from datasets.bdd100k_forcityscape import bdd100k_forcityscape

from datasets.cityscape_adain import cityscape_adain
from datasets.sim10k_adain import sim10k_adain
from datasets.pascal_voc_adainclipart import pascal_voc_adainclipart
from datasets.pascal_voc_water_adain import pascal_voc_water_adain
from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.vg import vg

from datasets.GMUKitchen import GMUKitchen
from datasets.OpenImageSimpleCategory import OpenImageSimpleCategory

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
  for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

for split in ['train']:
  name = 'cityscape_adain_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_adain(split))

for split in ['train']:
  name = 'sim10k_adain_{}'.format(split)
  __sets[name] = (lambda split=split : sim10k_adain(split))

for split in ['train']:
  name = 'pascal_voc_adainclipart_{}'.format(split)
  __sets[name] = (lambda split=split : pascal_voc_adainclipart(split))

for split in ['train']:
  name = 'pascal_voc_water_adain_{}'.format(split)
  __sets[name] = (lambda split=split : pascal_voc_water_adain(split))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

for split in ['train_normal', 'test_normal']:
  name = 'cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape(split))
for split in ['train_normal', 'test_normal']:
  name = 'cityscape_car_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_car(split))
for split in ['train_normal']:
  name = 'cityscape_10class_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_10class(split))
for split in ['train_foggy', 'test_foggy']:
  name = 'foggy_cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : foggy_cityscape(split))
for split in ['train']:
  name = 'sim10k_{}'.format(split)
  __sets[name] = (lambda split=split : sim10k(split))
for split in ['train', 'val']:
  name = 'sim10k_cycle_{}'.format(split)
  __sets[name] = (lambda split=split: sim10k_cycle(split))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_water_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc_water(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_cycleclipart_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cycleclipart(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_cyclewater_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cyclewater(split, year))
for year in ['2007']:
  for split in ['traintest']:   # 'trainval', 'test',
    name = 'clipart_{}'.format(split)
    __sets[name] = (lambda split=split : clipart(split,year))
for year in ['2007']:
  for split in ['train', 'test']:
    name = 'water_{}'.format(split)
    __sets[name] = (lambda split=split : water(split,year))

for split in ['train']:
  name = 'kitti_car_{}'.format(split)
  __sets[name] = (lambda split=split : kitti_car(split))

# to list all file list in ImageSets
for split in ['train', 'val', 'trainval']:
  for cond in ['clear', 'dawndusk', 'daytime', 'night', 'overcast', 'partlycloudy', 'rainy', 'snowy', 'foggy',
               'clear_daytime', 'clear_night',
               'overcast_daytime', 'partlycloudy_daytime', 'rainy_daytime', 'snowy_daytime', 'foggy_daytime']:
    name = 'bdd100k_{}_{}'.format(split, cond)
    __sets[name] = (lambda split=split, cond=cond: bdd100k(split, cond))

for split in ['train', 'val']:
  for cond in ['daytime']:
    name = 'bdd100k_forcityscape_{}_{}'.format(split, cond)
    __sets[name] = (lambda split=split, cond=cond: bdd100k_forcityscape(split, cond))

for split in ['train', 'trainbest', 'FOLD1trainval', 'FOLD1test', 'FOLD2trainval', 'FOLD2test', 'FOLD3trainval', 'FOLD3test', 'FOLD1testSampled']:
  name = 'GMUKitchen_{}'.format(split)
  __sets[name] = (lambda split=split : GMUKitchen(split))

for split in ['train', 'validation', 'test']:
  name = 'OpenImageSimpleCategory_{}'.format(split)
  __sets[name] = (lambda split=split : OpenImageSimpleCategory(split))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
