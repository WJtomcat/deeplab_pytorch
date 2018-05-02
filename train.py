import os
import sys

import torch
import numpy as np
import cv2
from PIL import Image
import random
from docopt import docopt


from model import Mobilenet_deeplabv3

docstr = \
"""Train ResNet-DeepLab on VOC12 (scenes) in pytorch using MSCOCO pretrained initialization

Usage:
    train.py [options]

Options:
    -h, --help                  Print this message
    --GTpath=<str>              Ground truth path prefix [default: data/gt/]
    --IMpath=<str>              Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --LISTpath=<str>            Input image number list file [default: data/list/val.txt]
    --lr=<float>                Learning Rate [default: 0.00025]
    -i, --iterSize=<int>        Num iters to accumulate gradients over [default: 10]
    --wtDecay=<float>          Weight decay during training [default: 0.0005]
    --gpu0=<int>                GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
"""

args = docopt(docstr, version='v0.1')

def flip(img, flip_p):
  if flip_p > 0.5:
    return np.fliplr(img)
  else:
    return img


def read_file(path_to_file):
  with open(path_to_file) as f:
    img_list = []
    for line in f:
      img_list.append(line[:-1])
  return img_list
