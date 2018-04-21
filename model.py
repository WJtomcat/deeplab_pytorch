import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Mobilenet_block(nn.Module):
  """
  Mobilenet_block
  inputtensor: batch_size * width * height * channels
  if stride != 1: intput -> expand_conv -> depthwise_conv -> BatchNorm -> project_conv
    the width and height would reduced in half, and the number of channels would increase
  if stride = 1: input -> expand_conv -> depthwise_conv -> BatchNorm -> project_conv -> + input
    the width and height and the number of channels would maintain unchanged。
  """
  def __init__(self,
               nin,
               nout,
               stride=1,
               rate=1,
               expand_multi=6
               residule=True):
    super(Mobilenet_block, self).__init__()

    self.expand_size = nin * expand_multi

    if not expand_multi == 1:
      self.expand_conv = Expand_conv(nin, self.expand_size)
    else:
      self.expand_conv = None
    self.depthwise_conv = Depthwise_conv(self.expand_size)
    self.project_conv = Project_conv(self.expand_size, nout)

    self.residule = residule
    self.stride = stride
    self.nin = nin
    self.nout = nout

  def forward(self, x):
    if self.expand_conv:
      out = self.expand_conv(x)
    out = self.depthwise_conv(out)
    out = self.project_conv(out)
    if (self.residule and
        self.stride == 1 and
        self.nin == nout):
      out += x
    return out

def padsize(kernel_size, rate=1):
  """
  to calculate the padsize on the begin and end side of height and width dimension
  """
  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  padbeg = pad_total // 2
  padend = pad_total - padbeg
  return padbeg, padend


class Depthwise_conv(nn.Module):
  """
  Depthwise convolution with BatchNorm
  """
  def __init__(self, nin, kernel_size=3, stride=1, rate=1):
    super(Depthwise_conv, self).__init__()
    padbeg, padend = padsize(kernel_size, rate=rate)
    self.pad = nn.ZeroPad2d((padbeg, padend, padbeg, padend))
    self.depthwise_conv = nn.Conv2d(nin, nin, stride=stride, kernel_size=kernel_size, dilation=rate, groups=nin)
    self.batchnum = nn.BatchNorm2d(nin)

  def forward(self, x):
    out = self.pad(x)
    out = self.depthwise_conv(out)
    out = self.batchnum(out)
    return out

class Expand_conv(nn.Module):
  """
  Expand convolution to multi the channels using a pointwise 1x1 convlution.
  With BatchNorm and Relu6.
  """
  def __init__(self, nin, nout):
    super(Expand_conv, self).__init__()
    self.pointwise_conv = nn.Conv2d(nin, nout, kernel_size=1)
    self.batchnorm = nn.BatchNorm2d(nout)
    self.relu6 = nn.ReLU6()

  def forward(self, x):
    out = self.pointwise_conv(x)
    out = self.batchnorm(out)
    out = self.relu6(out)
    return out

class Project_conv(nn.Module):
  """
  Project convolution using a pointwise conv to shrink the number of channels, with a batchnorm layer.
  """
  def __init__(self, nin, nout):
    super(Project_conv, self).__init__()
    self.project_conv = nn.Conv2d(nin, nout, kernel_size=1)
    self.batchnorm = nn.BatchNorm2d(nout)

  def forward(self, x):
    out = self.project_conv(x)
    out = self.batchnorm(out)
    return out


class Mobilenet_base(nn.Module):
  """
  Mobilenet_base
  """
  def __init__(self):
    super(Mobilenet_base, self).__init__()
    self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=2)
    self.batchnorm = nn.BatchNorm2d(32)
    self.expanded_conv = Mobilenet_block(32, 16, stride=1, expand_multi=1, residule=False)
    self.expanded_conv1 = Mobilenet_block(16, 24, stride=2)
    self.expanded_conv2 = Mobilenet_block(24, 24)
    self.expanded_conv3 = Mobilenet_block(24, 32, stride=2)
    self.expanded_conv4 = Mobilenet_block(32, 32)
    self.expanded_conv5 = Mobilenet_block(32, 32)
    self.expanded_conv6 = Mobilenet_block(32, 64, stride=2, rate=2)
    self.expanded_conv7 = Mobilenet_block(64, 64, rate=2)
    self.expanded_conv8 = Mobilenet_block(64, 64, rate=2)
    self.expanded_conv9 = Mobilenet_block(64, 64, rate=2)
    self.expanded_conv10 = Mobilenet_block(64, 96, rate=2)
    self.expanded_conv11 = Mobilenet_block(96, 96, rate=2)
    self.expanded_conv12 = Mobilenet_block(96, 96, rate=2)
    self.expanded_conv13 = Mobilenet_block(96, 160, stride=2, rate=4)
    self.expanded_conv14 = Mobilenet_block(160, 160, rate=4)
    self.expanded_conv15 = Mobilenet_block(160, 160, rate=4)
    self.expanded_conv16 = Mobilenet_block(160, 320, rate=4)

  def forward(self, x):
    pass


class Mobilenet_deeplabv3(nn.Module):
  """
  Deeplabv3 using Mobilenetv2 as base net.
  """
  def __init__(self):
    super(Mobilenet_deeplabv3, self).__init__()
    self.basenet = Mobilenet_base()

    self.imagePool = nn.AvgPool2d()
    self.imageConv = nn.Conv2d(320, 256, kernel_size=1)
    self.imageBatchnorm = nn.BatchNorm2d(256)
    self.imageRelu = nn.ReLU()
    self.resizeBilinear = nn.UpsamplingBilinear2d(size=(65, 65))

    self.headConv = nn.Conv2d(320, 256, kernel_size=1)
    self.headBatchnorm = nn.BatchNorm2d(256)
    self.headRelu = nn.ReLU()

    self.concatConv = nn.Conv2d(512, 256, kernel_size=1)
    self.concatBatchnorm = nn.BatchNorm2d(256)
    self.concatRelu = nn.ReLU()

    self.predict = nn.Conv2d(256, 21, kernel_size=1)

    self.upsample = nn.UpsamplingBilinear2d(size=(513, 513))




  def forward(self, x):
    feature = self.basenet(x)

    imagepool = self.imagePool(feature)
    imagepool = self.imageConv(imagepool)
    imagepool = self.imageBatchnorm(imagepool)
    imagepool = self.imageRelu(imagepool)
    imagepool = self.resizeBilinear(imagepool)

    headconv = self.headConv(feature)
    headconv = self.headBatchnorm(headconv)
    headconv = self.headRelu(headconv)

    concat = torch.cat((imagepool, headconv), 1)

    concat = self.concatConv(concat)
    concat = self.concatBatchnorm(concat)
    concat = self.concatRelu(concat)

    predict = self.predict(concat)

    values, indices = torch.max(predict, 0)
    return indices
