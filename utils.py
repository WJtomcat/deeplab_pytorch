import numpy as np
import cv2
from PIL import Image

_MEAN_RGB = [123.15, 115.90, 103.06]

def preprocess_train(data, size=(513, 513)):
  image_path, gt_path = data
  im = cv2.imread(image_path)
  im = cv2.resize(im, size)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

  gt_im = np.array(Image.open(gt_path).resize(size))
  return im, gt_im
