from pascal_voc import VOCDataset
import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from utils import preprocess_train

imdb = VOCDataset( 'train', './pascal_voc_seg/VOCdevkit/VOC2012/', 3, preprocess_train)
print(imdb)
im = imdb.next_batch()
print(imdb)