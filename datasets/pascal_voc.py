import os
import numpy as np
from imdb import ImageDataset
from multiprocessing import Pool




class VOCDataset:
  def __init__(self, imageset, datadir, batch_size, im_processor,
               processes=1, shuffle=True, dst_size=None):
    self._image_set = imageset
    self._data_dir = datadir
    self._batch_size = batch_size

    self._epoch = -1

    self._image_indexes = []
    self._image_names = []
    self._gt_names = []

    self._image_ext = '.jpg'

    self._shuffle = shuffle
    self._pool_processes = processes
    self.pool = Pool(self._pool_processes)
    self.gen = None
    self._im_processor = im_processor

    self.load_dataset()


  def next_batch(self):
    batch = {'images':[], 'gt':[]}
    i = 0
    while i < self.batch_size:
      try:
        images, gt = next(self.gen)
        batch['images'].append(images)
        batch['gt'].append(gt)
        i += 1

      except (StopIteration, AttributeError, TypeError):
        indexes = np.arange(len(self.image_names), dtype=np.int)
        if self._shuffle:
          np.random.shuffle(indexes)
        self.gen = self.pool.imap(self._im_processor,
                                  ((self.image_names[i], self.gt_names[i])
                                  for i in indexes),
                                  chunksize=self.batch_size)
        self._epoch += 1
        print(('epoch {} start...'.format(self._epoch)))
    batch['images'] = np.asarray(batch['images'])
    batch['gt'] = np.asarray(batch['gt'])
    return batch


  def close(self):
    self.pool.terminate()
    self.pool.join()
    self.gen = None



  def load_dataset(self):
    self._image_indexes = self._load_image_set_index()
    self._image_names = [self.image_path_from_index(index)
                         for index in self.image_indexes]
    self._gt_names = [self.gt_path_from_index(index)
                      for index in self.image_indexes]


  def _load_image_set_index(self):
    image_set_file = os.path.join(self._data_dir, 'ImageSets', 'Segmentation',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    return image_index

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_dir, 'JPEGImages',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_path_from_index(self, index):
    """
    Construct and image segmentation ground truth path from image's "index" identifier
    """
    gt_path = os.path.join(self._data_dir, '', 'SegmentationClass',
                           index + '.png')
    assert os.path.exists(gt_path), \
      'Path does not exist: {}'.format(gt_path)
    return gt_path

  @property
  def image_names(self):
    return self._image_names

  @property
  def gt_names(self):
    return self._gt_names

  @property
  def image_indexes(self):
    return self._image_indexes

  @property
  def batch_size(self):
    return self._batch_size
