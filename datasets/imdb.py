import os
import numpy as np
import cv2

class ImageDataset(object):
  def __init__(self, name, datadir, batch_size, im_processor,
               processes=3, shuffle=True):
    self._name = name
    self._data_dir = datadir
    self._batch_size = batch_size

    self._epoch = -1

    self._image_indexes = []
    self._image_names = []
    self._gt_names = []

    self.config = {}

    self._shuffle = shuffle
    self._pool_processes = processes
    self.pool = Pool(self._pool_processes)
    self.gen = None
    self._im_processor = im_processor

  def next_batch(self, size_index):
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
                                  ([self.image_names[i],
                                    self.gt_names[i]] for i in indexes),
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
    raise NotImplementedError

  @property
  def image_names(self):
    return self._image_names

  @property
  def image_indexes(self):
    return self._image_indexes
