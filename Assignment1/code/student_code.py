# %load ../code/student_code_ind4.py
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import random
import glob
import os
import numpy as np

import cv2
import numbers
import collections

import torch
from torch.utils import data

from utils import resize_image, load_image

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# These are helper functions or functions for demonstration
# You won't need to modify them
#################################################################################

class Compose(object):
  """Composes several transforms together.

  Args:
      transforms (list of ``Transform`` objects): list of transforms to compose.

  Example:
      >>> Compose([
      >>>     Scale(320),
      >>>     RandomSizedCrop(224),
      >>> ])
  """
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img):
    for t in self.transforms:
      img = t(img)
    return img

  def __repr__(self):
    repr_str = ""
    for t in self.transforms:
      repr_str += t.__repr__() + '\n'
    return repr_str

class RandomHorizontalFlip(object):
  """Horizontally flip the given numpy array randomly
     (with a probability of 0.5).
  """
  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be flipped.

    Returns:
        numpy array: Randomly flipped image
    """
    if random.random() < 0.5:
      img = cv2.flip(img, 1)
      return img
    return img

  def __repr__(self):
    return "Random Horizontal Flip"

#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
  """Rescale the input numpy array to the given size.

  Args:
      size (sequence or int): Desired output size. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          smaller edge of the image will be matched to this number.
          i.e, if height > width, then image will be rescaled to
          (size, size * height / width)

      interpolations (list of int, optional): Desired interpolation.
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      Pass None during testing: always use CV2.INTER_LINEAR
  """
  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
    assert (isinstance(size, int)
            or (isinstance(size, collections.Iterable)
                and len(size) == 2)
           )
    self.size = size
    # use bilinear if interpolation is not specified
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be scaled.

    Returns:
        numpy array: Rescaled image
    """
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]

    # scale the image
    if isinstance(self.size, int):
      h, w, c = img.shape
      ratio = self.size/min(h,w)
      img = resize_image(img, (int(w*ratio),int(h*ratio)), interpolation)
      return img
    else:
      img = resize_image(img, self.size, interpolation)
      return img

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])

class RandomSizedCrop(object):
  """Crop the given numpy array to random area and aspect ratio.

  A crop of random area of the original size and a random aspect ratio
  of the original aspect ratio is made. This crop is finally resized to given size.
  This is widely used as data augmentation for training image classification models

  Args:
      size (sequence or int): size of target image. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          output size will be (size, size).
      interpolations (list of int, optional): Desired interpolation.
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      area_range (list of int): range of the areas to sample from
      ratio_range (list of int): range of aspect ratio to sample from
      num_trials (int): number of sampling trials
  """

  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS,
               area_range=(0.25, 1.0), ratio_range=(0.8, 1.2), num_trials=10):
    self.size = size
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations
    self.num_trials = int(num_trials)
    self.area_range = area_range
    self.ratio_range = ratio_range

  def __call__(self, img):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]

    for attempt in range(self.num_trials):

      # sample target area / aspect ratio from area range and ratio range
      area = img.shape[0] * img.shape[1]
      target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
      aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

      #################################################################################
      # Fill in the code here
      #################################################################################
      # compute the width and height
      # note that there are two possibilities
      # crop the image and resize to output size
      
      # h*w = (w*aspect_ratio)*w = target_area
      hw_list = []
      width = (target_area/aspect_ratio)**0.5
      height = width*aspect_ratio
      height, width = int(height), int(width)
      
      # two possibilities:
      hw_list = [(height, width), (width,height)]
      for h, w in hw_list:
        # find a suitable crop area and aspect_ratio
        if h < img.shape[0] and w < img.shape[1]:
          randX = random.sample(range(0, img.shape[0]-h), 1)[0]
          randY = random.sample(range(0, img.shape[1]-w), 1)[0]
          if isinstance(self.size, int):
            img = resize_image(img[randX:(randX+h), randY:(randY+w)],
                                (self.size, self.size),
                                interpolation)
          else:
            img = resize_image(img[randX:(randX+h), randY:(randY+w)], 
                                self.size, 
                                interpolation)
          return img

    # Fall back
    if isinstance(self.size, int):
      im_scale = Scale(self.size, interpolations=self.interpolations)
      img = im_scale(img)
      #################################################################################
      # Fill in the code here
      #################################################################################
      # with a square sized output, the default is to crop the patch in the center
      # (after all trials fail)
      imgH, imgW = img.shape[0], img.shape[1]
      if imgH > imgW:
        offset = imgH - imgW
        img = img[offset//2:-(offset-offset//2),:]
      else:
        offset = imgW - imgH
        img = img[:, offset//2:-(offset-offset//2)]
      return img
    else:
      # with a pre-specified output size, the default crop is the image itself
      im_scale = Scale(self.size, interpolations=self.interpolations)
      img = im_scale(img)
      return img

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Random Crop" + \
           "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}%; Ratio {:.2f} - {:.2f}%]".format(
            target_size[0], target_size[1],
            self.area_range[0], self.area_range[1],
            self.ratio_range[0], self.ratio_range[1])


class RandomColor(object):
  """Perturb color channels of a given image
  Sample alpha in the range of (-r, r) and multiply 1 + alpha to a color channel.
  The sampling is done independently for each channel.

  Args:
      color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
  """
  def __init__(self, color_range):
    self.color_range = color_range

  def __call__(self, img):
    #################################################################################
    # Fill in the code here
    #################################################################################
    for c in range(img.shape[2]):
      alpha = random.uniform(-self.color_range, self.color_range)
      for i in range(img.shape[0]):
        for j in range(img.shape[1]):
          img[i,j,c]= np.clip(img[i,j,c]*(1+alpha), 0, 255).astype(np.uint8)
    return img

  def __repr__(self):
    return "Random Color [Range {:.2f} - {:.2f}%]".format(
            1-self.color_range, 1+self.color_range)

class FastRandomColor(RandomColor):
  """
  A fast implementation of RandomColor using pre-calculated lookup table.
  """
  def __init__(self, color_range):
    super().__init__(color_range)

  def __call__(self, img):
    for c in range(img.shape[2]):
      alpha = random.uniform(-self.color_range, self.color_range)
      color_dict = {}
      for i in range(256):
        color_dict[i] = np.clip(i*(1+alpha), 0, 255).astype(np.uint8)
      for i in range(img.shape[0]):
        for j in range(img.shape[1]):
          img[i,j,c]=color_dict[img[i,j,c]]
    return img

class FasterRandomColor(RandomColor):
  """
  A faster implementation of RandomColor using matrix calculations.
  """
  def __init__(self, color_range):
    super().__init__(color_range)

  def __call__(self, img):
    for c in range(img.shape[2]):
      alpha = random.uniform(-self.color_range, self.color_range)
      img[:,:,c] = np.clip(img[:,:,c].astype(float)*(1+alpha), 0, 255).astype(np.uint8)
    return img


class RandomRotate(object):
  """Rotate the given numpy array (around the image center) by a random degree.

  Args:
      degree_range (float): range of degree (-d ~ +d)
  """
  def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
    self.degree_range = degree_range
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]
    # sample rotation
    degree = random.uniform(-self.degree_range, self.degree_range)
    # ignore small rotations
    if np.abs(degree) <= 1.0:
      return img

    #################################################################################
    # Fill in the code here (Done)
    #################################################################################
    # get the max rectangular within the rotated image
    
    # 2D rotation matrix
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), degree,1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    # to find the max rectangular
    opt = 0
    # to work in grayscale
    arr = np.mean(img,axis=2)
    
    # loop over each row
    for y in range(arr.shape[0]):
      # find the nonzero elements with min-index and max-index
      index = np.where(arr[y,:]>0)[0]
      xmin, xmax = min(index), max(index)
      width = xmax - xmin
      
      # find heights
      height = min(max(np.where(arr[y:, xmin]>0)[0]), max(np.where(arr[y:, xmax]>0)[0]))
      area = height*width
      if area > opt:
        opt = area
        r, c, h, w = y, xmin, height, width
    return img[r:(r+h), c:(c+w)]

  def __repr__(self):
    return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range)

#################################################################################
# Additional helper functions
#################################################################################
class ToTensor(object):
  """Convert a ``numpy.ndarray`` image to tensor.
  Converts a numpy.ndarray (H x W x C) image in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  """
  def __call__(self, img):
    assert isinstance(img, np.ndarray)
    # convert image to tensor
    assert (img.ndim > 1) and (img.ndim <= 3)
    if img.ndim == 2:
      img = img[:, :, None]
      tensor_img = torch.from_numpy(np.ascontiguousarray(
        img.transpose((2, 0, 1))))
    if img.ndim == 3:
      tensor_img = torch.from_numpy(np.ascontiguousarray(
        img.transpose((2, 0, 1))))
    # backward compatibility
    if isinstance(tensor_img, torch.ByteTensor):
      return tensor_img.float().div(255.0)
    else:
      return tensor_img

class SimpleDataset(data.Dataset):
  """
  A simple dataset
  """
  def __init__(self, root_folder, file_ext, transforms=None):
    # root folder, split
    self.root_folder = root_folder
    self.transforms = transforms
    self.file_ext = file_ext

    # load all labels
    file_list = glob.glob(os.path.join(root_folder, '*.{:s}'.format(file_ext)))
    self.file_list = file_list

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, index):
    # load img and label (from file name)
    filename = self.file_list[index]
    img = load_image(filename)
    label = os.path.basename(filename)
    label = label.rstrip('.{:s}'.format(self.file_ext))
    # apply data augmentation
    if self.transforms is not None:
      img  = self.transforms(img)
    return img, label