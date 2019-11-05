# %load student_code.py
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math
from utils import resize_image
import custom_transforms as transforms

from model import MultiFCN

#################################################################################
# You will need to fill in the missing code in this file
#################################################################################


#################################################################################
# Part I: Understanding Convolutions
#################################################################################
class CustomConv2DFunction(Function):

  @staticmethod
  def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
    """
    Forward propagation of convolution operation.
    We only consider square filters with equal stride/padding in width and height!

    Args:
      input_feats: input feature map of size N * C_i * H * W
      weight: filter weight of size C_o * C_i * K * K
      bias: (optional) filter bias of size C_o
      stride: (int, optional) stride for the convolution. Default: 1
      padding: (int, optional) Zero-padding added to both sides of the input. Default: 0

    Outputs:
      output: responses of the convolution  w*x+b

    """
    # sanity check
    assert weight.size(2) == weight.size(3)
    assert input_feats.size(1) == weight.size(1)
    assert isinstance(stride, int) and (stride > 0)
    assert isinstance(padding, int) and (padding >= 0)

    # save the conv params
    kernel_size = weight.size(2)
    ctx.stride = stride
    ctx.padding = padding
    ctx.input_height = input_feats.size(2)
    ctx.input_width = input_feats.size(3)

    # make sure this is a valid convolution
    assert kernel_size <= (input_feats.size(2) + 2 * padding)
    assert kernel_size <= (input_feats.size(3) + 2 * padding)

    #################################################################################
    output_height = int(np.floor((input_feats.size(2) + 2.0 * padding - kernel_size)/stride) + 1)
    output_width = int(np.floor((input_feats.size(3) + 2.0 * padding - kernel_size)/stride) + 1)
    weight=weight.double()
    input_unfold = unfold(input_feats, (kernel_size, kernel_size), padding=padding, stride=stride).double()
    output_unfold = input_unfold.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2) + bias.view(1,bias.size(0),1)
    output = fold(output_unfold, (output_height, output_width), (1, 1))
    input_unfold = input_unfold.view(input_feats.size(0), input_feats.size(1), kernel_size*kernel_size, -1)
    H = output_height
    W = output_width
    input_unfold = input_unfold.permute(0, 3, 1, 2).reshape(input_feats.size(0)*H*W, input_feats.size(1)*kernel_size*kernel_size).double()
    ctx.save_for_backward(input_unfold, weight, bias)
#    output = torch.nn.functional.conv2d(inputf.double(), weight.double(), padding=padding, stride=stride)
#    print((torch.nn.functional.conv2d(inputf.double(), weight.double(), padding=padding, stride=stride).double() - output).abs().max())
    
    return output.double()

  @staticmethod
  def backward(ctx, grad_output):
    """
    Backward propagation of convolution operation

    Args:
      grad_output: gradients of the outputs

    Outputs:
      grad_input: gradients of the input features
      grad_weight: gradients of the convolution weight
      grad_bias: gradients of the bias term

    """
    # unpack tensors and initialize the grads
    feature_matrix, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None

    # recover the conv params
    kernel_size = weight.size(2)
    stride = ctx.stride
    padding = ctx.padding
    input_height = ctx.input_height
    input_width = ctx.input_width
    C_in = weight.size(1)
    N, C_out, OH, OW = grad_output.shape
    kernel_matrix = weight.view(C_out, -1).permute(1, 0).double()

    #################################################################################
    # Fill in the code here   
    # grad_output -- (N,C_out,OH,OW)
    
    grad_output_matrix = grad_output.permute(0,2,3,1).reshape(N*OH*OW,C_out)
    # grad_ouput -- (N*OH*OW,C_out)
    # feature_matrix  shape is  [N*OH*OW,C_in*K*K]  
    grad_weight = feature_matrix.permute(1,0).matmul(grad_output_matrix)  ##[C_in*K*K,C_out]
    
    grad_weight = grad_weight.view(C_in,kernel_size,kernel_size,C_out)
    grad_weight = grad_weight.permute(3,0,1,2) ## [C_out,C_in,K,K]
    # kernel_matrix --[C_in*K*K,C_out]
    grad_input_ = grad_output_matrix.matmul(kernel_matrix.permute(1,0)) # [N*OH*OW,C_in*K*K]
    grad_input_ = grad_input_.view(N,OH,OW,C_in,kernel_size,kernel_size) #[N,OH,OW,C_in,K,K]
    grad_input = torch.zeros(N,C_in,input_height+padding*2,input_width+padding*2).double() #[N,C_in,H,W]  (consider  padding)
    grad_input = Variable(grad_input)
    if torch.cuda.is_available():
        grad_input = grad_input.cuda()
    for y in range(OH):
        y_left = y*stride
        y_left_max = y_left + kernel_size 
        for x in range(OW):
            x_left = x*stride
            x_left_max = x_left + kernel_size 
            grad_input[:,:,y_left:y_left_max,x_left:x_left_max]= grad_input[:,:,y_left:y_left_max,x_left:x_left_max] + grad_input_[:,y,x,:,:,:]
    
    grad_input =  grad_input[:,:,padding:padding+input_height,padding:padding+input_width]

    if bias is not None and ctx.needs_input_grad[2]:
      # compute the gradients w.r.t. bias (if any)
      grad_bias = grad_output.sum((0,2,3))

    return grad_input, grad_weight, grad_bias, None, None

custom_conv2d = CustomConv2DFunction.apply

class CustomConv2d(Module):
  """
  The same interface as torch.nn.Conv2D
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
         padding=0, dilation=1, groups=1, bias=True):
    super(CustomConv2d, self).__init__()
    assert isinstance(kernel_size, int), "We only support squared filters"
    assert isinstance(stride, int), "We only support equal stride"
    assert isinstance(padding, int), "We only support equal padding"
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    # not used (for compatibility)
    self.dilation = dilation
    self.groups = groups

    # register weight and bias as parameters
    self.weight = nn.Parameter(torch.Tensor(
      out_channels, in_channels, kernel_size, kernel_size))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_channels))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
  	# initialization using Kaiming uniform
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in)
      nn.init.uniform_(self.bias, -bound, bound)

  def forward(self, input):
    # call our custom conv2d op
    return custom_conv2d(input, self.weight, self.bias, self.stride, self.padding)

  def extra_repr(self):
    s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
         ', stride={stride}, padding={padding}')
    if self.bias is None:
      s += ', bias=False'
    return s.format(**self.__dict__)

#################################################################################
# Part II: Design and train a network
#################################################################################
class SimpleNet(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(SimpleNet, self).__init__()
    # you can start from here and create a better model
    self.features = nn.Sequential(
      # conv1 block: conv 7x7
      conv_op(3, 64, kernel_size=7, stride=2, padding=3),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv2 block: simple bottleneck
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv3 block: simple bottleneck
      conv_op(256, 128, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(128, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(128, 512, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
    )
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def reset_parameters(self):
    # init all params
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.consintat_(m.bias, 0.0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

  def forward(self, x):
    # you can implement adversarial training here
    # if self.training:
    #   # generate adversarial sample based on x
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = x.float()
    x = self.fc(x)
    return x

class AdvSimpleNet(SimpleNet):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super().__init__(conv_op=conv_op, num_classes=num_classes)

  def forward(self, x):
    # you can implement adversarial training here
    if self.training:
      step_size = 0.01
      num_steps = 5
      epsilon = 0.1
      out = x.clone()
      out.requires_grad = True
      x.requires_grad = False
      for _ in range(num_steps):
        t = self.features(out)
        t = self.avgpool(t)
        t = t.view(out.size(0), -1)
        ypred = self.fc(t)

        target, clf = torch.min(ypred, 1)

        backprop = torch.cuda.FloatTensor(target.size()).fill_(1)
        target.backward(backprop)

        gradient = out.grad.data
        gradient_sign = torch.sign(out.grad.data)

        out.data = out.data + step_size * gradient_sign
        out.data = torch.max(torch.min(out.data, x + epsilon), x - epsilon)
        out.grad.zero_()
        x = out

    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

# change this to your model!
from model import MultiFCN
# default_model = SimpleNet
default_model = MultiFCN

# define data augmentation used for training, you can tweak things if you want
def get_train_transforms(normalize):
  train_transforms = []
  train_transforms.append(transforms.Scale(160))
  train_transforms.append(transforms.RandomHorizontalFlip())
  train_transforms.append(transforms.RandomColor(0.15))
  train_transforms.append(transforms.RandomRotate(15))
  train_transforms.append(transforms.RandomSizedCrop(128))
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  return train_transforms

#################################################################################
# Part III: Adversarial samples and Attention
#################################################################################
class PGDAttack(object):
  def __init__(self, loss_fn, num_steps=10, step_size=0.01, epsilon=0.1):
    """
    Attack a network by Project Gradient Descent. The attacker performs
    k steps of gradient descent of step size a, while always staying
    within the range of epsilon (under l infinity norm) from the input image.

    Args:
      loss_fn: loss function used for the attack
      num_steps: (int) number of steps for PGD
      step_size: (float) step size of PGD
      epsilon: (float) the range of acceptable samples
               for our normalization, 0.1 ~ 6 pixel levels
    """
    self.loss_fn = loss_fn
    self.num_steps = num_steps
    self.step_size = step_size
    self.epsilon = epsilon

  def perturb(self, model, input):
    """
    Given input image X (torch tensor), return an adversarial sample
    (torch tensor) using PGD of the least confident label.

    See https://openreview.net/pdf?id=rJzIBfZAb

    Args:
      model: (nn.module) network to attack
      input: (torch tensor) input image of size N * C * H * W

    Outputs:
      output: (torch tensor) an adversarial sample of the given network
    """
    # clone the input tensor and disable the gradients
    output = input.clone()
    output.requires_grad = True
    input.requires_grad = False

    # loop over the number of steps
    for _ in range(self.num_steps):
      #################################################################################
      # Fill in the code here
      ypred = model(output)
      target, clf = torch.min(ypred, 1)

      backprop = torch.cuda.FloatTensor(target.size()).fill_(1)
      target.backward(backprop)

      gradient = output.grad.data
      gradient_sign = torch.sign(output.grad.data)

      output.data = output.data + self.step_size * gradient_sign
      output.data = torch.max(torch.min(output.data, input + self.epsilon), input - self.epsilon)
      output.grad.zero_()
      #################################################################################

    return output

default_attack = PGDAttack


class GradAttention(object):
  def __init__(self, loss_fn):
    """
    Visualize a network's decision using gradients

    Args:
      loss_fn: loss function used for the attack
    """
    self.loss_fn = loss_fn

  def explain(self, model, input):
    """
    Given input image X (torch tensor), return a saliency map
    (torch tensor) by computing the max of abs values of the gradients
    given by the predicted label

    See https://arxiv.org/pdf/1312.6034.pdf

    Args:
      model: (nn.module) network to attack
      input: (torch tensor) input image of size N * C * H * W

    Outputs:
      output: (torch tensor) a saliency map of size N * 1 * H * W
    """
    # make sure input receive grads
    input.requires_grad = True
    if input.grad is not None:
      input.grad.zero_()

    #################################################################################
    # Fill in the code here
    ypred = model(input)
    value, index = torch.max(ypred, 0)
    ypred = ypred.gather(1, index.view(-1, 1)).squeeze()
    ypred.backward(torch.cuda.FloatTensor(ypred.size()).fill_(1))
    mag = input.grad
    mag = mag.abs()
    mag, _ = torch.max(mag, dim=1)
    mag = torch.reshape(mag, (mag.shape[0], 1, mag.shape[1], mag.shape[2]))
    output = mag
    #################################################################################

    return output

default_attention = GradAttention

def vis_grad_attention(input, vis_alpha=2.0, n_rows=10, vis_output=None):
  """
  Given input image X (torch tensor) and a saliency map
  (torch tensor), compose the visualziations

  Args:
    input: (torch tensor) input image of size N * C * H * W
    output: (torch tensor) input map of size N * 1 * H * W

  Outputs:
    output: (torch tensor) visualizations of size 3 * HH * WW
  """
  # concat all images into a big picture
  input_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
  if vis_output is not None:
    output_maps = make_grid(vis_output.cpu(), nrow=n_rows, normalize=True)

    # somewhat awkward in PyTorch
    # add attention to R channel
    mask = torch.zeros_like(output_maps[0, :, :]) + 0.5
    mask = (output_maps[0, :, :] > vis_alpha * output_maps[0,:,:].mean())
    mask = mask.float()
    input_imgs[0,:,:] = torch.max(input_imgs[0,:,:], mask)
  output = input_imgs
  return output

default_visfunction = vis_grad_attention
