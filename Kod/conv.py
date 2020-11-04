import torch
from torch import nn
from torch.nn import functional



class ConvolutionalModel(nn.Module):
  """A simple model for generating a convolutional neural network.
  It contains two convolutional layers, two pooling layers, and three 
  fully connected layers. In constructor you can specify parameters
  specific to your training set.
  """
  def __init__(self, in_channels, conv1_width, fc1_width, class_count):
    super(ConvolutionalModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(conv1_width, conv1_width * 2, kernel_size=5, stride=1, padding=2, bias=True)
    self.fc1 = nn.Linear(fc1_width * 16, fc1_width * 2, bias=True)
    self.fc2 = nn.Linear(fc1_width * 2, fc1_width, bias=True)
    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

    # parametri su već inicijalizirani pozivima Conv2d i Linear
    # ali možemo ih drugačije inicijalizirati
  
  def forward(self, x):
    """Forward propagation for training of the neural network.
    Classifies the given batch of input tensors.
    Parameters
    ----------
    self: ConvolutionalModel
      For calling the needed layers.
    x: torch.Tensor
      Batch of tensors for classification.
    Returns
    -------
    logits
      Predicitons of classes for the given batch of tensors
    """
    h = self.conv1(x)
    h = functional.relu(h)  
    h = self.pool1(h)

    h = self.conv2(h)
    h = functional.relu(h)
    h = self.pool1(h)

    h = h.view(h.shape[0], -1)

    #print(h.size())

    h = self.fc1(h)
    h = functional.relu(h)

    h = self.fc2(h)
    h = functional.relu(h)
    logits = self.fc_logits(h)
    return logits

