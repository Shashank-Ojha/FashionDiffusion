# Shashank Ojha
#
# Implements a simple diffusion model with a UNet as the backbone.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

## --- Copy over some networks
class SinusoidalPositionEmbeddings(nn.Module):
    """Construct sinusoidal position embeddings. 

    The shape of the embeddings is ()
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """ Forward pass.

        Return shape = (B, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims,
                  labels, kernel_size = 3, downsample=True):
        super().__init__()
        
        self.time_embedding_dims = time_embedding_dims
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)

        self.labels = labels
        if labels:
            self.label_mlp = nn.Linear(1, channels_out)
        
        self.downsample = downsample
        
        if downsample:
            self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size, padding=1)
            self.final = nn.Conv2d(channels_out, channels_out, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(2 * channels_in, channels_out, kernel_size, padding=1)
            self.final = nn.ConvTranspose2d(channels_out, channels_out, 4, 2, 1)
            
        self.bnorm1 = nn.BatchNorm2d(channels_out)
        self.bnorm2 = nn.BatchNorm2d(channels_out)
        
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)
        self.relu = nn.ReLU()

    def forward(self, x, t, **kwargs):
        """ Forward pass.

        x: Tensor, shape = (B, channels_in, H, W)
        t: Tensor, shape = (B, )

        Return shape = (B, channels_out, H_out, W_out)
        """
        # Shape (B, channels_out, H_out, W_out)
        o = self.bnorm1(self.relu(self.conv1(x)))
        o_time = self.relu(self.time_mlp(self.time_embedding(t)))
        o = o + o_time[(..., ) + (None, ) * 2]
        if self.labels:
            label = kwargs.get('labels')
            o_label = self.relu(self.label_mlp(label))
            o = o + o_label[(..., ) + (None, ) * 2]
            
        o = self.bnorm2(self.relu(self.conv2(o)))

        return self.final(o)

class UNet(nn.Module):
    """Construct a UNet model.
    """
    def __init__(self, img_channels = 1, time_embedding_dims = 128, labels = False,
                  sequence_channels = (64, 128, 256, 512, 1024)):
        super().__init__()
        self.time_embedding_dims = time_embedding_dims
        
        self.downsampling = nn.ModuleList([
            Block(channels_in, channels_out, time_embedding_dims, labels) 
            for channels_in, channels_out in 
            zip(sequence_channels, sequence_channels[1:])
          ])
        
        self.upsampling = nn.ModuleList([
            Block(channels_in, channels_out, time_embedding_dims, labels,downsample=False)
            for channels_in, channels_out in 
            zip(sequence_channels[::-1], sequence_channels[::-1][1:])
          ])
        
        # Convolution takes an image with shape (B, in_channels, H, W) and
        # outputs an image with shape (B, out_channels, H_out, W_out).
        # The first 3 args to this are (in_channels, out_channels, kernel_size)
        # More info at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv1 = nn.Conv2d(img_channels, sequence_channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(sequence_channels[0], img_channels, 1)

    
    def forward(self, x, t, **kwargs):
        """Forward pass.

        Args:
            x: Input image with Shape (B, C, H, W)
            t: Time steps with Shape (B,)

        Return image with Shape (B, C, H, W)
        """
        residuals = []
        # Shape = (B, 3, H_out, W_out)
        o = self.conv1(x)
        for ds in self.downsampling:
            # Shape = (B, 3, H_out/2, W_out/2)
            o = ds(o, t, **kwargs)
            residuals.append(o)

        for us, res in zip(self.upsampling, reversed(residuals)):
            o = us(torch.cat((o, res), dim=1), t, **kwargs)
            
        return self.conv2(o)

## -------------------------

# Config parameters for the Diffusion Model class below.
@dataclass
class DiffusionConfig:
  start_schedule: float
  end_schedule: float
  time_steps: int


# Diffusion Model model.
class DiffusionModel(torch.nn.Module):
  def __init__(self, config) -> None:
    """Initializes the Model."""
    super().__init__()
    self.config = config

    """
    if 
        betas = [0.1, 0.2, 0.3, ...]
    then
        alphas = [0.9, 0.8, 0.7, ...]
        alphas_cumprod = [0.9, 0.9 * 0.8, 0.9 * 0.8, * 0.7, ...]
    """ 
    # Shape = (time_steps,)
    self.betas = torch.linspace(config.start_schedule, config.end_schedule, config.time_steps)
    self.alphas = 1 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

  def destroy(self, x_0, t, device):
    """Returns the noisy image at step t + the sampled noise.

    x_0: tensor with shape (B, C, H, W)
    t: tensor with shape (B,)

    The noisy image is the image that has gone to t steps of diffusion.
    q(x_t | x_0) = N(x_t ; sqrt(a_bar_t) * x_0, (1-a_bar_t_) I)  

    Note to sample from a normal distribution, N(mean, var), you can sample
    noise ~ N(0, 1) and then do mean + sqrt(var) * noise.
    """
    assert x_0.shape[0] == t.shape[0], f'x_0 has batch size {x_0.shape[0]}, but t has batch size {t.shape[0]}'
    (batch_size, num_channels, height, width) = x_0.shape

    # Shape = (batch_size, num_channels, height, width)
    noise = torch.randn_like(x_0)

    # Shape = (batch_size, 1, 1, 1)
    self.alphas_cumprod = self.alphas_cumprod.to(device)
    a_bar_t = torch.gather(self.alphas_cumprod, 0, t).reshape(batch_size, 1, 1, 1)

    # Shape = (batch_size, num_channels, height, width)
    mean = torch.sqrt(a_bar_t) * x_0
    std_dev = torch.sqrt(1 - a_bar_t)

    return mean + std_dev * noise, noise


  def predict_noise(self, noisy_images, t):
    """Returns the predicted noise that was added to the original image
    to get the noisy image at step t.

    noisy_images: tensor with shape (B, C, H, W)
    t: tensor with shape (B,)
    """
