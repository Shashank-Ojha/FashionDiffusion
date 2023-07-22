# Shashank Ojha
#
# Implements a simple diffusion model with a UNet as the backbone.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

class SinusoidalPositionEmbeddings(nn.Module):
    """Construct sinusoidal position embeddings of shape (B, dim)."""
    def __init__(self, dim):
        """ Constructor.
        Args:
            dim (int): the number of dimensions
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """ Forward pass.
        Args:
            time: Tensor with shape = (B, )

        Return shape = (B, dim)
        """
        device = time.device
        # Scalars.
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        # Shape = (dim/2, )
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Shape = (B, 1) * (1, dim/2) == (B, dim/2)
        embeddings = time[:, None] * embeddings[None, :]
        # Shape = (B, dim)
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class Block(nn.Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims,
                  labels, kernel_size = 3, downsample=True):
        """ Block Constructor.

        If we are upsampling, then 

        Args
            channels_in (int): number of input channels
            channels_out (int): number of output channels
            time_embedding_dims (int): number of time embedding dimensions
            labels (bool): whether to use labels
            kernel_size (int): kernel size
            downsample (bool): whether to downsample or upsample
        """
        super().__init__()
        
        self.time_embedding_dims = time_embedding_dims
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)

        self.labels = labels
        if labels:
            self.label_mlp = nn.Linear(1, channels_out)
        
        self.downsample = downsample
        
        if downsample:
            # With the default kernel size, this convolution preserves the 
            # H and W of the image.
            self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size, padding=1)
            # Halves the H and W of the image.
            self.final = nn.Conv2d(channels_out, channels_out, 4, 2, 1)
        else:
            # With the default kernel size, this convolution preserves the 
            # H and W of the image. The reason there is a 2 * channels_in here
            # is because the user is expected to concat the corresponding
            # residual to the image which doubles the number of channels.
            self.conv1 = nn.Conv2d(2 * channels_in, channels_out, kernel_size, padding=1)
            # Doubles the H and W of the image.
            self.final = nn.ConvTranspose2d(channels_out, channels_out, 4, 2, 1)
            
        self.bnorm1 = nn.BatchNorm2d(channels_out)
        self.bnorm2 = nn.BatchNorm2d(channels_out)
        
        # Preserves shape of image.
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)
        self.relu = nn.ReLU()

    def forward(self, x, t, **kwargs):
        """ Forward pass.

        x: Tensor, shape = (B, channels_in, H, W)
        t: Tensor, shape = (B, )
        kwargs: May contains 'labels' which has shape (B,)

        Return shape = (B, channels_out, H_out, W_out), where
        H_out = H // 2 if downsample else 2 * H if we upsample.
        """
        # Shape (B, channels_out, H_out, W_out)
        o = self.bnorm1(self.relu(self.conv1(x)))
        # Shape (B, time_embedding_dims) -> (B, channels_out)
        o_time = self.relu(self.time_mlp(self.time_embedding(t)))
        # Shape (B, channels_out, H_out, W_out)
        o = o + o_time[(..., ) + (None, ) * 2]
        if self.labels:
            # Shape (B, 1)
            label = kwargs.get('labels')
            # Shape (B, channels_out)
            o_label = self.relu(self.label_mlp(label))
            # Shape (B, channels_out, H_out, W_out)
            o = o + o_label[(..., ) + (None, ) * 2]

        # Shape (B, channels_out, H_out, W_out)
        o = self.bnorm2(self.relu(self.conv2(o)))
        # Shape (B, channels_out, H_out, W_out)
        return self.final(o)

class UNet(nn.Module):
    """Construct a UNet model."""
    def __init__(self, img_channels = 1, time_embedding_dims = 128, labels = False,
                  sequence_channels = (64, 128, 256, 512, 1024)):
        """Constructor.

        Args:
            img_channels (int): number of input channels
            time_embedding_dims (int): number of time embedding dimensions
            labels (bool): whether to use labels
            sequence_channels (int): number of channels in each convolutional
            block.
        """
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
        # Both of these conv's preserve the H and W of the image.
        self.conv1 = nn.Conv2d(img_channels, sequence_channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(sequence_channels[0], img_channels, 1)

    def forward(self, x, t, **kwargs):
        """Forward pass.

        Args:
            x: Input image with Shape (B, C, H, W)
            t: Time steps with Shape (B,)

        The H and W dimensions change at the following schedule with the 
        number of channels for our image of 32 by 32 with 1 channel:
        Channels  |    H and  W
        1                 32
        64                32
        128               16
        256               8
        512               4
        1024              2
        512               4
        256               8
        128               16
        64                32 
        1                 32     

        Returns image with Shape (B, C, H, W)
        """
        residuals = []
        # Shape = (B, sequence_channels[0], H, W)
        o = self.conv1(x)
        for ds in self.downsampling:
            # Shape = (B, sequence_channels[i], H_in/2, W_in/2)
            o = ds(o, t, **kwargs)
            residuals.append(o)

        for us, res in zip(self.upsampling, reversed(residuals)):
            # Shape = (B, sequence_channels[-i], 2*H_in, 2*W_in)
            o = us(torch.cat((o, res), dim=1), t, **kwargs)

        # -- Note that o is back (B, sequence_channels[0], H, W)
        
        # Shape = (B, C, H, W) 
        return self.conv2(o)

# Config parameters for the Diffusion Model class below.
@dataclass
class DiffusionConfig:
  # The fraction of the image to distort with noise at the first time step of
  # diffusion.
  start_schedule: float
  # The fraction of the image to distort with noise at the last time step of
  # diffusion.
  end_schedule: float
  # The number of time steps over which to diffuse the image.
  time_steps: int

# Diffusion Model model.
class DiffusionModel(torch.nn.Module):
  def __init__(self, config) -> None:
    """Initializes the Model.

    Args:
        config: DiffusionConfig

    Betas are the fraction of the image to distort at each time step. So if
    betas[t] = x, then x% of the image at time t is distorted to generate
    image t+1.

    Alphas are 1-betas. And alpha_cumprod are the cumulative product of alphas.
    """
    super().__init__()
    # Shape = (time_steps,)
    self.betas = torch.linspace(config.start_schedule, config.end_schedule, config.time_steps)
    self.alphas = 1 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

  def destroy(self, x_0, t, device):
    """Returns the noisy image at step t + the sampled noise.

    x_0: tensor with shape (B, C, H, W)
    t: tensor with shape (B,)

    The noisy image is the image that has gone to t steps of diffusion.
    q(x_t | x_0) = N(x_t ; sqrt(a_bar_t) * x_0, sqrt(1-a_bar_t) I)  

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

  def denoise(self, noisy_images, t, model, device, **kwargs):
    """Returns the 1 step denoised image from the noisy_images. 

    noisy_images: tensor with shape (B, C, H, W)
    t: tensor with shape (B,)
    model: This is a model that takes in |noisy_images| and the corresponding
    |t| and predicts the noise.

    It's assumed noisy_images contains the image x_t and t is the corresponding
    time step for x_t.  This function returns x_(t-1).

    p(x_(t-1) | x_t) = 
       N(x_(t-1) ; 1/(sqrt(a_t)) (x_t - (1-a_t)/(sqrt(1-a_bar_t)) * model(x_t, t)),
                   sqrt(betas_t) I)  ss
    """
    assert noisy_images.shape[0] == t.shape[0], f'noisy_images has batch size {noisy_images.shape[0]}, but t has batch size {t.shape[0]}'
    (batch_size, num_channels, height, width) = noisy_images.shape

    self.betas = self.betas.to(device)
    self.alphas = self.alphas.to(device)
    self.alphas_cumprod = self.alphas_cumprod.to(device)

    # Shape = (batch_size, 1, 1, 1)
    alphas_t = torch.gather(self.alphas, 0, t).reshape(batch_size, 1, 1, 1)
    a_bar_t = torch.gather(self.alphas_cumprod, 0, t).reshape(batch_size, 1, 1, 1)
    betas_t = torch.gather(self.betas, 0, t).reshape(batch_size, 1, 1, 1)

    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - a_bar_t)
    sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas_t)
    # Shape = (batch_size, num_channels, height, width)
    pred_noise = model(noisy_images, t, **kwargs)
    mean = sqrt_recip_alphas_t * (noisy_images - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

    # Shape = (batch_size, 1, 1, 1)
    mask = (t == 0).float().reshape(batch_size, 1, 1, 1)
    std_dev = torch.sqrt(betas_t)
    # Shape = (batch_size, num_channels, height, width)
    noise = torch.randn_like(noisy_images) * mask
    return mean + std_dev * noise

