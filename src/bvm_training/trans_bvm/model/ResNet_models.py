import torch
import torch.nn as nn
import torchvision.models as models
from model.ResNet import B2_ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn import Parameter, Softmax
import torch.nn.functional as F
from model.HolisticAttention import HA
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
import numpy as np
from typing import List


class Descriptor(nn.Module):
    """
    Used to fuse RGB images with corresponding spatial prior images, outputting single-channel feature maps
    Focuses on the fusion and compression of 'image features' and 'prior maps'.
    Through up-sampling, down-sampling and multi-layer convolution, the model can process and fuse prior information at different scales, outputting target features.
    The method is simple and efficient, commonly used as a prior encoder or coarse prediction module in VAE/generators.
    """

    def __init__(self, channel):
        super(Descriptor, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # Input (B, 3, H, W) RGB image, downsample, expand channels
        self.sconv1 = nn.Conv2d(3, channel, kernel_size=3, stride=2, padding=1)
        self.sconv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.sconv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.sconv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)

        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.bn3 = nn.BatchNorm2d(512)
        # self.bn4 = nn.BatchNorm2d(1024)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 1024)
        # Multi-step downsampling branch after fusing seg prior
        self.conv_pred = nn.Conv2d(1, channel, 3, 1, 1)
        # Dimension reduction after fusion
        self.conv1 = nn.Conv2d(channel * 2, channel, 3, 2, 1)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(channel, channel, 3, 2, 1)
        self.conv4 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(channel, 1, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)
        self.bn4 = nn.BatchNorm2d(channel)
        # Upsample, 2x enlargement, align with seg prior image
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        """
        Build a classifier module (Classifier_Module) with multi-scale dilated convolution for extracting features with larger receptive fields
        Args:
            block: Classifier_Module
            dilation_series: List of dilation rates for dilated convolution
            padding_series: List of padding rates for dilated convolution
            NoLabels: Number of output channels
            input_channel: Number of input channels
        Returns:
            block: Classifier_Module
        """
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, input, seg):
        """
        Args:
            input: RGB image, shape (B, 3, H, W)
            seg: Prior image, shape (B, 1, H, W)
        Returns:
            x: Output feature map, shape (B, 1, H/16, W/16)
        """
        x1 = self.sconv1(input)
        # # x1 = self.bn1(x1)
        # x1 = self.relu(x1)
        # x2 = self.sconv2(x1)
        # # x2 = self.bn2(x2)
        # x2 = self.relu(x2)
        # x2 = self.maxpool(x2)
        # x3 = self.sconv3(x2)
        # # x3 = self.bn3(x3)
        # x3 = self.relu(x3)
        # x3 = self.maxpool(x3)
        # x4 = self.sconv4(x3)
        # # x4 = self.bn4(x4)
        # x4 = self.relu(x4)
        # x5 = self.layer5(x4)
        int_feat = self.upsample(x1)
        seg_conv = self.conv_pred(seg)
        # Fuse RGB image and prior image
        feature_map = torch.cat((int_feat, seg_conv), 1)
        x = self.conv1(feature_map)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        return x


class Encoder_x(nn.Module):
    """
    VAE prior encoder, only encodes RGB images, outputting latent space distribution
    The role of the prior encoder is to encode input images into latent space distribution for generating initial saliency maps
    """

    def __init__(self, input_channels, channels, latent_size):
        """
        Args:
            input_channels: Number of input image channels
            channels: Number of encoder channels
            latent_size: Dimension of latent space
        """
        super(Encoder_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        # 5 convolutional layers with progressive downsampling
        # Output feature map size: feature_map_size = (input_image_size - kernel_size + 2 * padding) / stride + 1
        # Input channels: input_channels; Output channels: channels
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # Output channels: 2 * channels
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        # Output channels: 4 * channels
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        # Output channels: 8 * channels
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        # Output channels: 8 * channels
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        # 3 fully connected layers for different input image sizes
        # For input image size 256, output latent space distribution size is channels*8*8*8
        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # For input image size 352, output latent space distribution size is channels*8*11*11
        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # For input image size larger than 352, output latent space distribution size is channels*8*14*14
        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        """
        Args:
            input: Input image, shape (B, 3, H, W)
        Returns:
            dist: Latent space distribution, Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1), independent normal distribution N(μ,σ) for each dimension, also called multi-dimensional diagonal Gaussian
            mu: Latent space mean, shape (B, latent_dim)
            logvar: Latent space log variance, shape (B, latent_dim)
        """
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))

        # print(output.size())
        if input.shape[2] == 256:
            # print('************************256********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 8 * 8)

            mu = self.fc1_1(output)  # Mean μ, shape=[B, latent_dim]
            logvar = self.fc2_1(output)  # Log variance log σ²
            dist = Independent(
                Normal(loc=mu, scale=torch.exp(logvar)),  # Base distribution: independent normal distribution N(μ,σ) for each dimension
                1,  # Treat the last dimension as the event dimension to form a multivariate diagonal Gaussian
            )
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        elif input.shape[2] == 352:
            # print('************************352********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 11 * 11)

            mu = self.fc1_2(output)
            logvar = self.fc2_2(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        else:
            # print('************************bigger********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 14 * 14)

            mu = self.fc1_3(output)
            logvar = self.fc2_3(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        # output = output.view(-1, self.channel * 8 * 11 * 11)
        # # print(output.size())
        # # output = self.tanh(output)
        #
        # mu = self.fc1(output)
        # logvar = self.fc2(output)
        # dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        # # print(output.size())
        # # output = self.tanh(output)
        #
        # return dist, mu, logvar


class Encoder_xy(nn.Module):
    """
    VAE posterior encoder, takes RGB image and saliency map, concatenates and encodes to output latent space distribution
    """

    def __init__(self, input_channels, channels, latent_size):
        """
        Args:
            input_channels: Number of input image channels
            channels: Number of encoder channels
            latent_size: Dimension of latent space
        """
        super(Encoder_xy, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        # Five convolutional layers with progressive downsampling
        # Output feature map size: feature_map_size = (input_image_size - kernel_size + 2 * padding) / stride + 1
        # Input channels: input_channels; Output channels: channels
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # Output channels: 2 * channels
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        # Output channels: 4 * channels
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        # Output channels: 8 * channels
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        # Output channels: 8 * channels
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        # Three fully connected layers for different input image sizes
        # For input image size 256, output latent space distribution size is channels*8*8*8
        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # For input image size 352, output latent space distribution size is channels*8*11*11
        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # For input image size larger than 352, output latent space distribution size is channels*8*14*14
        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        """
        Args:
            x: Input image, shape (B, 3, H, W)
        Returns:
            dist: Latent space distribution, Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1), independent normal distribution N(μ,σ) for each dimension, also called multi-dimensional diagonal Gaussian
            mu: Latent space mean, shape (B, latent_dim)
            logvar: Latent space log variance, shape (B, latent_dim)
        """
        output = self.leakyrelu(self.bn1(self.layer1(x)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))
        # print(output.size())
        if x.shape[2] == 256:
            # print('************************256********************')
            # print(x.size())
            output = output.view(-1, self.channel * 8 * 8 * 8)

            mu = self.fc1_1(output)
            logvar = self.fc2_1(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        elif x.shape[2] == 352:
            # print('************************352********************')
            # print(x.size())
            output = output.view(-1, self.channel * 8 * 11 * 11)

            mu = self.fc1_2(output)
            logvar = self.fc2_2(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        else:
            # print('************************bigger********************')
            # print(x.size())
            output = output.view(-1, self.channel * 8 * 14 * 14)

            mu = self.fc1_3(output)
            logvar = self.fc2_3(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar


class Generator(nn.Module):
    """
    Implements a VAE-based saliency map generator (Saliency Generator), with both posterior and prior encoders, and a joint saliency feature decoder.
    Outputs initial/refined saliency maps and KL divergence loss during training, and only the final probability map during inference.
    """

    def __init__(self, channel, latent_dim):
        """
        Args:
            channel: Number of channels
            latent_dim: Latent space dimension
        """
        super(Generator, self).__init__()
        self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)  # Takes RGB image and latent vector, outputs initial/refined saliency maps
        self.xy_encoder = Encoder_xy(
            4, channel, latent_dim
        )  # Takes RGB image and saliency map, concatenates to produce posterior latent distribution
        self.x_encoder = Encoder_x(3, channel, latent_dim)  # Prior encoder, encodes only RGB image

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        """
        Compute KL divergence loss between posterior and prior latent space distributions
        Args:
            posterior_latent_space: Posterior latent space distribution
            prior_latent_space: Prior latent space distribution
        Returns:
            kl_div: KL divergence loss (KL(posterior || prior))
        """
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Sample from latent space distribution using reparameterization trick
        Args:
            mu: Latent space mean ([B, latent_dim])
            logvar: Latent space log variance ([B, latent_dim])
        Returns:
            eps: Sampled latent space vector
        """
        std = logvar.mul(0.5).exp_()  # Standard deviation ([B, latent_dim]), exp_() is in-place
        # eps = Variable(std.data.new(std.size()).normal_())
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, y=None, training=True):
        """
        Forward function
        Args:
            x: Input image ([B, 3, H, W]).
            y: Saliency map ([B, 1, H, W]).
            training: Whether in training mode.
        Returns:
            sal_init_post: Posterior initial saliency map ([B, 1, H, W]).
            sal_ref_post: Posterior refined saliency map ([B, 1, H, W]).
            sal_init_prior: Prior initial saliency map ([B, 1, H, W]).
            sal_ref_prior: Prior refined saliency map ([B, 1, H, W]).
            latent_loss: Mean KL divergence loss, representing the difference between posterior and prior latent space distributions.
            prob_pred: Predicted saliency map ([B, 1, H, W]), only returned in inference mode.
        Process:
            1. In training mode, input RGB image and saliency map, encode to get posterior latent space distribution and initial/refined saliency maps.
            2. In training mode, input RGB image, encode to get prior latent space distribution and initial/refined saliency maps.
            3. In training mode, compute KL divergence loss between posterior and prior latent space distributions.
            4. In inference mode, input RGB image, encode to get prior latent space distribution and predicted saliency map.
        """
        if training:
            # 1. Posterior distribution
            self.posterior, muxy, logvarxy = self.xy_encoder(torch.cat((x, y), 1))
            # 2. Prior distribution
            self.prior, mux, logvarx = self.x_encoder(x)
            # 3. Compute KL divergence loss
            latent_loss = torch.mean(self.kl_divergence(self.posterior, self.prior))
            # 4. Reparameterization sampling
            z_noise_post = self.reparametrize(muxy, logvarxy)
            z_noise_prior = self.reparametrize(mux, logvarx)
            # 5. Decode initial/refined saliency maps via encoder
            self.sal_init_post, self.sal_ref_post = self.sal_encoder(x, z_noise_post)
            self.sal_init_prior, self.sal_ref_prior = self.sal_encoder(x, z_noise_prior)

            self.sal_init_post = F.interpolate(self.sal_init_post, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
            self.sal_ref_post = F.interpolate(self.sal_ref_post, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
            self.sal_init_prior = F.interpolate(self.sal_init_prior, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
            self.sal_ref_prior = F.interpolate(self.sal_ref_prior, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
            return self.sal_init_post, self.sal_ref_post, self.sal_init_prior, self.sal_ref_prior, latent_loss
        else:
            # Inference mode, only output predicted saliency map
            _, mux, logvarx = self.x_encoder(x)
            z_noise = self.reparametrize(mux, logvarx)
            _, self.prob_pred = self.sal_encoder(x, z_noise)
            return self.prob_pred


class CAM_Module(nn.Module):
    """
    Channel Attention Module (CAM)
    --------------------------------
    Implements channel self-attention as in "Dual Attention Network for Scene Segmentation".

    Args
    ----
    in_dim : int
        Number of input feature channels C.

    Forward input
    -------
    x : torch.Tensor
        Feature map of shape (B, C, H, W).

    Returns
    ----
    out : torch.Tensor
        Channel re-weighted feature map, same shape as input.
    attention : torch.Tensor
        Channel attention weights (B, C, C), optional return.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.channel_in = in_dim
        # gamma initialized as 0, learnable during training for attention strength
        self.gamma: Parameter = Parameter(torch.zeros(1))  # Collected in model.parameters(), updated during backprop
        # Softmax normalizes along the last (channel) dimension
        self.softmax = Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()  # B=batch size, C=channels, N=H*W
        proj_query = x.view(B, C, -1)  # (B, C, N), treat all spatial positions of each channel as a vector
        proj_key = proj_query.permute(0, 2, 1)  # (B, N, C), for torch.bmm() similarity matrix

        # Channel similarity matrix (B, C, C)
        energy = torch.bmm(proj_query, proj_key)  # energy[b,i,j]: similarity between channel i and j in sample b

        # Training stability trick: convert energy to "distance" form
        energy_new = (
            torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        )  # For each row (channel j), find max similarity; convert similarity to distance
        attention = self.softmax(
            energy_new
        )  # (B, C, C), most channel "distances" > 0, softmax emphasizes less similar channels, reducing redundancy and highlighting complementary info

        proj_value = proj_query  # (B, C, N)
        out = torch.bmm(attention, proj_value)  # (B, C, N)
        out = out.view(B, C, H, W)  # Restore spatial dims

        out = self.gamma * out + x  # Residual connection
        return out


class PAM_Module(nn.Module):
    """
    Position Attention Module (PAM)
    --------------------------------
    Implements spatial self-attention as in "Dual Attention Network for Scene Segmentation".

    Args
    ----
    in_dim : int
        Number of input feature channels C.

    Forward input
    -------
    x : torch.Tensor
        Feature map of shape (B, C, H, W).

    Returns
    ----
    out : torch.Tensor
        Position re-weighted feature map, same shape as input.
    attention : torch.Tensor
        Spatial attention weights (B, H*W, H*W), optional return.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.channel_in = in_dim

        # query/key 1x1 conv reduces to C/8 channels for efficiency
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv: nn.Conv2d = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        # value keeps original channel count
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.gamma: Parameter = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        N = H * W

        # (B, N, C/8)
        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)
        # (B, C/8, N)
        proj_key = self.key_conv(x).view(B, -1, N)

        # Spatial similarity (B, N, N)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # Value mapping (B, C, N)
        proj_value = self.value_conv(x).view(B, C, N)

        # Weighted sum and restore shape
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(B, C, H, W)

        out = self.gamma * out + x  # Residual connection
        return out


class Classifier_Module(nn.Module):
    """
    Classifier Module with Multi-scale Dilation Convolutions
    Multi-scale dilated convolution classifier module: uses several convolution layers with the same input/output channels but different dilation and padding rates to extract multi-scale features and enlarge the receptive field.
    Each branch processes the input feature map in parallel, and the outputs are summed to obtain the final output with multi-scale context information.
    """

    def __init__(self, dilation_series, padding_series, NumLabels, input_channel):
        """
        Args:
            dilation_series: Dilation rates for dilated convolution
            padding_series: Padding rates for dilated convolution
            NumLabels: Number of output channels
            input_channel: Number of input channels
        """
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()  # type: List[nn.Conv2d]
        # Add multiple conv layers to the list, each with different dilation and padding
        for d, p in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NumLabels, kernel_size=3, stride=1, padding=p, dilation=d, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Args:
            x: Input feature map, shape (B, C, H, W), B is batch size, C is channels, H and W are height and width
        Returns:
            out: Output feature map, shape (B, NumLabels, H, W)
        """
        out = self.conv2d_list[0](x)  # Output of the first conv layer
        # For subsequent conv layers, add their output to the first
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out  # out.shape = (B, NumLabels, H, W)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    """
    Channel Attention Layer (CALayer)
    Generates a scalar weight for each channel of the input feature map, highlighting useful features and suppressing redundant information via channel attention.
    This layer compresses the feature map to a channel descriptor by global average pooling, then generates channel weights through two convolution layers.
    """

    def __init__(self, channel, reduction=16):
        """
        Args:
            channel: Number of input channels
            reduction: Channel reduction ratio, default 16
        """
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (1, 1)  # Compress each channel's feature map to a single value (global descriptor)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Channel re-weighting
        Args:
            x: Input feature map, shape (B, C, H, W)
        Returns:
            x*y: Re-weighted feature map, same shape as input
        """
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB)
    Reference: Image Super-Resolution Using Very Deep Residual Channel Attention Networks
    Implements a residual channel attention block with two convolution layers and a channel attention layer.
    The input feature map passes through two conv layers and a channel attention layer, then is added to the input as a residual connection.
    """

    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Args:
            n_feat: Number of input channels
            kernel_size: Convolution kernel size, default 3
            reduction: Channel reduction ratio, default 16
            bias: Whether to use bias, default True
            bn: Whether to use batch normalization, default False
            act: Activation function, default ReLU
            res_scale: Residual scaling factor, default 1
        """
        super(RCAB, self).__init__()
        modules_body = []
        # Stack two conv layers and a channel attention layer (optional BN + activation)
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:  # Add activation after the first conv layer
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))  # Add channel attention after two conv layers
        self.body = nn.Sequential(*modules_body)  # Compose all layers into a sequence
        self.res_scale = res_scale  # Residual scaling factor for controlling residual strength

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        """
        Default convolution layer, output channels same as input, uses padding to keep output size unchanged.
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            bias: Whether to use bias (default True)
        Returns:
            nn.Conv2d: Convolution layer
        """
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        """
        Forward function
        Args:
            x: Input feature map, shape (B, C, H, W)
        Returns:
            res: Feature map after residual connection, same shape as input
        """
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class BasicConv2d(nn.Module):
    """
    Basic Convolutional Layer with Batch Normalization
    Wraps a convolution layer and a batch normalization layer into a basic conv module.
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        """
        Args:
            in_planes: Number of input channels
            out_planes: Number of output channels
            kernel_size: Kernel size
            stride: Stride (default 1)
            padding: Padding (default 0)
            dilation: Dilation rate (default 1)
        """
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,  # No bias since followed by batch normalization
            ),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x):
        """
        Forward function
        Args:
            x: Input feature map, shape (B, C, H, W)
        Returns:
            x: Output feature map, same shape as input
        """
        x = self.conv_bn(x)
        return x


class Triple_Conv(nn.Module):
    """
    Triple Convolutional Layer
    Wraps three convolution layers into a triple conv module.
    """

    def __init__(self, in_channel, out_channel):
        """
        Args:
            in_channel: Number of input channels
            out_channel: Number of output channels
        """
        super(Triple_Conv, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),  # 1x1 conv for channel reduction
            BasicConv2d(out_channel, out_channel, 3, padding=1),  # 3x3 conv for feature extraction
            BasicConv2d(out_channel, out_channel, 3, padding=1),  # Another 3x3 conv for further feature extraction
        )

    def forward(self, x):
        """
        Forward function
        Args:
            x: Input feature map, shape (B, C, H, W)
        Returns:
            self.reduce(x): Output feature map, same shape as input
        """
        return self.reduce(x)


class Saliency_feat_encoder(nn.Module):
    """
    Merges features from a backbone with latent variables and applies
    attention mechanisms to produce initial and refined saliency maps.
    """

    # resnet based encoder decoder
    def __init__(self, channel, latent_dim):
        """
        Args:
            channel: Number of channels
            latent_dim: Latent space dimension
        """
        super(Saliency_feat_encoder, self).__init__()
        self.resnet = B2_ResNet()  # Backbone feature extractor based on ResNet
        # self.resnet=res2net50_v1b_26w_4s(pretrained=True)
        # self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )  # Upsample, scale_factor=8 means enlarging feature map 8x to restore original input size
        self.dropout = nn.Dropout(0.3)
        # Build two multi-scale dilated convolution classifiers
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2048)
        self.layer6 = self._make_pred_layer(
            Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 3
        )  # Change channels to 1 for initial saliency prediction

        # After downsampling, unify all layers to 'channel' channels; for ResNet's 2nd/3rd/4th outputs (256/512/1024 channels), use 1x1 conv to reduce to 'channel', then two 3x3 convs for feature extraction
        self.conv2_1 = nn.Conv2d(512, channel, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(1024, channel, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(2048, channel, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

        # Reserved multi-branch feature fusion and dimensionality reduction conv
        self.conv_feat = nn.Conv2d(32 * 5, channel, kernel_size=3, padding=1)

        # 4x and 2x upsampling for size alignment in multi-level feature fusion
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Spatial attention
        self.pam_attention5 = PAM_Module(channel)
        self.pam_attention4 = PAM_Module(channel)
        self.pam_attention3 = PAM_Module(channel)
        self.pam_attention2 = PAM_Module(channel)
        self.pam_attention1 = PAM_Module(channel)
        # Channel attention
        self.cam_attention4 = CAM_Module(channel)
        self.cam_attention3 = CAM_Module(channel)
        self.cam_attention2 = CAM_Module(channel)

        self.racb_layer = RCAB(channel * 4)  # Residual channel attention block, input channels = channel * 4

        # Multi-scale feature fusion conv layers, for ResNet's 4th/3rd/2nd outputs, used in refined prediction branch
        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        # Residual channel attention blocks, input channels = channel * 2/3/4
        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)
        # Triple conv layers, input channels = 2*channel/3*channel/4*channel
        self.conv43 = Triple_Conv(2 * channel, channel)
        self.conv432 = Triple_Conv(3 * channel, channel)
        self.conv4321 = Triple_Conv(4 * channel, channel)

        self.HA = HA()  # Holistic Attention module, fuses initial saliency map with features

        # Second feature fusion branch, a full set of fusion, attention, fusion, classifier for refinement
        self.conv4_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.pam_attention4_2 = PAM_Module(channel)
        self.pam_attention3_2 = PAM_Module(channel)
        self.pam_attention2_2 = PAM_Module(channel)
        self.cam_attention4_2 = CAM_Module(channel)
        self.cam_attention3_2 = CAM_Module(channel)
        self.cam_attention2_2 = CAM_Module(channel)
        self.racb_43_2 = RCAB(channel * 2)
        self.racb_432_2 = RCAB(channel * 3)
        self.conv43_2 = Triple_Conv(2 * channel, channel)
        self.conv432_2 = Triple_Conv(3 * channel, channel)
        self.conv4321_2 = Triple_Conv(4 * channel, channel)

        self.spatial_axes = [2, 3]  # Indices for spatial dimensions in tile

        self.conv_depth1 = BasicConv2d(
            3 + latent_dim, 3, kernel_size=3, padding=1
        )  # Concatenate input RGB image and latent z, then 3x3 conv to 3 channels for ResNet input

        self.layer7 = self._make_pred_layer(
            Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4
        )  # Final multi-scale classifier, output 1 channel for refined saliency map

        if self.training:
            self.initialize_weights()

    def _make_pred_layer(self, block, dilation_series, padding_series, NumLabels, input_channel):
        """
        Create a multi-scale dilated convolution classifier module
        Args:
            block: Classifier module class
            dilation_series: List of dilation rates
            padding_series: List of padding rates
            NumLabels: Number of output channels
            input_channel: Number of input channels
        Returns:
            block: Classifier module instance
        """
        return block(dilation_series, padding_series, NumLabels, input_channel)

    def tile(self, a, dim, n_tile):
        """
        This function is taken from PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        Used to repeat tensor a n_tile times along the specified dimension.
        Args:
            a (torch.Tensor): Input tensor
            dim (int): Specified dimension
            n_tile (int): Number of repetitions
        Returns:
            torch.Tensor: Tiled tensor
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, x, z):
        """
        Forward function
        Args:
            x: Input feature map, shape (B, C, H, W)
            z: Latent variable, shape (B, latent_dim)
        Returns:
            sal_init: Initial saliency map, shape (B, 1, H, W)
            sal_ref: Refined saliency map, shape (B, 1, H, W)
        """

        z = torch.unsqueeze(z, 2)  # Add a dimension before H axis
        z = self.tile(z, 2, x.shape[self.spatial_axes[0]])  # Tile z along H to match input feature map height
        z = torch.unsqueeze(z, 3)  # Add a dimension before W axis
        z = self.tile(z, 3, x.shape[self.spatial_axes[1]])  # Tile z along W to match input feature map width, shape (B, latent_dim, H, W)
        x = torch.cat((x, z), 1)  # Concatenate with RGB image along channel dim, shape (B, C + latent_dim, H, W)
        x = self.conv_depth1(x)  # 3x3 conv to 3 channels for ResNet input

        # ResNet feature extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8

        # Feature fusion and attention mechanisms
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv2_feat1 = self.pam_attention2(conv2_feat)
        conv2_feat2 = self.cam_attention2(conv2_feat)
        conv2_feat = conv2_feat1 + conv2_feat2
        conv3_feat = self.conv3(x3)
        conv3_feat1 = self.pam_attention3(conv3_feat)
        conv3_feat2 = self.cam_attention3(conv3_feat)
        conv3_feat = conv3_feat1 + conv3_feat2
        conv4_feat = self.conv4(x4)
        conv4_feat1 = self.pam_attention4(conv4_feat)
        conv4_feat2 = self.cam_attention4(conv4_feat)
        conv4_feat = conv4_feat1 + conv4_feat2
        conv4_feat = self.upsample2(conv4_feat)

        # Process conv4_feat and conv3_feat simultaneously, generate initial saliency map, multi-layer fusion
        conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)
        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)
        # conv432 = self.conv432(conv432)
        #
        # conv432 = self.upsample2(conv432)
        # conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        # conv4321 = self.racb_4321(conv4321)
        # conv4321 = self.conv4321(conv4321)

        sal_init = self.layer6(conv432)  # Initial saliency map, shape (B, 1, H, W)

        # Second feature fusion branch, complete copy of fusion, attention, fusion, classifier for saliency refinement
        x2_2 = self.HA(sal_init.sigmoid(), x2)  # Use initial saliency to guide mid-level features
        x3_2 = self.resnet.layer3_2(x2_2)  # 1024 x 16 x 16
        x4_2 = self.resnet.layer4_2(x3_2)  # 2048 x 8 x 8

        conv2_feat = self.conv2_2(x2_2)
        conv2_feat1 = self.pam_attention2_2(conv2_feat)
        conv2_feat2 = self.cam_attention2_2(conv2_feat)
        conv2_feat = conv2_feat1 + conv2_feat2
        conv3_feat = self.conv3_2(x3_2)
        conv3_feat1 = self.pam_attention3_2(conv3_feat)
        conv3_feat2 = self.cam_attention3_2(conv3_feat)
        conv3_feat = conv3_feat1 + conv3_feat2
        conv4_feat = self.conv4_2(x4_2)
        conv4_feat1 = self.pam_attention4_2(conv4_feat)
        conv4_feat2 = self.cam_attention4_2(conv4_feat)
        conv4_feat = conv4_feat1 + conv4_feat2

        conv4_feat = self.upsample2(conv4_feat)

        conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        conv43 = self.racb_43_2(conv43)
        conv43 = self.conv43_2(conv43)

        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432_2(conv432)

        conv432 = self.conv432_2(conv432)

        conv432 = self.upsample2(conv432)
        conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        conv4321 = self.racb_4321(conv4321)
        sal_ref = self.layer7(conv4321)

        return self.upsample8(sal_init), self.upsample4(sal_ref)

    def initialize_weights(self):
        """
        Initialize ResNet weights
        """
        res50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Load pretrained ResNet50 model
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():  # Direct mapping: if the custom ResNet layer name matches the pretrained model, copy weights directly
                v = pretrained_dict[k]
                all_params[k] = v
            elif "_1" in k:  # For layers with "_1" in the name, remove "_1" to match the original layer name
                name = k.split("_1")[0] + k.split("_1")[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif "_2" in k:  # For layers with "_2" in the name, remove "_2" to match the original layer name
                name = k.split("_2")[0] + k.split("_2")[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
