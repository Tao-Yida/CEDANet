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
from model.Res2Net import res2net50_v1b_26w_4s


class Encoder_x(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)

        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)

        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
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

            mu = self.fc1_1(output)
            logvar = self.fc2_1(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
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
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_xy, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)

        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)

        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
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
    def __init__(self, channel, latent_dim, num_filters):
        super(Generator, self).__init__()
        self.sal_encoder = Saliency_feat_encoder(channel, latent_dim, num_filters)  # bnn
        self.xy_encoder = Encoder_xy(4, channel, latent_dim)  # uncertainty estimation network
        self.x_encoder = Encoder_x(3, channel, latent_dim)  #

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        """
        Sample from the latent space distribution using the reparameterization trick.
        Args:
            mu: Mean of the latent space
            logvar: Log variance of the latent space
        Returns:
            Latent vector sampled from the distribution
        """
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)  # Use torch.randn_like to automatically match device and dtype
        return eps.mul(std).add_(mu)

    def forward(self, x, y=None, training=True):
        if training:
            self.posterior, muxy, logvarxy = self.xy_encoder(torch.cat((x, y), 1))
            self.prior, mux, logvarx = self.x_encoder(x)
            lattent_loss = torch.mean(
                self.kl_divergence(self.posterior, self.prior)
            )  # Latent loss: the KL divergence between the posterior and prior distributions
            z_noise_post = self.reparametrize(muxy, logvarxy)
            z_noise_prior = self.reparametrize(mux, logvarx)

            self.sal_init_post, self.sal_ref_post, self.output_post = self.sal_encoder(x, z_noise_post)
            self.sal_init_prior, self.sal_ref_prior, self.output_prior = self.sal_encoder(x, z_noise_prior)
            # print(self.sal_init_prior.shape, self.sal_ref_prior.shape)

            self.sal_init_post = F.interpolate(
                self.sal_init_post, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True
            )  # instead of F.upsample
            self.sal_ref_post = F.interpolate(self.sal_ref_post, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
            self.sal_init_prior = F.interpolate(self.sal_init_prior, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
            self.sal_ref_prior = F.interpolate(self.sal_ref_prior, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
            return self.sal_init_post, self.sal_ref_post, self.sal_init_prior, self.sal_ref_prior, lattent_loss, self.output_post, self.output_prior
        else:
            _, mux, logvarx = self.x_encoder(x)
            z_noise = self.reparametrize(mux, logvarx)
            _, self.prob_pred, _ = self.sal_encoder(x, z_noise)
            return self.prob_pred


### BNN modules #######################################################################
class CAM_Module(nn.Module):
    """Channel attention module"""

    # paper: Dual Attention Network for Scene Segmentation
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature ( B X C X H X W)
            attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class PAM_Module(nn.Module):
    """Position attention module"""

    # paper: Dual Attention Network for Scene Segmentation
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature ( B X C X H X W)
            attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class Triple_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Triple_Conv, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
        )

    def forward(self, x):
        return self.reduce(x)


class Saliency_feat_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel, latent_dim, num_filters):
        super(Saliency_feat_encoder, self).__init__()
        self.channel = channel
        self.resnet = B2_ResNet()  # as backbone
        # self.resnet=res2net50_v1b_26w_4s(pretrained=True)
        # self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.dropout = nn.Dropout(0.3)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2048)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 3)

        # self.conv1 = nn.Conv2d(256, channel, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(512, channel, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(1024, channel, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(2048, channel, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

        # self.conv1 = Triple_Conv(256, channel)
        # self.conv2 = Triple_Conv(512, channel)
        # self.conv3 = Triple_Conv(1024, channel)
        # self.conv4 = Triple_Conv(2048, channel)

        self.conv_feat = nn.Conv2d(32 * 5, channel, kernel_size=3, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.pam_attention5 = PAM_Module(channel)
        self.pam_attention4 = PAM_Module(channel)
        self.pam_attention3 = PAM_Module(channel)
        self.pam_attention2 = PAM_Module(channel)

        self.cam_attention4 = CAM_Module(channel)
        self.cam_attention3 = CAM_Module(channel)
        self.cam_attention2 = CAM_Module(channel)

        self.pam_attention1 = PAM_Module(channel)
        self.racb_layer = RCAB(channel * 4)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)

        self.conv43 = Triple_Conv(2 * channel, channel)
        self.conv432 = Triple_Conv(3 * channel, channel)
        self.conv4321 = Triple_Conv(4 * channel, channel)

        self.HA = HA()
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
        self.spatial_axes = [2, 3]
        self.conv_depth1 = BasicConv2d(3 + latent_dim, 3, kernel_size=3, padding=1)
        self.layer7 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)

        # Contrastive loss specific layers
        self.num_filters = num_filters
        self.cont_c1_a = nn.Sequential(
            nn.Conv2d(in_channels=4 * self.channel, out_channels=self.num_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(),
        )
        self.cont_c1_b = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=1, bias=False)

        if self.training:
            self.initialize_weights()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        # The tiling operation is used to ensure that the latent variable z matches the spatial dimensions of the input tensor x before concatenation.
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, x, z):
        # print(x.shape, z.shape) # [7, 3, 352, 352], [7, 3]
        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, x.shape[self.spatial_axes[0]])
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, x.shape[self.spatial_axes[1]])
        x = torch.cat((x, z), 1)
        x = self.conv_depth1(x)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8

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

        conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)

        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)
        sal_init = self.layer6(conv432)
        x2_2 = self.HA(sal_init.sigmoid(), x2)
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

        # print(f"conv4321 {conv4321.shape}")
        sal_ref = self.layer7(conv4321)
        # print(sal_init.shape, sal_ref.shape)
        # print("finfish sal encoder")
        # print(f"channel {self.channel}")

        # Constastive loss specifc layers
        output1 = self.upsample4(conv4321)  # [7, 128, 88, 88] -> [7, 128, 352, 352]
        # print(f"output1 {output1.shape}")
        output2 = self.cont_c1_a(output1)  # [7, 16, 352, 352]
        # print(f"output2 {output2.shape}")
        output3 = self.cont_c1_b(output2)  # [7, 16, 352, 352]
        # print(f"output3 {output3.shape}")

        # sal_init=[7, 1, 44, 44], sal_ref=[7, 1, 88, 88]
        return self.upsample8(sal_init), self.upsample4(sal_ref), output3  # [7, 1, 352, 352], [7, 1, 352, 352], [7, 16, 352, 352]

    def initialize_weights(self):
        # res50 = models.resnet50(pretrained=True)
        res50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif "_1" in k:
                name = k.split("_1")[0] + k.split("_1")[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif "_2" in k:
                name = k.split("_2")[0] + k.split("_2")[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
