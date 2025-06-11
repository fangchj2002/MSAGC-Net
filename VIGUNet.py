import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from nnunet.network_architecture.ZGQ.VIG_ZGQ.gcn_lib import Grapher, act_layer, AttentionGrapher
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
from typing import Optional
from collections import OrderedDict
from nnunet.network_architecture.neural_network import SegmentationNetwork


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.GELU())

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, scfactor=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scfactor, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=nn.GELU()) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation is not None:
            layers.append(activation)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class Attention_SE(nn.Module):
    '''
      Attention block/mechanism
    '''
    def __init__(self, dims,reduction=16):
        super(Attention_SE, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.BatchNorm2d(dims)
        self.fc = nn.Sequential(
            # nn.Linear(dims, dims // reduction, bias=False),
            nn.Conv2d(dims, dims // reduction, kernel_size=1, bias=True),
            nn.GELU(),
            # nn.Linear(dims// reduction, dims, bias=False),
            nn.Conv2d(dims // reduction, dims, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.ch_conv3 = nn.Sequential(
            BN_Conv2d(dims,dims // 2, 3, 1, 1, bias=False),
            BN_Conv2d(dims // 2, dims, 1, 1, 0, bias=False),
        )

    def forward(self, x, drop_rate=0.25):
        B, C, H, W = x.shape

        x_se = self.ch_conv3(x)
        x_se = self.norm(x_se)
        x_se = self.avg_pool(x_se)
        # x_se = x_se.view(B, C)
        gp = self.fc(x_se)
        # gp = gp.view(B, C, 1, 1)
        out_layer1 = x * gp.expand_as(x)# channel -- low channel

        return out_layer1


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, block_units=[3, 4, 9], width=32):
        super().__init__()
        # self.channels = 384
        self.k = 9
        self.act = 'gelu'
        self.conv = 'mr'
        self.norm = 'batch'  # batch or instance normalization {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.n_blocks = 6  # number of basic blocks in the backbone
        self.dropout = 0.1  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.stochastic = False  # stochastic for gcn, True or False
        self.drop_path = 0.0
        # self.blocks = [4, 6]
        if width > 32:
            self.group = 32
        else:
            self.group = width

        self.width = width
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, 2*self.n_blocks)]  # stochastic depth decay rule
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(self.k, 2 * self.k, 2*self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 768 // max(num_knn)
        reduce_ratios = [4, 2, 1, 1]

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(in_channels, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(self.group, width, eps=1e-6)),
            ('gelu', nn.GELU()),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width, cout=width * 4, cmid=width,))] +
                [(f'unit{i:d}/', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width, )) for i in
                 range(2, block_units[0] + 1)],
            ))),
        ]))

        self.CNNbolock1 = nn.Sequential(OrderedDict([
            ('block2/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2,  stride=2))] +
                [(f'unit{i:d}/', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2, )) for i in
                 range(2, block_units[1] + 1)],
            )))
        ]))
        self.CNNbolock2 = nn.Sequential(OrderedDict([
            ('block3/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}/', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

        self.GNNbolock1 = Seq(*[Seq(FFN(width*8, width*4 * 4, act=self.act, drop_path=dpr[i]),
                                 Grapher(width*8, kernel_size=num_knn[i], dilation=min(i // 4 + 1, max_dilation), conv=self.conv, act=self.act, norm=self.norm,
                                         bias=self.bias, stochastic=self.stochastic, epsilon=self.epsilon, r=1, n=64*64, drop_path=dpr[i]),
                                 ) for i in range(self.n_blocks)])
        self.down1 = Downsample(width * 4, width * 8)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=16 * width),
            nn.Conv2d(16 * width, 4 * width, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.0),
            nn.Conv2d(4 * width, 8 * width, kernel_size=(1, 1), bias=True),
            nn.Dropout(p=0.1)
        )

        self.GNNbolock2 = Seq(*[Seq(FFN(width * 16, width * 8 * 4, act=self.act, drop_path=dpr[i+self.n_blocks]),
                                 Grapher(width * 16, kernel_size=num_knn[i], dilation=min(i // 4 + 1, max_dilation), conv=self.conv, act=self.act, norm=self.norm,
                                         bias=self.bias, stochastic=self.stochastic, epsilon=self.epsilon, r=1, n=1024, drop_path=dpr[i+self.n_blocks]),
                                 ) for i in range(self.n_blocks)])
        self.down2 = Downsample(width * 8, width * 16)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=32 * width),
            nn.Conv2d(32 * width, 8 * width, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.0),
            nn.Conv2d(8 * width, 16 * width, kernel_size=(1, 1), bias=True),
            nn.Dropout(p=0.1)
        )
        # self.ParCblock1 = gcc_mf_block(dim=width * 16, meta_kernel_size=32)

    def forward(self, x):
        features = []
        x1 = self.root(x)
        features.append(x1)
        x2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x1)
        x2 = self.body(x2)
        features.append(x2)

        x3_1 = self.CNNbolock1(x2)
        x3_2 = self.GNNbolock1(self.down1(x2))
        x3 = self.conv1(torch.cat((x3_1, x3_2), dim=1))
        features.append(x3)

        x4_1 = self.CNNbolock2(x3)
        x4_2 = self.GNNbolock2(self.down2(x3))
        x4 = self.conv2(torch.cat((x4_1, x4_2), dim=1))
        # x5 = self.ParCblock1(x4)
        features.append(x4)

        # return x5, features[::-1]
        return features


class MSFBlock(nn.Module):
    def __init__(self, num=4):  # (224*64, 112*128, 56*256, 28*256, 14*512) dim = 256
        super(MSFBlock, self).__init__()
        self.channels = [32, 128, 256, 512]
        self.dim = 512
        self.split_list = [512, 512, 512, 512]
        self.k = 9
        self.act = 'gelu'
        self.conv = 'mr'
        self.norm = 'batch'
        self.bias = True
        self.n_blocks = num
        self.use_dilation = True
        self.epsilon = 0.2
        self.stochastic = False
        self.drop_path = 0.0

        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_blocks)]  # stochastic depth decay rule
        # print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(self.k, 2 * self.k, self.n_blocks)]  # number of knn's k
        # print('num_knn', num_knn)
        max_dilation = 768 // max(num_knn)

        self.body = nn.Sequential(OrderedDict([
            ('block1/', nn.Sequential(OrderedDict(
                [('unit1/', BN_Conv2d(self.channels[0], self.channels[0], 1, 1, 0))] +
                [('unit2/', BN_Conv2d(self.channels[0], self.dim, 8, 8, 0, groups=self.channels[0]))] +
                [('unit3/', BN_Conv2d(self.dim, self.dim, 1, 1, 0))],
            ))),
            ('block2/', nn.Sequential(OrderedDict(
                [('unit1/', BN_Conv2d(self.channels[1], self.channels[1], 1, 1, 0))] +
                [('unit2/', BN_Conv2d(self.channels[1], self.dim, 4, 4, 0, groups=self.channels[1]))] +
                [('unit3/', BN_Conv2d(self.dim, self.dim, 1, 1, 0))],
            ))),
            ('block3/', nn.Sequential(OrderedDict(
                [('unit1/', BN_Conv2d(self.channels[2], self.channels[2], 1, 1, 0))] +
                [('unit2/', BN_Conv2d(self.channels[2], self.dim, 2, 2, 0, groups=self.channels[2]))] +
                [('unit3/', BN_Conv2d(self.dim, self.dim, 1, 1, 0))],
            ))),
            ('block4/', nn.Sequential(OrderedDict(
                [('unit1/', BN_Conv2d(self.channels[3], self.dim, 1, 1, 0))],
            ))),
        ]))

        self.group_attention = Seq(*[Seq(FFN(512 * 4, 512, act=self.act, drop_path=dpr[i]),
                                 Grapher(512 * 4, 512,kernel_size=num_knn[i], dilation=min(i // 4 + 1, max_dilation), conv=self.conv, act=self.act, norm=self.norm,
                                         bias=self.bias, stochastic=self.stochastic, epsilon=self.epsilon, r=1, n=32*32, drop_path=dpr[i]),
                                 ) for i in range(self.n_blocks)])

        self.up = nn.Sequential(OrderedDict([
            ('up1/', nn.Sequential(OrderedDict(
                [('unit1/', up_conv(self.dim, self.channels[0], scfactor=8))],
            ))),
            ('up2/', nn.Sequential(OrderedDict(
                [('unit1/', up_conv(self.dim, self.channels[1], scfactor=4))],
            ))),
            ('up3/', nn.Sequential(OrderedDict(
                [('unit1/', up_conv(self.dim, self.channels[2], scfactor=2))]
            ))),
            ('block4/', nn.Sequential(OrderedDict(
                [('unit1/', BN_Conv2d(self.dim, self.channels[3], 1, 1, 0))],
            ))),
        ]))

    def forward(self, x):
        # Patch Matching
        for i, item in enumerate(x):
            item = self.body[i](item)
            x[i] = item
        x = tuple(x)
        x = torch.cat(x, dim=1)  # (B, H // win, W // win, N, C)
        x = self.group_attention(x)
        x = torch.split(x, self.split_list, dim=1)
        x = list(x)
        for j, item in enumerate(x):
            item = self.up[j](item)
            x[j] = item

        return x


class Conv2dGELU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        gelu = nn.GELU()

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dGELU, self).__init__(conv, bn, gelu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.attention = Attention_SE(in_channels + 2*skip_channels)
        self.attention2 = Attention_SE(in_channels)
        self.conv1 = Conv2dGELU(
            in_channels + 2*skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv12 = Conv2dGELU(
            in_channels ,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dGELU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None, skip2=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip, skip2], dim=1)

        x = self.attention2(x)  #MSF_VIG调用attention，VIG调用attention2
        x = self.conv12(x)      #MSF_VIG调用conv1，VIG调用conv12
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, hidden_size=512, decoder_channels=(256, 128, 64, 16), skip_channels=[512, 256, 64, 16]):
        super().__init__()
        head_channels = 512
        self.n_skip = 3
        self.skip_channels = skip_channels
        self.conv_more = Conv2dGELU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.n_skip != 0:
            skip_channels = self.skip_channels
            for i in range(4-self.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None, features2=None):
        x = self.conv_more(hidden_states)
        #以下此段循环为MSF_VIG调用，还需更改decoder_block
        # for i, decoder_block in enumerate(self.blocks):
        #     if features is not None:
        #         skip = features[i] if (i < self.n_skip) else None
        #         skip2 = features2[i] if (i < self.n_skip) else None
        #     else:
        #         skip = None
        #         skip2 = None
        #     x = decoder_block(x, skip=skip, skip2=skip2)
        # return x

        #以下此段循环为VIG调用,还需更改decoder_block
        for i, decoder_block in enumerate(self.blocks):
            if features2 is not None:
                skip = features[i] if (i < self.n_skip) else None
                skip2 = features2[i] if (i < self.n_skip) else None
            else:
                skip = None
                skip2 = None
            x = decoder_block(x, skip=skip, skip2=skip2)
        return x

class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, inc_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(inc_ch)
        self.gelu1 = nn.GELU()

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(inc_ch, inc_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(inc_ch)
        self.gelu2 = nn.GELU()

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(inc_ch, inc_ch, 3,padding=1)
        self.bn3 = nn.BatchNorm2d(inc_ch)
        self.gelu3 = nn.GELU()

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(inc_ch, inc_ch,3,padding=1)
        self.bn4 = nn.BatchNorm2d(inc_ch)
        self.gelu4 = nn.GELU()

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####
        self.conv = nn.Conv2d(512, inc_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(inc_ch)
        self.gelu = nn.GELU()

        self.conv5 = nn.Conv2d(inc_ch*2, inc_ch, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(inc_ch)
        self.gelu5 = nn.GELU()

        #####

        self.conv_d4 = nn.Conv2d(inc_ch*2, inc_ch, 3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(inc_ch)
        self.gelu_d4 = nn.GELU()

        self.conv_d3 = nn.Conv2d(inc_ch*2, inc_ch,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(inc_ch)
        self.gelu_d3 = nn.GELU()

        self.conv_d2 = nn.Conv2d(inc_ch*2, inc_ch, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(inc_ch)
        self.gelu_d2 = nn.GELU()

        self.conv_d1 = nn.Conv2d(inc_ch*2, inc_ch, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(inc_ch)
        self.gelu_d1 = nn.GELU()

        self.conv_d0 = nn.Conv2d(inc_ch, in_ch, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, y):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.gelu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.gelu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.gelu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.gelu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hy = self.gelu(self.bn(self.conv(y)))
        hx = torch.cat((hx, hy), dim=1)

        hx5 = self.gelu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.gelu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.gelu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.gelu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.gelu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual


class VIGUNet(SegmentationNetwork):
    def __init__(self,input_channels=1,num_classes=2, drop_path_rate=0.0, drop_rate=0.0, num_knn=9 ):
        super(VIGUNet, self).__init__()
        self.n_classes = num_classes  # Dimension of out_channels
        self.decoder_channels = (256, 128, 64, 16)

        self.embedding = EncoderBlock(3,[3,4,9])
        self.SpatialAwareTrans = MSFBlock(num=2)
        self.decoder = DecoderCup(hidden_size=512,decoder_channels=self.decoder_channels,skip_channels=[256, 128, 32, 16])
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.refunet = RefUnet(in_ch=2, inc_ch=32)
        self.model_init()
        self.do_ds = True

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)



        #x, feather = self.embedding(x)
        feather = self.embedding(x)
        #x_encode = x
        feathers = feather[::-1]
        x_encode = feathers.pop(0)
        feathers.append(x)
        feathers = self.SpatialAwareTrans(feather)
        x = feathers[3]
        feather = list(reversed(feathers[0:3]))
        x = self.decoder(x, feather)
        out = self.segmentation_head(x)

        out2 = self.refunet(out,x_encode)
        return out2


class VIGUNet2(SegmentationNetwork):
    """
    纯VIG
    """
    def __init__(self,input_channels=1,num_classes=2, drop_path_rate=0.0, drop_rate=0.0, num_knn=9 ):
        super(VIGUNet2, self).__init__()
        self.n_classes = num_classes  # Dimension of out_channels
        self.decoder_channels = (256, 128, 64, 16)

        self.embedding = EncoderBlock(3,[3,4,9])
        self.SpatialAwareTrans = MSFBlock(num=2)
        self.decoder = DecoderCup(hidden_size=512,decoder_channels=self.decoder_channels,skip_channels=[256, 128, 32, 16])
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.refunet = RefUnet(in_ch=2, inc_ch=32)
        self.model_init()
        self.do_ds = True

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        feather = self.embedding(x)
        feathers = feather[::-1]
        encodex = feathers.pop(0)

        x = self.decoder(encodex, feather)
        out = self.segmentation_head(x)
        return out



class MSF_VIGUNet(SegmentationNetwork):
    def __init__(self, input_channels=1, num_classes=2):
        super(MSF_VIGUNet, self).__init__()
        self.n_classes = num_classes  # Dimension of out_channels
        self.decoder_channels = (256, 128, 64, 16)

        self.encoder = EncoderBlock(3, [3, 4, 9])
        self.MSFBlock = MSFBlock()
        self.decoder = DecoderCup(hidden_size=512, decoder_channels=self.decoder_channels,
                                  skip_channels=[256, 128, 32, 16])
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.refunet = RefUnet(in_ch=2, inc_ch=32)
        self.model_init()
        self.do_ds = True

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        features = self.encoder(x)
        feature = features[::-1] #feathers[:]
        encodex = feature.pop(0)
        msf_features = self.MSFBlock(features)
        msf_feature = msf_features[::-1]
        x = msf_feature.pop(0)
        x = self.decoder(x, feature, msf_feature)
        out = self.segmentation_head(x)

        out2 = self.refunet(out, encodex)
        # return out, out2
        return out2


class MSF_VIGUNet2(SegmentationNetwork):
    def __init__(self, input_channels=1, num_classes=2):
        super(MSF_VIGUNet2, self).__init__()
        self.n_classes = num_classes  # Dimension of out_channels
        self.decoder_channels = (256, 128, 64, 16)

        self.encoder = EncoderBlock(3, [3, 4, 9])
        self.MSFBlock = MSFBlock()
        self.decoder = DecoderCup(hidden_size=512, decoder_channels=self.decoder_channels,
                                  skip_channels=[256, 128, 32, 16])
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.refunet = RefUnet(in_ch=2, inc_ch=32)
        self.model_init()
        self.do_ds = True

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        features = self.encoder(x)
        feature = features[::-1] #feathers[:]
        encodex = feature.pop(0)
        msf_features = self.MSFBlock(features)
        msf_feature = msf_features[::-1]
        x = msf_feature.pop(0)
        x = self.decoder(x, feature, msf_feature)
        out = self.segmentation_head(x)

        #out2 = self.refunet(out, encodex)
        # return out, out2
        return out


# net = VIGUNet2()
# print(net)
# x = torch.randn(1, 3, 512, 512)
# out2 = net(x)
# # print(out1.shape)
# print(out2.shape)
# total = sum([param.nelement() for param in net.parameters()])
# print("Number of parameter: %.2fM" % (total/1e6))
# print(total)

