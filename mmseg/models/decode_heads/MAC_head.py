# Copyright (c) OpenMMLab. All rights reserved.
# this is the 0629 version casflow head, for albation study
# we add pos/focal/3x3 conv/bottom up
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_, trunc_normal_init)
from mmcv.utils import to_2tuple
from mmseg.ops import resize

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from mmseg.ops import resize


@HEADS.register_module()
class MACHead(BaseDecodeHead):

    def __init__(self,
                 pool_scales=(1, 2, 3, 6),
                 emd_dim=96,
                 WMHA_cascades=[2, 4, 7],
                 num_cascades=3,
                 head_nums=3,
                 cas_type='resize',
                 high_type='PPM',
                 add_conv_fuse=False,
                 direction='top-down',
                 rl_pos=True,
                 **kwargs):
        super(MACHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        # CasFlow Module
        self.direction = direction
        self.WMHA_cascades = WMHA_cascades
        self.num_cascades=num_cascades

        self.lateral_convs = nn.ModuleList()
        for in_channels in self.in_channels:
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        self.cas_ins = nn.ModuleList()
        for cas_in in range(len(self.in_index) + 1):
            cas_in_layer = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.cas_ins.append(cas_in_layer)

        self.MSA_cascade = nn.ModuleList()
        for i in range(len(self.in_index) + 1):
            layer_attention = nn.ModuleList()
            for i in range(self.num_cascades):
                window_size = WMHA_cascades[i]
                lyr_attn = Cascade_block(
                    channels=emd_dim,
                    window_size=window_size,
                    num_heads=head_nums,
                    drop_rate=0.,
                    drop_path_rate=0.,
                    act_cfg=dict(type='GELU'),
                    norm_cfg=dict(type='LN'),
                    rl_pos=rl_pos)
                layer_attention.append(lyr_attn)
            self.MSA_cascade.append(layer_attention)

        # High_level Module, PPM of PSPNet
        if high_type == 'PPM':
            self.high_level = PPM(
                pool_scales,
                self.in_channels[-1],
                self.channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                align_corners=self.align_corners)
            bottle_neck_index = pool_scales
            bottle_neck_channel = self.channels
        else:
            raise ValueError("Not a valid cas type")

        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(bottle_neck_index) * bottle_neck_channel,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # whether to add additional 3x3 conv layer at the final fusion part
        if add_conv_fuse:
            self.fuse_out = nn.Sequential(
                nn.Conv2d(self.channels * 5, self.channels * 5, kernel_size=3, padding=1),
                nn.Conv2d(self.channels * 5, self.channels * 5, kernel_size=1, padding=1),
                nn.Conv2d(self.channels * 5, self.num_classes, kernel_size=1))
        else:
            self.fuse_out = nn.Conv2d(self.channels*5, self.num_classes, kernel_size=1)


        self.eca_layers = nn.ModuleList()
        for eca in range(len(self.in_index) + 1):
            eca_module = eca_layer(channel=self.channels, k_size=3)
            self.eca_layers.append(eca_module)

        self.se_layer = SELayer(channel=self.channels*5)

    def high_level_forward(self, inputs):

        """Forward function of PSP module."""
        x = inputs[-1]
        high_outs = [x]
        high_outs.extend(self.high_level(x))
        high_outs = torch.cat(high_outs, dim=1)
        output = self.bottleneck(high_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        fpn_ins = []
        for i in self.in_index:
            lateral_i = self.lateral_convs[i](inputs[i])
            fpn_ins.append(lateral_i)
        fpn_ins.append(self.high_level_forward(inputs))

        macs = []
        for i in range(len(self.in_index)+1):
            cas_in = self.cas_ins[i](fpn_ins[i])
            # print('casin', cas_in.shape)
            cas_1 = self.MSA_cascade[i][0](cas_in, cas_in)
            cas_2 = self.MSA_cascade[i][1](cas_1, cas_1)
            cas_3 = self.MSA_cascade[i][2](cas_2, cas_2)
            # print('cas3', cas_3.shape)
            cas_out = self.eca_layers[i](cas_3)
            # print('casout', cas_out.shape)
            ECA_out = resize(cas_out, [224, 224], mode='bilinear', align_corners=False)
            # print('eca', ECA_out.shape)
            macs.append(ECA_out)

        output = self.fuse_out(self.se_layer(torch.cat(macs, dim=1)))

        return output


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel=192, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


def de_conv2x2(in_planes, out_planes):
    "2x2 de_convolution without padding"
    return nn.ConvTranspose2d(in_planes,
                              out_planes,
                              kernel_size=2,
                              stride=2,
                              padding=0)


def de_conv2x2_bn_relu(in_planes, out_planes, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        de_conv2x2(in_planes, out_planes),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )


def de_conv4x4(in_planes, out_planes):
    "4x4 de_convolution without padding"
    return nn.ConvTranspose2d(in_planes,
                              out_planes,
                              kernel_size=4,
                              stride=4,
                              padding=0)


def de_conv4x4_bn_relu(in_planes, out_planes, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        de_conv4x4(in_planes, out_planes),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )

class up_sample(nn.Module):

    def __init__(self,
                 mode='bilinear',
                 align_corners=None,
                 channels=96,
                 norm_layer = nn.BatchNorm2d,
                 act_layer = nn.ReLU
                 ):
        super(up_sample, self).__init__()
        self.mode = mode
        self.align_corners = align_corners
        self.channels = channels
        self.conv_block = nn.ModuleList
        self.conv_block = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=1),
                                        norm_layer(self.channels),
                                        act_layer(inplace=False))

    def forward(self, x):

        h, w = x.shape[2:]
        size = [2*h, 2*w]
        x = F.interpolate(x, size, mode=self.mode, align_corners=self.align_corners)
        out = self.conv_block(x)

        return out

class down_sample(nn.Module):

    def __init__(self,
                 mode='bilinear',
                 align_corners=None,
                 channels=96,
                 norm_layer = nn.BatchNorm2d,
                 act_layer = nn.ReLU
                 ):
        super(down_sample, self).__init__()
        self.mode = mode
        self.align_corners = align_corners
        self.channels = channels
        self.conv_block = nn.ModuleList
        self.conv_block = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=1),
                                        norm_layer(self.channels),
                                        act_layer(inplace=False))

    def forward(self, x):

        h, w = x.shape[2:]
        size = [h//2, w//2]
        x = F.interpolate(x, size, mode=self.mode, align_corners=self.align_corners)
        out = self.conv_block(x)

        return out


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape   # [B, 56, 56, 96], window_size=7
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # [B*window_num, 7, 7, 96]
    # window_num = H*W// window_size=64 in first block
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Cascade_block(nn.Module):
    """ Window Attention copied followed Swin-TRM"""

    # Ref from SAGAN
    def __init__(self,
                 channels,
                 window_size,
                 num_heads,
                 drop_rate,
                 drop_path_rate,
                 act_cfg,
                 norm_cfg,
                 rl_pos):

        super(Cascade_block, self).__init__()
        self.channels = channels
        self.window_size = window_size
        self.window_sizes = to_2tuple(window_size)
        self.rl_pos = rl_pos
        self.num_heads = num_heads
        self.scale = 32 ** -0.5
        self.softmax = nn.Softmax(dim=-1)

        if self.rl_pos:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_sizes[0] - 1) * (2 * self.window_sizes[1] - 1),
                            self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # About 2x faster than original impl
            Wh, Ww = self.window_sizes
            rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
            rel_position_index = rel_index_coords + rel_index_coords.T
            rel_position_index = rel_position_index.flip(1).contiguous()
            self.register_buffer('relative_position_index', rel_position_index)

        # through linear function to obtain a transformer mapping
        self.x2q = nn.Linear(self.channels, self.channels, bias=True)
        self.y2k = nn.Linear(self.channels, self.channels, bias=True)
        self.x2v = nn.Linear(self.channels, self.channels, bias=True)
        self.proj = nn.Linear(self.channels, self.channels)
        self.proj_drop = nn.Dropout(drop_rate)

        self.Mlp = FFN(
            embed_dims=channels,
            feedforward_channels=channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)
        self.norm = build_norm_layer(norm_cfg, channels)[1]

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
                y : input feature maps( B X C X H X W)
            returns :
                out : attention value + x
                attention: B X (HxW) X (HxW)
        """
        b, c, h, w = x.size()  # torch.Size([1, 96, 28, 28])
        B, C, H, W = y.size()

        assert self.channels == c, "input feature has wrong dim"
        assert c == C, "input feature shapes are different"
        l, L = h*w, H*W

        x = x.flatten(2).transpose(1, 2)
        shortcut = x  # torch.Size([1, 784, 96])
        x = x.view(b, h, w, c)
        y = y.flatten(2).transpose(1, 2).view(B, H, W, C)  # torch.Size([1, 28, 28, 96])

        x = window_partition(x, self.window_size)
        y = window_partition(y, self.window_size)  # torch.Size([16, 7, 7, 96])

        x = x.view(-1, self.window_size * self.window_size, c)  # torch.Size([16, 49, 96])
        y = y.view(-1, self.window_size * self.window_size, C)
        b_, N, c = x.shape  # [16b, 49, 96]
        B_, N, C = y.shape

        q = self.x2q(x)
        k = self.y2k(y)
        v = self.x2v(x)

        q = q.reshape(b_, N, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        # torch.Size([16, 3, 49, 32])
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(b_, N, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))  # torch.Size([16, 3, 49, 49])
        # print('attn', attn.shape)

        if self.rl_pos:
            # re_pos information
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                self.window_sizes[0] * self.window_sizes[1],
                self.window_sizes[0] * self.window_sizes[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_window = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # torch.Size([16, 49, 96])
        attn_window = self.proj(attn_window)
        attn_window = self.proj_drop(attn_window)
        attn_window = attn_window.view(-1, self.window_size, self.window_size, C)  # torch.Size([16, 7, 7, 96])

        attn_out = window_reverse(attn_window, self.window_size, H, W)  # torch.Size([2, 28, 28, 96])
        attn_out = attn_out.view(b, h * w, c)  # torch.Size([2, 784, 96])
        attn_out = attn_out + shortcut  # torch.Size([2, 784, 96])

        # MLP
        identity = attn_out
        out = self.norm(attn_out)
        out = self.Mlp(out, identity=identity)
        out = out.transpose(1, 2).view(b, c, h, w)  # torch.Size([1, 96, 28, 28])

        return out

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


