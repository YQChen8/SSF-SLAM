import torch
from torch import nn
from torch import distributions as ds
from torch.distributions import transforms as tr
from torch.distributions import transformed_distribution as tds
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from lib import pointnet2_utils as pointutils

from utils import *
# from gflow import InvConvdLU

LEAKY_RATE = 0.1
use_bn = False
EPS = 1e-6

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointutils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points


class FeedforwardNet(nn.Module):
    def __init__(self, in_channel, mlps, activation_fn='Tanh'):
        super(FeedforwardNet, self).__init__()
        '''
        :param inputs: [B,C,N]
        :param mlps: [1, m]
        :param activation_fn:
        :return:
        '''
        last_channel = in_channel
        # layers = []
        self.layers = nn.ModuleList()
        for i, out_channel in enumerate(mlps):
            if i < len(mlps) and activation_fn is not None:
                # layer = []
                if activation_fn == 'Tanh':
                    self.layers.append(nn.Sequential(nn.Conv1d(last_channel, out_channel,
                                                          kernel_size=1, bias=False),
                                           nn.BatchNorm1d(out_channel),
                                                nn.Tanh()))

                elif activation_fn == 'Sigmoid':
                    self.layers.append(nn.Sequential(nn.Conv1d(last_channel, out_channel,
                                            kernel_size=1, bias=False),
                                  nn.BatchNorm1d(out_channel),
                                  nn.Sigmoid()))
            elif i < len(mlps) and activation_fn is None:
                self.layers.append(nn.Sequential(nn.Conv1d(last_channel, out_channel,
                                                      kernel_size=1, bias=False),
                           nn.BatchNorm1d(out_channel)))
            last_channel = out_channel


        # self.net = nn.Sequential(*layers)

    def forward(self, input, conditions):
        '''
        :param input:
        :param conditions:
        :return:
        '''
        x = torch.cat([input, conditions], dim=1)

        for i, conv in enumerate(self.layers):
            x = conv(x)

        return x


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Flow(tr.Transform, nn.Module):
    def __init__(self):
        # super().__init__()
        tr.Transform.__init__(self)
        nn.Module.__init__(self)

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)

    def __hash__(self):
        return nn.Module.__hash__(self)


class CouplingBijector(Flow):
    def __init__(self, parity, in_channel, trans_hiddens, scale_hiddens, bijective=True):
        super(CouplingBijector, self).__init__()
        self.bijective = bijective
        self.domain = torch.distributions.constraints.Constraint()
        self.codomain = torch.distributions.constraints.Constraint()
        self.parity = parity

        self.trans_fn = FeedforwardNet(3 + in_channel, trans_hiddens)
        self.scale_fn = FeedforwardNet(3 + in_channel, scale_hiddens)
        self.init_parameters()

    def forward(self, x, conditions):
        '''
        :param pos1: [B, 3, N]
        :param pos2: [B, 3, N]
        :param flow: [B, 3, N]
        :param up_flow: [B, 3, N]
        :return:
        '''
        self.__maybe_assert_valid_x(x)
        self.__maybe_assert_valid_x(conditions)
        B, C, N = x.shape

        if self.parity == 'even':
            mask = x[:, :C//2, :]
            non_mask = x[:, C//2:, :]
        else:
            non_mask = x[:, :C//2, :]
            mask = x[:, C//2:, :]

        scale = self.scale_fn(mask, conditions)
        trans = self.trans_fn(mask, conditions)

        part_1 = mask
        part_2 = non_mask * torch.exp(scale) + trans

        to_concat = [part_1, part_2] if self.parity == 'even' else [part_2, part_1]

        out = torch.cat(to_concat, dim=1)

        return out

    def log_abs_det_jacobian(self, x, conditions):
        self.__maybe_assert_valid_x(x)
        self.__maybe_assert_valid_x(conditions)

        C = x.shape[1]
        masked_slice = (
            slice(None, C // 2)
            if self.parity == 'even'
            else slice(C // 2, None))
        masked_x = x[:, masked_slice]
        # nonlinearity_output_size = C - masked_x.shape[1]
        scale = self.scale_fn(masked_x, conditions)

        log_det_jacobian = scale.view(scale.size(0), -1).sum(-1) #???

        return log_det_jacobian

    def _call(self, x, conditions):
        '''
                :param pos1: [B, 3, N]
                :param pos2: [B, 3, N]
                :param flow: [B, 3, N]
                :param up_flow: [B, 3, N]
                :return:
                '''
        self.__maybe_assert_valid_x(x)
        self.__maybe_assert_valid_x(conditions)
        B, C, N = x.shape

        if self.parity == 'even':
            mask = x[:, :C//2, :]
            non_mask = x[:, C//2:, :]
        else:
            non_mask = x[:, :C//2, :]
            mask = x[:, C//2:, :]

        scale = self.scale_fn(mask, conditions)
        trans = self.trans_fn(mask, conditions)

        part_1 = mask
        part_2 = non_mask * torch.exp(scale) + trans

        to_concat = [part_1, part_2] if self.parity == 'even' else [part_2, part_1]

        out = torch.cat(to_concat, dim=1)

        return out

    def _inverse(self, y, conditions):
        self.__maybe_assert_valid_x(y)
        self.__maybe_assert_valid_x(conditions)

        C = y.shape[1]
        if self.parity == 'even':
            masked_y = y[:, :C // 2]
            non_masked_y = y[:, C // 2:]
        else:
            non_masked_y = y[:, :C // 2]
            masked_y = y[:, C // 2:]

        # s(y_{1:d}) in paper
        scale = self.scale_fn(masked_y, conditions, non_masked_y.shape[-1])
        # t(y_{1:d}) in paper
        translation = self.translation_fn(masked_y, conditions, non_masked_y.shape[-1])
        exp_scale = torch.exp(-scale)

        # y_{d+1:D} = (y_{d+1:D} - t(y_{1:d})) * exp(-s(y_{1:d}))
        part_1 = masked_y
        part_2 = (non_masked_y - translation) * exp_scale

        to_concat = [part_1, part_2] if self.parity == 'even' else [part_2, part_1]
        out = torch.cat(to_concat, dim=1)

        return out

    def __maybe_assert_valid_x(self, x):
        if x.is_contiguous():
            return x
        raise NotImplementedError("_maybe_assert_valid_x")


class RealNVPBijector(nn.Module):
    def __init__(self, in_channel, nsample, num_coupling_layers, translation_hiddens, scale_hiddens,
                 bijective=True, squash=True):
        super(RealNVPBijector, self).__init__()
        '''
        :param in_channel: the channel of flow embedding
        :param nsample:
        :param num_coupling_layers:
        :param translation_hiddens:
        :param scale_hiddens:
        '''
        self.nsample = nsample
        self.num_cpls = num_coupling_layers
        self.bijective = bijective
        self.squash = squash

        self.bijects = []
        for i in range(self.num_cpls):
            parity = 'even' if i % 2 else 'odd'
            layer = CouplingBijector(parity, in_channel, translation_hiddens, scale_hiddens, self.bijective)
            self.bijects.append(layer)

        self.transforms = tr.ComposeTransform(self.bijects)
        self.bijectors = nn.ModuleList(self.bijects)
        # self.base_distribution = ds.multivariate_normal.MultivariateNormal(
        #     loc=torch.zeros(3), covariance_matrix=torch.eye(3))
        self.base_distribution = ds.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(6), covariance_matrix=torch.eye(6))
        self.distribution = tds.TransformedDistribution(base_distribution=self.base_distribution,
                                                        transforms=self.transforms)

    def _pre_process(self, flow, pos):
        '''
        To compute the scene flow variance of the neighbourhoods
        :param pos_list: [[B,C,N1], [B,C,N2]] N1>N2
        :param flow_list: [[B,C,N1], [B,C,N2]] N1>N2
        :return:
        '''
        known = pos.permute(0, 2, 1).contiguous()
        unknown = pos.permute(0, 2, 1).contiguous()
        _, knn_idx = pointutils.knn(self.nsample, unknown, known)
        # flow_groupped = pointutils.grouping_operation(flow_list[0], knn_idx)
        # flow_curvature = torch.sum(flow_groupped - flow_list[1].unsqueeze(-1), dim=-1) / (self.nsample * 1.0)
        flow_groupped = pointutils.grouping_operation(flow, knn_idx)
        flow_curvature = torch.sum(flow_groupped, dim=-1) / (self.nsample * 1.0)
        x = torch.cat([flow, flow_curvature], dim=1)
        return x

    def log_flow_for(self, raw_flow, conditions):
        log_flow = self.distribution.log_prob(raw_flow, conditions)

        if self.squash:
            log_flow -= self.squash_correction(raw_flow)

        return log_flow

    def forward(self, pos, flow, flow_feats, with_log_prob=False):
        '''
        :param pos_list: [[B,C,N1], [B,C,N2]] N1>N2
        :param flow_list: [[B,C,N1], [B,C,N2]] N1>N2
        :param flow_feats: [B, C, N]
        :return:
        '''
        # if len(flow_list) == 2:
        self.__maybe_assert_valid_x(flow)
        self.__maybe_assert_valid_x(flow_feats)
        hidden_flow = self._pre_process(flow, pos)
        # else:
        #     hidden_flow = flow_list
        # sum_log_prob_flow = torch.zeros_like(flow.view(flow.size(0), -1)).sum(-1)
        sum_log_prob_flow = torch.zeros_like(flow)
        for layer in self.bijectors:
            hidden_flow = layer.forward(hidden_flow, flow_feats)
            # out = hidden_flow
            if with_log_prob:
                log_prob_flow = self.log_flow_for(hidden_flow, flow_feats)
                # out = layer.forward(out, flow_feats)
                sum_log_prob_flow += log_prob_flow

        if with_log_prob:
            return (hidden_flow[:, :3, :]).contiguous(), sum_log_prob_flow

        return (hidden_flow[:, :3, :]).contiguous()

    def log_abs_det_jacobian(self, pos, flow, flow_feats):
        # flow = flow_list[1]
        self.__maybe_assert_valid_x(flow)
        self.__maybe_assert_valid_x(flow_feats)
        sum_log_abs_det_jacobians = torch.zeros_like(flow.view(flow.size(0), -1)).sum(-1)

        out = flow
        for layer in self.layers:
            log_abs_det_jacobian = layer.log_abs_det_jacobian(out, flow_feats)
            out = layer.forward(out, flow_feats)
            assert (sum_log_abs_det_jacobians.shape.as_list()
                    == log_abs_det_jacobian.shape.as_list())

            sum_log_abs_det_jacobians += log_abs_det_jacobian

        return sum_log_abs_det_jacobians

    def _call(self, pos, flow, flow_feats):
        # flow = flow_list[1]
        for layer in self.layers:
            flow = layer(flow, flow_feats)

        return flow

    def _inverse(self, pos, flow, flow_feats):
        # flow = flow_list[1]
        self.__maybe_assert_valid_x(flow)
        self.__maybe_assert_valid_x(flow_feats)

        out = flow
        for layer in reversed(self.layers):
            out = layer.inverse(out, flow_feats)

        return out

    def __maybe_assert_valid_x(self, x):
        if x.is_contiguous():
            return x
        raise NotImplementedError("_maybe_assert_valid_x")


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x


class FlowLayer(nn.Module):
    def __init__(self, in_channel, kernel_size, padding, mlp, use_leaky=True, use_bn=True, use_relu=True):
        super(FlowLayer, self).__init__()
        last_channel = in_channel
        self.use_relu = use_relu

        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.feats_net = nn.Sequential(
            nn.Conv2d(last_channel, mlp[0], kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(mlp[0]) if use_bn else nn.Identity(),
            relu,
            nn.Conv2d(mlp[0], mlp[1], kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(mlp[1]) if use_bn else nn.Identity(),
            relu
        )

    def forward(self, feats):
        '''
        :param feats: [BN,C,K,K]
        :return: [BN,C,K,K]
        '''
        feats = self.feats_net(feats)
        # if self.use_relu:
        #     feats = self.relu(feats)
        return feats


class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], bn=True):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights = F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights


class PointWarping(nn.Module):
    def forward(self, pos1, pos2, flow1=None, nsample=None):
        if flow1 is None:
            return pos2

        # move pos1 to pos2'
        pos1_to_2 = pos1 + flow1

        # interpolate flow
        B, C, N1 = pos1.shape
        _, _, N2 = pos2.shape
        pos1_to_2_t = pos1_to_2.permute(0, 2, 1).contiguous() # B 3 N1
        pos2_t = pos2.permute(0, 2, 1).contiguous() # B 3 N2
        # flow1_t = flow1.permute(0, 2, 1).contiguous()
        if nsample is None:
            _, knn_idx = pointutils.three_nn(pos2_t, pos1_to_2_t)
        else:
            _, knn_idx = pointutils.knn(nsample, pos2_t, pos1_to_2_t)
        grouped_pos_norm = pointutils.grouping_operation(pos1_to_2, knn_idx) - pos2.view(B, C, N2, 1)
        dist = torch.norm(grouped_pos_norm, dim=1).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim = True)
        weight = (1.0 / dist) / norm

        grouped_flow1 = pointutils.grouping_operation(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, 1, N2, nsample) * grouped_flow1, dim=-1)
        warped_pos2 = (pos2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_pos2.clamp(-100.0, 100.0)


def fft_convolve(signal, kernel):
    import torch.fft as fft
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, if_IN=False, IN_affine=False, if_BN=False):
    if isReLU:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )
    else:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True)
            )


class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        '''
        :param xyz: [B,C,N]
        :param sparse_xyz: [B,C,N]
        :param sparse_flow: [B,C,N]
        :return: [B,C,N]
        '''
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz_t = xyz.permute(0, 2, 1).contiguous()  # B N 3
        sparse_xyz_t = sparse_xyz.permute(0, 2, 1).contiguous()  # B S 3
        # sparse_flow_t = sparse_flow.permute(0, 2, 1).contiguous() # B S 3
        # knn_idx = knn_point(3, sparse_xyz, xyz)
        _, knn_idx = pointutils.three_nn(xyz_t, sparse_xyz_t)
        grouped_xyz_norm = pointutils.grouping_operation(sparse_xyz, knn_idx) - xyz.view(B, C, N, 1)
        dist = torch.norm(grouped_xyz_norm, dim=1).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=-1, keepdim=True)
        weight = (1.0 / dist) / norm
        grouped_flow = pointutils.grouping_operation(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, 1, N, 3) * grouped_flow, dim=-1) #[B,C,N]

        return dense_flow.clamp(-100.0, 100.0)


def squeeze_2x2(x, reverse=False, alt_order=False):
    """For each spatial position, a sub-volume of shape `1x1x(N^2 * C)`,
    reshape into a sub-volume of shape `NxNxC`, where `N = block_size`.

    Adapted from:
        https://github.com/tensorflow/models/blob/master/research/real_nvp/real_nvp_utils.py

    See Also:
        - TensorFlow nn.depth_to_space: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
        - Figure 3 of RealNVP paper: https://arxiv.org/abs/1605.08803

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        reverse (bool): Whether to do a reverse squeeze (unsqueeze).
        alt_order (bool): Whether to use alternate ordering.
    """
    block_size = 2
    if alt_order:
        n, c, h, w = x.size()

        if reverse:
            if c % 4 != 0:
                raise ValueError('Number of channels must be divisible by 4, got {}.'.format(c))
            c //= 4
        else:
            if h % 2 != 0:
                raise ValueError('Height must be divisible by 2, got {}.'.format(h))
            if w % 2 != 0:
                raise ValueError('Width must be divisible by 4, got {}.'.format(w))
        # Defines permutation of input channels (shape is (4, 1, 2, 2)).
        squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                       [[[0., 0.], [0., 1.]]],
                                       [[[0., 1.], [0., 0.]]],
                                       [[[0., 0.], [1., 0.]]]],
                                      dtype=x.dtype,
                                      device=x.device)
        perm_weight = torch.zeros((4 * c, c, 2, 2), dtype=x.dtype, device=x.device)
        for c_idx in range(c):
            slice_0 = slice(c_idx * 4, (c_idx + 1) * 4)
            slice_1 = slice(c_idx, c_idx + 1)
            perm_weight[slice_0, slice_1, :, :] = squeeze_matrix
        shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(c)]
                                        + [c_idx * 4 + 1 for c_idx in range(c)]
                                        + [c_idx * 4 + 2 for c_idx in range(c)]
                                        + [c_idx * 4 + 3 for c_idx in range(c)])
        perm_weight = perm_weight[shuffle_channels, :, :, :]

        if reverse:
            x = F.conv_transpose2d(x, perm_weight, stride=2)
        else:
            x = F.conv2d(x, perm_weight, stride=2)
    else:
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)

        if reverse:
            if c % 4 != 0:
                raise ValueError('Number of channels {} is not divisible by 4'.format(c))
            x = x.view(b, h, w, c // 4, 2, 2)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.contiguous().view(b, 2 * h, 2 * w, c // 4)
        else:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError('Expected even spatial dims HxW, got {}x{}'.format(h, w))
            x = x.view(b, h // 2, 2, w // 2, 2, c)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, h // 2, w // 2, c * 4)

        x = x.permute(0, 3, 1, 2)

    return x


if __name__ == "__main__":
    import os
    import sys
    import argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--n_points', type=int, default=2048)
    args = parser.parse_args()

    sparse_input = torch.randn(12, 3, 512).cuda()
    input = torch.randn(12, 3, 2048).cuda()
    # label = torch.randn(4, 16)
    # (self, nsample, radius, in_channel, afn_mlp, fe_mlp, knn=True)
    # model = SOFlowEmbedding(16, 0.5, 3, [16, 16], [16, 16]).cuda()
    # model = HNFFlowEstimator2(nsample=5 ,in_channel=3, flow_mlp=[64, 128, 128], ctx_mlp=[128, 128]).cuda()
    # model = FeedforwardNet(6, [16, 32,64], activation_fn='Tanh').cuda()
    # __init__(self, in_channel, nsample, num_coupling_layers, translation_hiddens, scale_hiddens)
    model = RealNVPBijector(3, 5, 2, translation_hiddens=[128, 128, 3], scale_hiddens=[128, 128, 3]).cuda()
    total_num_paras = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is %d\t" % (total_num_paras))
    device_ids = [0]
    if len(device_ids) == 0:
        net = nn.DataParallel(model)
    else:
        net = nn.DataParallel(model, device_ids=device_ids)
    print("Let's use ", len(device_ids), " GPUs!")
    output = model([input, sparse_input], [input, sparse_input], sparse_input)
    print(output.shape)