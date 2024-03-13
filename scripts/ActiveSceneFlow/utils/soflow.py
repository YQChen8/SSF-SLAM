import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from lib import pointnet2_utils as pointutils
from utils.datasets.carla import add_Seg_after_FLow
from .utils import *
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch_scatter import scatter_softmax, scatter_sum

# from gflow import InvConvdLU

LEAKY_RATE = 0.1
use_bn = False


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointutils.grouping_operation(points_flipped.float(), knn_idx.int()).permute(0, 2, 3, 1).contiguous()

    return new_points


class PointConvTransFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, attn_temp=1, bn=use_bn, use_leaky=True):
        super(PointConvTransFlow, self).__init__()
        self.attn_temp = attn_temp
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel * 2 + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.mlp_convs2 = nn.ModuleList()
        if bn:
            self.mlp_bns2 = nn.ModuleList()
        last_channel = in_channel * 2 + 3
        for out_channel in mlp:
            self.mlp_convs2.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns2.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, xyz2w, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()
        if xyz2w != None:
            xyz2w = xyz2w.permute(0, 2, 1).contiguous()
        else:
            xyz2w = xyz1
        points1 = points1.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()

        # point-to-patch Volume
        # knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        _, knn_idx = pointutils.knn(self.nsample, xyz1, xyz2)
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        _, knn_idxw = pointutils.knn(self.nsample, xyz1, xyz2w)
        neighbor_xyzw = index_points_group(xyz2w, knn_idxw)
        direction_xyzw = neighbor_xyzw - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx)  # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim=-1)  # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        grouped_points2w = index_points_group(points2, knn_idxw)  # B, N1, nsample, D2
        new_pointsw = torch.cat([grouped_points1, grouped_points2w, direction_xyzw], dim=-1)  # B, N1, nsample, D1+D2+3
        new_pointsw = new_pointsw.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_pointsw = self.relu(bn(conv(new_pointsw)))
            else:
                new_pointsw = self.relu(conv(new_pointsw))

        # weight_qk = torch.matmul(grouped_points2, grouped_points2w.permute(0,1,3,2).contiguous()) # B, N1, nsample, nsmaple
        weight_qk = torch.matmul(new_points, new_pointsw.permute(0, 1, 3, 2).contiguous())  # B, N1, nsample, nsmaple
        weight_qk = (torch.softmax(weight_qk / self.attn_temp, -1) / math.sqrt(C)).clamp(min=1e-10)
        # new_pointsw = torch.sum(weight_qk.unsqueeze(1) * new_pointsw.unsqueeze(3), dim=-1)  # B C N S

        new_pointsw = torch.matmul(new_pointsw.permute(0, 1, 3, 2).contiguous(), weight_qk)  # B C N S
        # weighted sum
        # weightsw = self.weightnet1(direction_xyzw.permute(0, 3, 2, 1))  # B C nsample N1
        # point_to_patch_costw = torch.sum(weightsw.permute(0,1,3,2).contiguous() * new_pointsw, dim=-1)  # B C N

        point_to_patch_costw = torch.sum(new_pointsw, dim=-1)  # B C N

        # for i, conv in enumerate(self.mlp_convs2):
        #     if self.bn:
        #         bn = self.mlp_bns2[i]
        #         new_points = self.relu(bn(conv(new_points)))
        #     else:
        #         new_points = self.relu(conv(new_points))
        new_points = torch.matmul(weight_qk, new_points)  # B C N S)
        # weighted sum
        # weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1
        # point_to_patch_cost = torch.sum(weights * new_points, dim=-2)  # B C N

        point_to_patch_cost = torch.sum(new_points, dim=-2)  # B C N

        # point_to_patch_cost = torch.cat([point_to_patch_costw, point_to_patch_cost], dim=1)

        # Patch to Patch Cost
        # knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        _, knn_idx = pointutils.knn(self.nsample, xyz1, xyz1)
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1),
                                                         knn_idx)  # B, N1, nsample, C
        grouped_point_to_patch_costw = index_points_group(point_to_patch_costw.permute(0, 2, 1),
                                                          knn_idx)  # B, N1, nsample, C
        grouped_point_to_patch_cost = grouped_point_to_patch_cost + grouped_point_to_patch_costw
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim=2)  # B C N

        # coarse_sf_cost = torch.max(new_pointsw, dim=-1)[0] + torch.max(new_points, dim=-1)[0]
        # max_new_points_weight = torch.max(torch.sum(weight_qk, dim=-2).unsqueeze(1), dim=-1)[0] + torch.max(torch.sum(weight_qk, dim=-1).unsqueeze(1), dim=-1)[0]
        # coarse_sf_cost = coarse_sf_cost / max_new_points_weight.clamp(1e-10)
        return patch_to_patch_cost


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=64, dim=128, temperature=10000):
        super(PositionalEncodingFourier, self).__init__()
        self.token_projection = nn.Linear(hidden_dim * 3, dim)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim

    def forward(self, pos_embed, max_len=(1, 1, 1)):
        z_embed, y_embed, x_embed = pos_embed.chunk(3, 1)
        z_max, y_max, x_max = max_len

        eps = 1e-6
        z_embed = z_embed / (z_max + eps) * self.scale
        y_embed = y_embed / (y_max + eps) * self.scale
        x_embed = x_embed / (x_max + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=pos_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        pos_z = z_embed / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(),
                             pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(),
                             pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(),
                             pos_z[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=1)

        pos = self.token_projection(pos)
        return pos


class IntraPatchCost(nn.Module):
    def __init__(self, voxel_size=0.25, resolution=6, out_channel=64, attn_temp=1.0):
        super(IntraPatchCost, self).__init__()
        self.out_channel = out_channel
        self.attn_temp = attn_temp
        self.voxel_size = voxel_size
        self.resolution = resolution

        self.pec = PositionalEncodingFourier(8, self.out_channel)

        self.input_embed = nn.Sequential(  # input.shape = [N*K, L]
            nn.Linear(3, self.out_channel),
            Rearrange('n k c -> n c k'),
            nn.BatchNorm1d(self.out_channel),
            Rearrange('n c k -> n k c'),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channel, self.out_channel))

        self.pre_mlp = nn.Sequential(
            nn.Linear(self.out_channel, self.out_channel),
            Rearrange('n k c -> n c k'),
            nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
            Rearrange('n c k -> n k c'),
            nn.ReLU(),
            nn.Linear(self.out_channel, self.out_channel),
            Rearrange('n k c -> n c k'),
            nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
            Rearrange('n c k -> n k c'),
            nn.ReLU(),
            nn.Linear(self.out_channel, self.out_channel // 2),
            Rearrange('n k c -> n c k'),
            nn.BatchNorm1d(self.out_channel // 2, eps=1e-3, momentum=0.01),
            Rearrange('n c k -> n k c'),
            nn.Linear(self.out_channel // 2, 1),
            Rearrange('n k c -> n c k'),
            nn.BatchNorm1d(1, eps=1e-3, momentum=0.01),
        )
        # self.score = nn.Linear(self.out_channel//2, 1)

    def forward(self, pos_diff):
        '''
        pos_diff: [B,3,N,S]
        feat: [B,C,N]
        '''
        B, C, S, N = pos_diff.shape
        pos_diff = pos_diff.permute(0, 3, 2, 1).reshape(-1, S, C).contiguous()
        with torch.no_grad():
            r = self.voxel_size
            dis_voxel = torch.round(pos_diff / r)  # sub-voxel
            valid_scatter = (torch.abs(dis_voxel) <= np.floor(self.resolution / 2)).all(dim=-1)  # [m,k,3]
            valid_scatter = valid_scatter.detach()

        inp_feats = self.input_embed(pos_diff)  # X,Y,Z, shape=[N,C]
        pe_raw = (pos_diff - dis_voxel * r) / r
        pe_raw = pe_raw.view(-1, 3)
        inp_feats = inp_feats + (self.pec(pe_raw)).reshape(B * N, S, -1)
        inp_feats = self.pre_mlp(inp_feats)
        # attn = F.softmax(self.score(inp_feats)) * valid_scatter.view(-1, 1)
        # attn = F.gumbel_softmax(inp_feats.squeeze(), tau=self.attn_temp, hard=False, dim=-1) #* valid_scatter
        attn = F.softmax(inp_feats.squeeze(), dim=-1)

        # return attn.reshape(B,N,S).permute(0,2,1).contiguous()
        return inp_feats.permute(0, 2, 1).reshape(B, N, S, self.out_channel).contiguous(), attn.permute(0, 2,
                                                                                                        1).reshape(B, N,
                                                                                                                   S,
                                                                                                                   self.out_channel).contiguous()


class PointConvTransFlowV2(nn.Module):
    def __init__(self, nsample, in_channel, sf_channel, mlp, flow_mlp, bn=use_bn, use_leaky=True, use_flow=True):
        super(PointConvTransFlowV2, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.use_flow = use_flow
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel * 2
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.mlp_convs2 = nn.ModuleList()
        if bn:
            self.mlp_bns2 = nn.ModuleList()
        last_channel = in_channel * 2
        for out_channel in mlp:
            self.mlp_convs2.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns2.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channel),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_channel, out_channel // 2, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channel // 2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_channel // 2, 1, kernel_size=1))
        self.softmax = nn.Softmax(dim=2)

        self.mlp_convs3 = nn.ModuleList()
        if bn:
            self.mlp_bns3 = nn.ModuleList()
        last_channel = last_channel + sf_channel + 3
        for out_channel in mlp:
            self.mlp_convs3.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns3.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.mlp_convs4 = nn.ModuleList()
        if bn:
            self.mlp_bns4 = nn.ModuleList()
        last_channel = last_channel * 2 + sf_channel + 3
        for out_channel in mlp:
            self.mlp_convs4.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns4.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        last_channel = out_channel
        self.flow_mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(flow_mlp):
            self.flow_mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out
        if add_Seg_after_FLow:
            self.fc = nn.Conv1d(last_channel, 4, 1)  # 12032881
        else:
            self.fc = nn.Conv1d(last_channel, 3, 1)

    @staticmethod
    def calculate_corr(fmap1, fmap2):
        corr = torch.matmul(fmap1, fmap2.permute(0, 1, 3, 2).contiguous())  # B, N1, nsample, nsmaple
        corr = corr / torch.sqrt(torch.tensor(fmap1.shape[-1]).float())  # N, K , K
        return corr[:, :, 0, :]

    def forward(self, xyz1, xyz2, xyz2w, points1, points2, sf=None, sf_feat=None):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()
        if xyz2w != None:
            xyz2w = xyz2w.permute(0, 2, 1).contiguous()
        else:
            xyz2w = xyz2
        points1 = points1.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()

        # point-to-patch Volume
        # _, knn_idx = pointutils.knn(self.nsample, xyz1, xyz2) # B, N1, nsample
        # if add_Seg and sf != None:  # 12032881
        #     sf = sf[:, 0: 3, :]

        if sf != None and self.use_flow:
            sf = sf.permute(0, 2, 1).contiguous()
            if add_Seg_after_FLow:  # 12032881
                _, knn_idx = pointutils.knn(self.nsample, xyz1 + sf[:, :, 0: 3], xyz2)
            else:
                _, knn_idx = pointutils.knn(self.nsample, xyz1 + sf, xyz2)
        else:
            _, knn_idx = pointutils.knn(self.nsample, xyz1, xyz2)
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx)  # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)

        new_points = torch.cat([grouped_points1, grouped_points2], dim=-1)  # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        _, knn_idxw = pointutils.knn(self.nsample, xyz1, xyz2w)
        neighbor_xyzw = index_points_group(xyz2, knn_idxw)
        direction_xyzw = neighbor_xyzw - xyz1.view(B, N1, 1, C)
        grouped_points2w = index_points_group(points2, knn_idxw)  # B, N1, nsample, D2

        new_pointsw = torch.cat([grouped_points1, grouped_points2w], dim=-1)  # B, N1, nsample, D1+D2+3
        new_pointsw = new_pointsw.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs2):
            if self.bn:
                bn = self.mlp_bns2[i]
                new_pointsw = self.relu(bn(conv(new_pointsw)))
            else:
                new_pointsw = self.relu(conv(new_pointsw))

        weight_qk = torch.matmul(new_points.permute(0, 3, 2, 1).contiguous(),
                                 new_pointsw.permute(0, 3, 1, 2).contiguous())  # B, N1, nsample, nsmaple
        weight_qk = torch.softmax(weight_qk, -2) * torch.softmax(weight_qk, -1)

        if sf_feat != None:
            sf_feat = sf_feat.permute(0, 2, 1).contiguous()
            grouped_sf_feats = sf_feat.view(B, N1, 1, sf_feat.shape[-1]).repeat(1, 1, self.nsample, 1).permute(0, 3, 2,
                                                                                                               1).contiguous()
            new_points_cost = torch.cat([new_points, grouped_sf_feats, direction_xyz.permute(0, 3, 2, 1).contiguous()],
                                        dim=1)  # B, N1, nsample, D1+D2+3
            new_pointsw_cost = torch.cat(
                [new_pointsw, grouped_sf_feats, direction_xyzw.permute(0, 3, 2, 1).contiguous()],
                dim=1)  # B, N1, nsample, D1+D2+3
        else:
            new_points_cost = torch.cat([new_points, direction_xyz.permute(0, 3, 2, 1).contiguous()],
                                        dim=1)  # B, N1, nsample, D1+D2+3
            new_pointsw_cost = torch.cat([new_pointsw, direction_xyzw.permute(0, 3, 2, 1).contiguous()],
                                         dim=1)  # B, N1, nsample, D1+D2+3

        for i, conv in enumerate(self.mlp_convs3):
            if self.bn:
                bn = self.mlp_bns3[i]
                new_points_cost = self.relu(bn(conv(new_points_cost)))
            else:
                new_points_cost = self.relu(conv(new_points_cost))

        for i, conv in enumerate(self.mlp_convs3):
            if self.bn:
                bn = self.mlp_bns3[i]
                new_pointsw_cost = self.relu(bn(conv(new_pointsw_cost)))
            else:
                new_pointsw_cost = self.relu(conv(new_pointsw_cost))

        new_points_ = new_points + torch.matmul(weight_qk, new_pointsw.permute(0, 3, 2, 1).contiguous()).permute(0, 3,
                                                                                                                 2,
                                                                                                                 1).contiguous()  # B N S C
        new_pointsw_ = new_pointsw + torch.matmul(new_points.permute(0, 3, 1, 2).contiguous(), weight_qk).permute(0, 2,
                                                                                                                  3,
                                                                                                                  1).contiguous()  # (B N C S)

        weight_feats = self.weightnet1(new_points_)  # B C nsample N1
        weight_featsw = self.weightnet1(new_pointsw_)  # B C nsample N1

        # weight_qk = torch.matmul(new_points_cost.permute(0,3,2,1).contiguous(), new_pointsw_cost.permute(0,3,1,2).contiguous()) # B, N1, nsample, nsmaple
        # weight_qk = torch.softmax(weight_qk, -2) * torch.softmax(weight_qk, -1)

        # weight_feats  = torch.matmul(weight_qk, new_pointsw_cost.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous() # B N S C
        # weight_featsw = torch.matmul(new_points_cost.permute(0,3,1,2).contiguous(), weight_qk).permute(0,2,3,1).contiguous() # (B N C S)

        weights1 = self.softmax(weight_feats)

        knn_idxw_flatten = knn_idxw.view(B, -1).long()
        point_to_patch_costw_flatten = new_pointsw_cost.permute(0, 3, 2, 1).reshape(B, -1,
                                                                                    new_points_cost.shape[1])  # [B,N,C]
        weight_bwd = scatter_softmax(weight_featsw.permute(0, 3, 2, 1).reshape(B, -1, weight_featsw.shape[1]),
                                     knn_idxw_flatten, dim=1)
        weight_bwd_cpu = weight_bwd.cpu()
        if weight_bwd_cpu[0, 0, 0] == None:
            print('!!!Error...')

        point_to_patch_costw_flatten = point_to_patch_costw_flatten * weight_bwd
        point_to_patch_cost_bwd = scatter_sum(point_to_patch_costw_flatten, knn_idxw_flatten, dim=1)
        point_to_patch_cost_bwd_cpu = point_to_patch_cost_bwd.cpu()
        if point_to_patch_cost_bwd_cpu[0, 0, 0] == None:
            print('!!!Error...')

        point_to_patch_cost_fwd = torch.sum(weights1 * new_points_cost, dim=2)  # B C N

        # weights for group cost
        grouped_cost_bwd = index_points_group(point_to_patch_cost_bwd, knn_idx)  # B, N1, nsample, D2
        grouped_cost_fwd = point_to_patch_cost_fwd.view(B, N1, 1, point_to_patch_cost_fwd.shape[1]).repeat(1, 1,
                                                                                                           self.nsample,
                                                                                                           1)
        if sf_feat != None:
            grouped_point_to_patch_cost = torch.cat(
                [grouped_cost_fwd, grouped_cost_bwd, grouped_sf_feats.permute(0, 3, 2, 1).contiguous(), direction_xyz],
                dim=-1)  # B, N1, nsample, D1+D2+3
        else:
            grouped_point_to_patch_cost = torch.cat([grouped_cost_fwd, grouped_cost_bwd, direction_xyz],
                                                    dim=-1)  # B, N1, nsample,

        grouped_point_to_patch_cost = grouped_point_to_patch_cost.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]

        for i, conv in enumerate(self.mlp_convs4):
            if self.bn:
                bn = self.mlp_bns4[i]
                grouped_point_to_patch_cost = self.relu(bn(conv(grouped_point_to_patch_cost)))
            else:
                grouped_point_to_patch_cost = self.relu(conv(grouped_point_to_patch_cost))
        patch_to_patch_cost = torch.max(grouped_point_to_patch_cost, dim=2)[0]

        for conv in self.flow_mlp_convs:
            patch_to_patch_cost = conv(patch_to_patch_cost)

        re_sf = self.fc(patch_to_patch_cost)
        re_sf = re_sf.clamp(-50.0, 50.0)

        if sf is not None:
            # if add_Seg:  # 12032881
            #     re_sf = re_sf[:, 0: 3, :]
                # print(re_sf[:, 0: 3, :].size()
            re_sf = re_sf + sf.permute(0, 2, 1).contiguous()

        return point_to_patch_cost_fwd, point_to_patch_cost_bwd.permute(0, 2,
                                                                        1).contiguous(), patch_to_patch_cost, re_sf.clamp(
            -50.0, 50.0)


class Conv1DBn(nn.Module):
    def __init__(self, in_channel, mlp, kernel_size=1, use_leaky=True, bn=use_bn):
        super(Conv1DBn, self).__init__()
        # self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                x = self.relu(bn(conv(x)))
            else:
                x = self.relu(conv(x))
        return x


class PointConvTransFlowV3(nn.Module):
    def __init__(self, nsample, in_channel, mlp, voxel_size=0.25, resolution=6, attn_temp=1.0, bn=use_bn,
                 use_leaky=True, use_flow=True):
        super(PointConvTransFlowV3, self).__init__()
        self.attn_temp = attn_temp
        self.nsample = nsample
        self.bn = bn
        self.use_flow = use_flow
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel * 2
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.mlp_convs2 = nn.ModuleList()
        if bn:
            self.mlp_bns2 = nn.ModuleList()
        last_channel = in_channel * 2
        for out_channel in mlp:
            self.mlp_convs2.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns2.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # self.bilinear = nn.Bilinear(last_channel, last_channel, last_channel)
        self.mlp_convs3 = nn.ModuleList()
        if bn:
            self.mlp_bns3 = nn.ModuleList()
        last_channel = last_channel + 3
        for out_channel in mlp:
            self.mlp_convs3.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns3.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.mlp_convs4 = nn.ModuleList()
        if bn:
            self.mlp_bns4 = nn.ModuleList()

        last_channel = last_channel * 2

        for out_channel in mlp:
            self.mlp_convs4.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns4.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # self.weightnet1 = IntraPatchCost(voxel_size=voxel_size, resolution=resolution, out_channel=64, attn_temp=attn_temp)
        self.weightnet2 = IntraPatchCost(voxel_size=voxel_size, resolution=resolution, out_channel=64,
                                         attn_temp=attn_temp)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    @staticmethod
    def calculate_corr(fmap1, fmap2):
        corr = torch.matmul(fmap1, fmap2.permute(0, 1, 3, 2).contiguous())  # B, N1, nsample, nsmaple
        corr = corr / torch.sqrt(torch.tensor(fmap1.shape[-1]).float())  # N, K , K
        return corr[:, :, 0, :]

    def forward(self, xyz1, xyz2, xyz2w, points1, points2, sf=None):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()
        if xyz2w != None:
            xyz2w = xyz2w.permute(0, 2, 1).contiguous()
        else:
            xyz2w = xyz2
        points1 = points1.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()

        # point-to-patch Volume
        # knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        if sf != None and self.use_flow:
            sf = sf.permute(0, 2, 1).contiguous()
            _, knn_idx = pointutils.knn(self.nsample, xyz1 + sf, xyz2)
        else:
            _, knn_idx = pointutils.knn(self.nsample, xyz1, xyz2)
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        _, knn_idxw = pointutils.knn(self.nsample, xyz1, xyz2w)
        neighbor_xyzw = index_points_group(xyz2, knn_idxw)
        direction_xyzw = neighbor_xyzw - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx)  # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2], dim=-1)  # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]

        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        grouped_points2w = index_points_group(points2, knn_idxw)  # B, N1, nsample, D2
        new_pointsw = torch.cat([grouped_points1, grouped_points2w], dim=-1)  # B, N1, nsample, D1+D2+3
        new_pointsw = new_pointsw.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs2):
            if self.bn:
                bn = self.mlp_bns2[i]
                new_pointsw = self.relu(bn(conv(new_pointsw)))
            else:
                new_pointsw = self.relu(conv(new_pointsw))

        new_points_cost = torch.cat([new_points, direction_xyz.permute(0, 3, 2, 1).contiguous()],
                                    dim=1)  # B, N1, nsample, D1+D2+3
        new_pointsw_cost = torch.cat([new_pointsw, direction_xyzw.permute(0, 3, 2, 1).contiguous()],
                                     dim=1)  # B, N1, nsample, D1+D2+3

        for i, conv in enumerate(self.mlp_convs3):
            if self.bn:
                bn = self.mlp_bns3[i]
                new_points_cost = self.relu(bn(conv(new_points_cost)))
            else:
                new_points_cost = self.relu(conv(new_points_cost))

        for i, conv in enumerate(self.mlp_convs3):
            if self.bn:
                bn = self.mlp_bns3[i]
                new_pointsw_cost = self.relu(bn(conv(new_pointsw_cost)))
            else:
                new_pointsw_cost = self.relu(conv(new_pointsw_cost))

        weight_qk = torch.matmul(new_points.permute(0, 3, 2, 1).contiguous(), new_pointsw.permute(0, 3, 1,
                                                                                                  2).contiguous()) / self.attn_temp  # B, N1, nsample, nsmaple
        weight_qk = torch.softmax(weight_qk, -2) * torch.softmax(weight_qk, -1)

        new_pointsw_cost = torch.matmul(weight_qk, new_pointsw_cost.permute(0, 3, 2, 1).contiguous())  # B N S C
        new_points_cost = torch.matmul(new_points_cost.permute(0, 3, 1, 2).contiguous(), weight_qk)  # (B N C S)

        point_to_patch_costw = torch.sum(new_pointsw_cost, dim=-2)
        point_to_patch_cost = torch.sum(new_points_cost, dim=-1)

        # Patch to Patch Cost
        # knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        _, knn_idx = pointutils.knn(self.nsample, xyz1, xyz1)
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1

        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost,
                                                         knn_idx)  # B, N1, nsample, C
        grouped_point_to_patch_costw = index_points_group(point_to_patch_costw,
                                                          knn_idx)  # B, N1, nsample, C
        grouped_point_to_patch_cost = torch.cat([grouped_point_to_patch_cost, grouped_point_to_patch_costw], dim=-1)

        grouped_point_to_patch_cost = grouped_point_to_patch_cost.permute(0, 3, 2, 1).contiguous()
        for i, conv in enumerate(self.mlp_convs4):
            if self.bn:
                bn = self.mlp_bns4[i]
                grouped_point_to_patch_cost = self.relu(bn(conv(grouped_point_to_patch_cost)))
            else:
                grouped_point_to_patch_cost = self.relu(conv(grouped_point_to_patch_cost))

        patch_to_patch_cost = torch.sum(weights.unsqueeze(1) * grouped_point_to_patch_cost, dim=2)  # B C N

        return patch_to_patch_cost


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=0):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class reg3d(nn.Module):
    def __init__(self, in_channels, mlp=[64, 64, 64], bias=False):
        super(reg3d, self).__init__()

        self.prob_convs = nn.ModuleList()
        last_ch = in_channels
        for _, ch_out in enumerate(mlp):
            self.prob_convs.append(ConvBnReLU(last_ch, ch_out, kernel_size=1))
            last_ch = ch_out

    def forward(self, x):
        for conv in self.prob_convs:
            x = conv(x)

        return x


class SceneFlowEstimatorProbPointConv3(nn.Module):
    def __init__(self, in_channel, feat_ch, cost_ch, flow_ch=3, channels=[128, 128], mlp=[128, 64], neighbors=9,
                 clamp=[-20, 20], use_bn=True,
                 use_leaky=True, use_flow=True, use_flow_feats=True):
        super(SceneFlowEstimatorProbPointConv3, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        self.use_bn = use_bn
        self.use_flow = use_flow
        self.use_flow_feats = use_flow_feats
        self.nsample = neighbors

        self.pointconv_list = nn.ModuleList()
        self.use_flow = use_flow
        if use_flow:
            last_channel = feat_ch + cost_ch + flow_ch
        else:
            last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv2(neighbors, last_channel + 3, ch_out, bn=True, use_leaky=True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)
        # self.regnet = reg3d(last_channel, mlp=[64, 64, 3])
        # self.coarse_regnet = reg3d(cost_ch, mlp=[64, 64, 3])
        self.regnet = reg3d(last_channel, mlp=[64, 64, 1])
        self.coarse_regnet = reg3d(cost_ch, mlp=[64, 64, 1])

    def forward(self, xyz, cost_volume, feats=None, flow=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if self.use_flow:
            # if feats is None:
            #     new_points = cost_volume
            # else:
            new_points = torch.cat([feats, cost_volume, flow], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        re_flow = self.fc(new_points)
        # re_flow = re_flow.clamp(self.clamp[0], self.clamp[1])

        if flow is not None:
            # refine_prob = self.regnet(new_points)
            # coarse_prob = self.coarse_regnet(cost_volume)
            # re_flow = coarse_prob * flow + refine_prob * re_flow
            # if flow is not None:
            flow = re_flow + flow

        return new_points, re_flow.clamp(self.clamp[0], self.clamp[1])


class SceneFlowEstimatorProbPointConv2(nn.Module):
    def __init__(self, in_channel, feat_ch, cost_ch, flow_ch=3, channels=[128, 128], mlp=[128, 64], neighbors=9,
                 clamp=[-20, 20], use_bn=True,
                 use_leaky=True, use_flow=True, use_flow_feats=True):
        super(SceneFlowEstimatorProbPointConv2, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        self.use_bn = use_bn
        self.use_flow = use_flow
        self.use_flow_feats = use_flow_feats
        self.nsample = neighbors
        if use_flow:
            # if use_flow_feats:
            #     last_channel = cost_ch + feat_ch +  flow_ch
            # else:
            last_channel = cost_ch + flow_ch
        else:
            # if use_flow_feats:
            #     last_channel = cost_ch + feat_ch
            # else:
            last_channel = cost_ch

        # if use_flow_feats:
        self.mlp_convs = nn.ModuleList()
        if self.use_bn:
            self.mlp_bns = nn.ModuleList()
        for out_channel in channels:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if self.use_bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.gru = ConvGRU(input_dim=feat_ch, hidden_dim=last_channel)

        self.mlp_convs2 = nn.ModuleList()
        if use_flow_feats:
            last_channel = in_channel + last_channel
        else:
            last_channel = cost_ch + in_channel
        for _, ch_out in enumerate(mlp):
            self.mlp_convs2.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)
        # self.regnet = reg3d(last_channel, mlp=[64, 64, 3])
        # self.coarse_regnet = reg3d(cost_ch, mlp=[64, 64, 3])
        self.regnet = reg3d(last_channel, mlp=[64, 64, 1])
        self.coarse_regnet = reg3d(cost_ch, mlp=[64, 64, 1])
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points, cost_volume, feats=None, flow=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if self.use_flow:
            # if self.use_flow_feats:
            #     new_points = torch.cat([feats, cost_volume, flow], dim=1)
            # else:
            new_points = torch.cat([cost_volume, flow], dim=1)
        else:
            # if self.use_flow_feats:
            #     # last_channel = cost_ch + feat_ch
            #     new_points = torch.cat([feats, cost_volume], dim=1)
            # else:
            new_points = cost_volume

        B, _, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        _, idx = pointutils.knn(self.nsample, xyz_t, xyz_t)
        # grouped_xyz = pointutils.grouping_operation(xyz, idx)
        new_points = pointutils.grouping_operation(new_points, idx)
        # grouped_xyz_norm = grouped_xyz - xyz.view(B, 3, N, 1)
        # new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1)

        for i, conv in enumerate(self.mlp_convs):
            if self.use_bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        new_points = torch.max(new_points, dim=-1)[0]

        if self.use_flow_feats:
            new_points = self.gru(feats, new_points)

        if self.use_flow_feats:
            new_points = torch.cat([points, new_points], dim=1)
        else:
            new_points = torch.cat([points, cost_volume], dim=1)

        for conv in self.mlp_convs2:
            new_points = conv(new_points)

        re_flow = self.fc(new_points)
        # re_flow = re_flow.clamp(self.clamp[0], self.clamp[1])

        if flow is not None:
            refine_prob = self.regnet(new_points)
            coarse_prob = self.coarse_regnet(cost_volume)
            re_flow = coarse_prob * flow + refine_prob * re_flow
        # if flow is not None:
        # re_flow = re_flow + flow

        return new_points, re_flow.clamp(self.clamp[0], self.clamp[1])


class SceneFlowEstimatorProbPointConv(nn.Module):
    def __init__(self, in_channel, feat_ch, cost_ch, flow_ch=3, channels=[128, 128], mlp=[128, 64], neighbors=9,
                 clamp=[-20, 20],
                 use_leaky=True, use_flow_feats=True):
        super(SceneFlowEstimatorProbPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        self.use_bn = use_bn
        self.use_flow_feats = use_flow_feats
        self.nsample = neighbors
        if use_flow_feats:
            # if self.use_flow_feats:
            last_channel = in_channel + feat_ch + cost_ch + flow_ch
            # else:
            # last_channel = in_channel + feat_ch + cost_ch + flow_ch
        else:
            last_channel = in_channel + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv2(neighbors, last_channel + 3, ch_out, bn=True, use_leaky=True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)
        # self.regnet = reg3d(last_channel, mlp=[64, 64, 3])
        # self.coarse_regnet = reg3d(cost_ch, mlp=[64, 64, 3])
        self.regnet = reg3d(last_channel, mlp=[64, 64, 1])
        self.coarse_regnet = reg3d(cost_ch, mlp=[64, 64, 1])

    def forward(self, xyz, points, cost_volume, feats=None, flow=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if self.use_flow_feats:
            # if feats is None:
            #     new_points = torch.cat([points, cost_volume, flow], dim=1)
            # else:
            new_points = torch.cat([points, feats, cost_volume, flow], dim=1)
        else:
            new_points = torch.cat([points, cost_volume], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        re_flow = self.fc(new_points)
        # re_flow = re_flow.clamp(self.clamp[0], self.clamp[1])

        # if flow is not None:
        #     refine_prob = self.regnet(new_points)
        #     coarse_prob = self.coarse_regnet(cost_volume)
        #     re_flow = coarse_prob * flow + refine_prob * re_flow
        # # if flow is not None:
        #     # flow = re_flow + flow

        return new_points, re_flow.clamp(self.clamp[0], self.clamp[1])


class SceneFlowEstimatorProbPointConvV2(nn.Module):
    def __init__(self, feat_ch, cost_ch, flow_ch=3, channels=[128, 128], mlp=[128, 64], neighbors=9, clamp=[-20, 20],
                 use_leaky=True, use_flow=False, use_bn=True):
        super(SceneFlowEstimatorProbPointConvV2, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.use_bn = use_bn
        # self.pointconv_list = nn.ModuleList()
        if use_flow:
            last_channel = feat_ch + cost_ch + flow_ch
        else:
            last_channel = feat_ch + cost_ch
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        # for _, ch_out in enumerate(channels):
        # pointconv = PointConv2(neighbors, last_channel+3, ch_out, bn=True, use_leaky=True)
        #     self.pointconv_list.append(pointconv)
        #     last_channel = ch_out

        # self.mlp_convs = nn.ModuleList()
        # for _, ch_out in enumerate(mlp):
        #     self.mlp_convs.append(Conv1d(last_channel, ch_out))
        #     last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        if self.use_bn:
            self.mlp_bns = nn.ModuleList()

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            if self.use_bn:
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.fc = nn.Conv1d(last_channel, 3, 1)
        # self.regnet = reg3d(last_channel, mlp=[64, 64, 3])
        # self.coarse_regnet = reg3d(cost_ch, mlp=[64, 64, 3])
        self.regnet = reg3d(last_channel, mlp=[64, 64, 1])
        self.coarse_regnet = reg3d(cost_ch, mlp=[64, 64, 1])

    def forward(self, xyz, cost_volume, feats=None, flow=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim=1)

        # for _, pointconv in enumerate(self.pointconv_list):
        #     new_points = pointconv(xyz, new_points)

        # for conv in self.mlp_convs:
        #     new_points = conv(new_points)

        for i, conv in enumerate(self.mlp_convs):
            if self.use_bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        re_flow = self.fc(new_points)

        if flow is not None:
            refine_prob = self.regnet(new_points)
            coarse_prob = self.coarse_regnet(cost_volume)
            re_flow = coarse_prob * flow + refine_prob * re_flow

        return new_points, re_flow.clamp(self.clamp[0], self.clamp[1])


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


def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    _, idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


class PointConv2(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConv2, self).__init__()
        self.bn = bn
        self.nsample = nsample
        # self.weightnet = WeightNet(3, 1)
        # self.softmax = nn.Softmax(dim=1)
        # self.linear = nn.Linear(in_channel, out_channel)
        self.linear = nn.Conv2d(in_channel, out_channel, 1)
        if bn:
            self.bn_linear = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, C, N = xyz.shape

        xyz_t = xyz.permute(0, 2, 1).contiguous()
        # points_t = points.permute(0, 2, 1).contiguous()

        # new_points, grouped_xyz_norm = group(self.nsample, xyz_t, points_t)
        # grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        _, idx = pointutils.knn(self.nsample, xyz_t, xyz_t)
        grouped_xyz = pointutils.grouping_operation(xyz, idx)
        grouped_points = pointutils.grouping_operation(points, idx)
        grouped_xyz_norm = grouped_xyz - xyz.view(B, C, N, 1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1)

        # weights = self.softmax(self.weightnet(grouped_xyz_norm))
        # # new_points = torch.matmul(input=new_points.permute(0, 2, 1, 3), other=weights.permute(0, 2, 3, 1)).view(B,N,-1)
        # new_points = torch.matmul(input=new_points.permute(0, 2, 1, 3), other=weights.permute(0, 2, 3, 1)).squeeze(-1).contiguous()
        new_points = self.linear(new_points)
        if self.bn:
            # new_points = self.bn_linear(new_points.permute(0, 2, 1))
            new_points = self.bn_linear(new_points)
        # else:
        #     # new_points = new_points.permute(0, 2, 1)
        #     new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)
        new_points = torch.max(new_points, dim=-1)[0]

        return new_points


class PointWarping(nn.Module):
    def forward(self, pos1, pos2, flow1=None, nsample=None):
        if flow1 is None:
            return pos2

        # move pos1 to pos2'
        if add_Seg_after_FLow:
            pos1_to_2 = pos1 + flow1[:, 0:3, :]  # 12032881
        else:
            pos1_to_2 = pos1 + flow1  # 12032881

        # interpolate flow
        B, C, N1 = pos1.shape
        _, _, N2 = pos2.shape
        pos1_to_2_t = pos1_to_2.permute(0, 2, 1).contiguous()  # B 3 N1
        pos2_t = pos2.permute(0, 2, 1).contiguous()  # B 3 N2
        # flow1_t = flow1.permute(0, 2, 1).contiguous()
        if nsample is None:
            nsample = 3
            _, knn_idx = pointutils.three_nn(pos2_t, pos1_to_2_t)
        else:
            _, knn_idx = pointutils.knn(nsample, pos2_t, pos1_to_2_t)
        grouped_pos_norm = pointutils.grouping_operation(pos1_to_2, knn_idx) - pos2.view(B, C, N2, 1)
        dist = torch.norm(grouped_pos_norm, dim=1).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow1 = pointutils.grouping_operation(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, 1, N2, nsample) * grouped_flow1, dim=-1)

        if add_Seg_after_FLow: # 12032881
            warped_pos2 = pos2 - flow2[:, 0: 3,:] # 12032881
        else:
            warped_pos2 = pos2 - flow2  # B 3 N2

        return warped_pos2.clamp(-10.0, 10.0)


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


class SceneFlowEstimatorPointConv(nn.Module):
    def __init__(self, feat_ch, cost_ch, flow_ch=3, channels=[128, 128], mlp=[128, 64], neighbors=9, clamp=[-20, 20],
                 use_leaky=True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv2(neighbors, last_channel + 3, ch_out, bn=True, use_leaky=True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, if_IN=False, IN_affine=False,
         if_BN=False):
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


class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn=use_bn, use_leaky=True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel * 2 + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()
        points1 = points1.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()

        # point-to-patch Volume
        # knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        _, knn_idx = pointutils.knn(self.nsample, xyz1, xyz2)
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx)  # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim=-1)  # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1

        point_to_patch_cost = torch.sum(weights * new_points, dim=2)  # B C N

        # Patch to Patch Cost
        # knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        _, knn_idx = pointutils.knn(self.nsample, xyz1, xyz1)
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1),
                                                         knn_idx)  # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim=2)  # B C N

        return patch_to_patch_cost


class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow, k=3):
        '''
        :param xyz: [B,C,N]
        :param sparse_xyz: [B,C,N]
        :param sparse_flow: [B,C,N]
        :return: [B,C,N]
        '''
        # import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz_t = xyz.permute(0, 2, 1).contiguous()  # B N 3
        sparse_xyz_t = sparse_xyz.permute(0, 2, 1).contiguous()  # B S 3
        # sparse_flow_t = sparse_flow.permute(0, 2, 1).contiguous() # B S 3
        # knn_idx = knn_point(3, sparse_xyz, xyz)
        if k == 3:
            _, knn_idx = pointutils.three_nn(xyz_t, sparse_xyz_t)
        else:
            _, knn_idx = pointutils.knn(k, xyz_t, sparse_xyz_t)
        grouped_xyz_norm = pointutils.grouping_operation(sparse_xyz, knn_idx) - xyz.view(B, C, N, 1)
        dist = torch.norm(grouped_xyz_norm, dim=1).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=-1, keepdim=True)
        weight = (1.0 / dist) / norm
        # vv = torch.max(weight)
        # if vv > 1e4:
        #     print(torch.max(weight))

        grouped_flow = pointutils.grouping_operation(sparse_flow.float(), knn_idx)
        dense_flow = torch.sum(weight.view(B, 1, N, k) * grouped_flow, dim=-1)  # [B,C,N]
        # vv = torch.max(dense_flow)
        # if vv > 1e4:
        #     print(torch.max(dense_flow))
        return dense_flow.clamp(-100.0, 100.0)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=True, use_bn=True, use_leaky=True):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # TODO：
        last_channel = last_channel * 2 * 4
        # relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        if self.nsample == 8:
            for i, out_channel in enumerate(mlp):
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=2, bias=False))
                if use_bn:
                    self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                else:
                    self.mlp_bns.append(nn.Identity())
                last_channel = out_channel

            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False))
            if use_bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            else:
                self.mlp_bns.append(nn.Identity())
        else:
            for i, out_channel in enumerate(mlp):
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=3, bias=False))
                if use_bn:
                    self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                else:
                    self.mlp_bns.append(nn.Identity())
                last_channel = out_channel

            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=2, bias=False))
            if use_bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            else:
                self.mlp_bns.append(nn.Identity())

        if group_all:
            self.queryandgroup = pointutils.GroupAll()
        else:
            self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

    def forward(self, xyz, feats):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, C, N = feats.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        # if points is not None:
        #     points = points.permute(0, 2, 1).contiguous()

        # 选取邻域点
        if self.group_all == False:
            fps_idx = pointutils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
            new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, N]
        else:
            new_xyz = xyz

        new_xyz_t = new_xyz.permute(0, 2, 1).contiguous()
        if self.knn:
            _, idx = pointutils.knn(self.nsample, new_xyz_t, new_xyz_t)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, new_xyz_t, new_xyz_t)

        pos_grouped = pointutils.grouping_operation(new_xyz, idx)  # [B,3,N,K]
        feats_grouped = pointutils.grouping_operation(feats, idx)  # [B,C,N,K]
        pos_diff = pos_grouped - new_xyz.view(B, 3, self.npoint, 1)
        feats_grouped = torch.cat([pos_diff, feats_grouped], dim=1)
        # Structure
        feats_grouped_corr = torch.cat([feats_grouped.unsqueeze(-1),
                                        torch.ones_like(feats_grouped.unsqueeze(-1))], dim=1)

        feats_grouped_corr_t = torch.cat([feats_grouped.unsqueeze(-2),
                                          torch.ones_like(feats_grouped.unsqueeze(-2))], dim=1)
        mat_feats = torch.matmul(feats_grouped_corr, feats_grouped_corr_t)  # [B,C,N,K1,K2]
        mat_feats = mat_feats.permute(0, 2, 1, 3, 4).reshape(-1, 2 * (C + 3), self.nsample,
                                                             self.nsample)  # [BN,C,K1,K2]
        mat_feats = squeeze_2x2(mat_feats, reverse=False, alt_order=True)
        # feats = self.feats_net(mat_feats)
        # new_xyz: sampled points position data, [B, C, npoint]
        # new_points: sampled points data, [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            mat_feats = F.relu(bn(conv(mat_feats)))
        new_points = mat_feats.squeeze().view(B, self.npoint, -1).permute(0, 2, 1).contiguous()
        return new_xyz, new_points, fps_idx


class PointNetSetUpConv(nn.Module):
    def __init__(self, nsample, radius, f1_channel, f2_channel, mlp, mlp2, knn=True):
        super(PointNetSetUpConv, self).__init__()
        self.nsample = nsample
        self.radius = radius
        self.knn = knn
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = f2_channel + 3
        for out_channel in mlp:
            self.mlp1_convs.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
        if len(mlp) != 0:
            last_channel = mlp[-1] + f1_channel
        else:
            last_channel = last_channel + f1_channel
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm1d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
            Feature propagation from xyz2 (less points) to xyz1 (more points)
        Inputs:
            xyz1: (batch_size, 3, npoint1)
            xyz2: (batch_size, 3, npoint2)
            feat1: (batch_size, channel1, npoint1) features for xyz1 points (earlier layers, more points)
            feat2: (batch_size, channel1, npoint2) features for xyz2 points
        Output:
            feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)

            TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, C, N = pos1.shape
        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, pos2_t, pos1_t)

        pos2_grouped = pointutils.grouping_operation(pos2, idx)
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)  # [B,3,N1,S]

        feat2_grouped = pointutils.grouping_operation(feature2, idx)
        feat_new = torch.cat([feat2_grouped, pos_diff], dim=1)  # [B,C1+3,N1,S]
        for conv in self.mlp1_convs:
            feat_new = conv(feat_new)
        # max pooling
        feat_new = feat_new.max(-1)[0]  # [B,mlp1[-1],N1]
        # concatenate feature in early layer
        if feature1 is not None:
            feat_new = torch.cat([feat_new, feature1], dim=1)
        # feat_new = feat_new.view(B,-1,N,1)
        for conv in self.mlp2_convs:
            feat_new = conv(feat_new)

        return feat_new


class PointConv1DComposed(nn.Module):
    def __init__(self, nsample, in_channel, mlp, knn=True, bn=use_bn, use_leaky=True):
        super(PointConv1DComposed, self).__init__()
        self.in_channels = in_channel
        self.nsample = nsample
        self.knn = knn

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel  # TODO：
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, pos):
        """
                PointConv without strides size, i.e., the input and output have the same number of points.
                Input:
                    pos: input points position data, [B, C, N]
                Return:
                    new_pos: sampled points position data, [B, C, S]
                    new_points_concat: sample points feature data, [B, D', S]
                """
        B, C, N = pos.shape
        pos_t = pos.permute(0, 2, 1).contiguous()

        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos_t, pos_t)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, pos_t, pos_t)

        new_points = pointutils.grouping_operation(pos, idx)  # [B,3,N,K]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = bn(conv(new_points))

        new_points = torch.max(new_points, -1)[0]
        return new_points


class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, mlp, knn=True, bn=use_bn, use_leaky=True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.knn = knn
        self.nsample = nsample
        self.feats_convs = nn.ModuleList()
        self.feats_bns = nn.ModuleList()
        last_channel = in_channel * 4
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.feats_net = nn.Sequential(
            nn.Conv2d(last_channel, mlp[0], kernel_size=3, stride=1),
            nn.BatchNorm2d(mlp[0]) if bn else nn.Identity(),
            nn.Conv2d(mlp[0], mlp[1], kernel_size=2, stride=1),
            nn.BatchNorm2d(mlp[1]) if bn else nn.Identity(),
            # relu
        )

    def forward(self, pos):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            pos: input points position data, [B, C, N]
        Return:
            new_pos: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, C, N = pos.shape
        pos_t = pos.permute(0, 2, 1).contiguous()

        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos_t, pos_t)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, pos_t, pos_t)

        pos_grouped = pointutils.grouping_operation(pos, idx)  # [B,3,N,K]
        # Structure
        pos_grouped_corr = torch.cat([pos_grouped.unsqueeze(-1),
                                      torch.ones_like(pos_grouped.unsqueeze(-1))], dim=1)

        pos_grouped_corr_t = torch.cat([pos_grouped.unsqueeze(-2),
                                        torch.ones_like(pos_grouped.unsqueeze(-2))], dim=1)
        mat_pos = torch.matmul(pos_grouped_corr, pos_grouped_corr_t)  # [B,C,N,K1,K2]
        mat_pos = mat_pos.permute(0, 2, 1, 3, 4).reshape(-1, 2 * C, self.nsample, self.nsample)  # [BN,C,K1,K2]
        mat_pos = squeeze_2x2(mat_pos, reverse=False, alt_order=True)
        feats = self.feats_net(mat_pos)
        feats = feats.squeeze().view(B, N, -1).permute(0, 2, 1).contiguous()

        return feats


class PointConvMiniSqueeze(nn.Module):
    def __init__(self, nsample, in_channel, mlp, knn=True, bn=use_bn, use_leaky=True):
        super(PointConvMiniSqueeze, self).__init__()
        self.bn = bn
        self.knn = knn
        self.nsample = nsample
        self.feats_convs = nn.ModuleList()
        self.feats_bns = nn.ModuleList()
        last_channel = in_channel * 4
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.feats_net = nn.Sequential(
            nn.Conv2d(last_channel, mlp[0], kernel_size=2, stride=1),
            nn.BatchNorm2d(mlp[0]) if bn else nn.Identity(),
            nn.Conv2d(mlp[0], mlp[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(mlp[1]) if bn else nn.Identity(),
            # relu
        )

    def forward(self, pos, feats=None):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            pos: input points position data, [B, C, N]
        Return:
            new_pos: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, C, N = pos.shape
        pos_t = pos.permute(0, 2, 1).contiguous()

        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos_t, pos_t)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, pos_t, pos_t)

        if feats is not None:
            tmp_pos_grouped = pointutils.grouping_operation(feats, idx)
        else:
            tmp_pos_grouped = pointutils.grouping_operation(pos, idx)  # [B,3,N,K]
        pos_grouped = torch.zeros([B, C, N, 16], requires_grad=True)
        inds_pos = torch.tensor([0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9]).long().to(pos.device)
        inds_pos = inds_pos.view(1, 1, 1, 16).expand(B, C, N, 16)
        pos_grouped = torch.gather(input=tmp_pos_grouped, index=inds_pos, dim=-1)
        # Structure
        pos_grouped = pos_grouped.permute(0, 2, 1, 3).reshape(-1, 3, 16, 1).reshape(-1, 3, 4, 4)  # [BN,C,K1,K2]
        pos_grouped = squeeze_2x2(pos_grouped, reverse=False, alt_order=True)
        feats = self.feats_net(pos_grouped)
        feats = feats.squeeze().view(B, N, -1).permute(0, 2, 1).contiguous()

        return feats


if __name__ == "__main__":
    import os
    import sys
    import argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--n_points', type=int, default=2048)
    args = parser.parse_args()

    sparse_input = torch.randn(5, 3, 512).cuda()
    input = torch.randn(5, 3, 2048).cuda()
    f_input = torch.randn(5, 128, 2048).cuda()
    # label = torch.randn(4, 16)
    # (self, nsample, radius, in_channel, afn_mlp, fe_mlp, knn=True)
    model = SGUSceneFlowEstimatorMini(nsample=5, in_channel=128, flow_mlp=[128, 64, 64], ctx_mlp=[32, 16, 16]).cuda()
    # model = PointConvMiniSqueeze(nsample=10,in_channel=3, mlp=[16, 16], knn=True, bn=True, use_leaky=True).cuda()
    total_num_paras = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is %d\t" % (total_num_paras))
    device_ids = [0]
    if len(device_ids) == 0:
        net = nn.DataParallel(model)
    else:
        net = nn.DataParallel(model, device_ids=device_ids)
    print("Let's use ", len(device_ids), " GPUs!")
    # forward(self, pos1, pos2, feats1, feats2, sparse_pos1, sparse_flow):
    output = model(input, input, f_input, f_input, sparse_input, sparse_input)
    # output = model(sparse_input)
    print(output.shape)
