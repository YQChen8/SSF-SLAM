import torch
import torch.nn as nn

import numpy as np
from fractions import Fraction
from lib import pointnet2_utils as pointutils

from HPR import *

from collections import namedtuple
import itertools

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
# ActionRot = namedtuple('ActionRot', ['v', 'r'])

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    mask = group_idx != N
    cnt = mask.sum(dim=-1)
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    dists = sqrdists[group_idx]
    mask2 = 1 - mask
    return group_idx, cnt, dists, mask2

def DownsamplePoints(xyz, feats, npoint=20):
    """
    Input:
        xyz: input points position data, [B, C, N]
        feats: input points data, [B, D, N]
    Return:
        new_xyz: sampled points position data, [B, S, C]
        new_feats: sample points feature data, [B, S, D']
    """
    device = xyz.device
    B, C, N = xyz.shape
    xyz_t = xyz.permute(0, 2, 1).contiguous()
    fps_idx = pointutils.furthest_point_sample(xyz_t, npoint)  # [B, N]
    new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, N]
    new_feats = pointutils.gather_operation(feats, fps_idx)
    return new_xyz, new_feats

def ComputeNewPointCost(xyz, feats, idx):
    B, N, S = idx.shape
    xyz_grouped = pointutils.grouping_operation(xyz, idx)
    new_feats = (pointutils.grouping_operation(feats, idx[:,:,1].unsqueeze(-1).contiguous())).squeeze(-1)

    # 1-NN feature similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    feats_cost = cos(feats, new_feats)
    feats_cost = torch.exp(feats_cost).unsqueeze(1)
    feats_cost_grouped = pointutils.grouping_operation(feats_cost, idx) #[B,C,N,S]
    aug_feats_cost = feats_cost_grouped.transpose(1,2).contiguous().view(B*N, 1, 1, S).repeat(1,1,S,1) #[BN,C,S,S]
    # aug_feats_cost = torch.tiiru(aug_feats_cost,1).transpose(3,2) + torch.triu(aug_feats_cost)
    # k-NN total distances as cost
    aug_xyz = xyz_grouped.transpose(1,2).contiguous().view(B*N, 3, S, 1).repeat(1, 1, 1, S) #[BN,C,S,S]
    aug_xyz = aug_xyz - aug_xyz.transpose(2,3).contiguous() 
    aug_dists = torch.abs(torch.sqrt(torch.sum(aug_xyz ** 2, 1, keepdim=True)))#[BN,1,S,S]
    # aug_dists = dists.view(B, N, S, 1).repeat(1, 1, 1, S)
    # aug_dists = torch.triu(aug_dists,1).transpose(3,2) + torch.triu(aug_dists)
    point_cost = aug_dists + aug_feats_cost
    point_cost = torch.sum(point_cost, -1, keepdim=False)#[B,N,S]

    return point_cost


def ComputePointCost(sparse_xyz, xyz, feats, nsample=4, radius=1.0, knn=True):
    """
    Input:
        xyz: input points position data, [B, C, N]
        feats: input points data, [B, D, N]
    Return:
        new_xyz: sampled points position data, [B, S, C]
        new_feats: sample points feature data, [B, S, D']
    """
    device = xyz.device
    B, C, N = xyz.shape
    sparse_xyz_t = sparse_xyz.permute(0, 2, 1).contiguous()
    xyz_t = xyz.permute(0, 2, 1).contiguous()

    if knn:
        dists, idx = pointutils.knn(nsample, xyz_t, xyz_t)
        cnt = torch.ones([B, N]) * nsample
    else:
        idx, cnt, dists, mask = query_ball_point(radius, nsample, xyz_t, xyz_t)

    # xyz_grouped = pointutils.grouping_operation(xyz, idx)
    # xyz_diff = xyz_grouped - xyz.view(B, -1, N, 1)  # [B,3,N1,S]

    new_feats = (pointutils.grouping_operation(feats, idx[:,:,1].unsqueeze(-1).contiguous())).squeeze(-1)

    # 1-NN feature similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    feats_cost = cos(feats, new_feats)
    feats_cost = torch.exp(feats_cost).unsqueeze(1)
    feats_cost_grouped = pointutils.grouping_operation(feats_cost, idx) #[B,C,N,S]
    aug_feats_cost = feats_cost_grouped.transpose(1,2).contiguous().view(B*N, 1, 1, nsample).repeat(1,1,nsample,1) #[BN,C,S,S]
    # aug_feats_cost = torch.triu(aug_feats_cost,1).transpose(3,2) + torch.triu(aug_feats_cost)
    # k-NN total distances as cost
    # aug_xyz = xyz_grouped.transpose(1,2).contiguous().view(B*N, C, nsample, 1).repeat(1,1,1,nsample) #[BN,C,S,S]
    # aug_xyz = aug_xyz - aug_xyz.transpose(2,3).contiguous() 
    # aug_xyz_norm = torch.abs(torch.sqrt(torch.sum(aug_xyz ** 2, 1, keepdim=True)))#[BN,1,S,S]
    aug_dists = dists.view(B*N, 1, nsample, 1).repeat(1,1,1,nsample)
    aug_dists = torch.triu(aug_dists,1).transpose(3,2) + torch.triu(aug_dists)
    point_cost = aug_dists + aug_feats_cost
    # mask = torch.ones([B, N, nsample])
    if not knn:
        point_cost = torch.mul(mask, point_cost)
    point_cost = torch.sum(point_cost, dim=-1, keepdim=False).squeeze().view(B, N, -1)#[B,N,S]

    return point_cost, cnt, idx

def HarmonicSeries(n):
    value = 0.0
    n = np.int(n.numpy())
    for item in range(n):
        value = value + 1.0 / (item+1)

    return value

def SetCover(pc1, feats1):
    """
    Input:
        xyz: input points position data, [B, C, N]
        points: input points data, [B, D, N]
    Return:
        new_xyz: sampled points position data, [B, S, C]
        new_points_concat: sample points feature data, [B, S, D']
    """

    pc1_set, feats1_set = DownsamplePoints(pc1, feats1)
    pc1_set_cost, cnt, idx = ComputePointCost(pc1_set, pc1, feats1)

    B, N, K = pc1_set_cost.shape

    bacth_max_cnt, _ = torch.max(cnt, dim=-1, keepdim=True)
    total_cost = torch.sum(pc1_set_cost.view(B, N*K), dim=-1, keepdim=True)
    optimal_cost = torch.zeros([B,1])
    for i in range(B):
        optimal_cost[i, 0] = total_cost[i,0] / HarmonicSeries(bacth_max_cnt[i,0])

    return pc1_set_cost, cnt, optimal_cost, idx


def PositionSelector(pc1, pc2, feats1, feats2, v_pref, radian=np.pi/6, v_samples=5, radian_samples=8):
    device = pc1.device
    B, C, N = pc1.shape
    pc1_set_cost, cnt, optimal_cost, idx = SetCover(pc1, feats1)
    np_pc2 = pc2.transpose(1,2).cpu().numpy()
    _, _, K = pc1_set_cost.shape

    speeds = [(np.exp((i + 1) / v_samples) - 1) / (np.e - 1) * v_pref for i in range(v_samples)]
    # speeds = [0，v_pref/(self.speed_samples-1), v_pref/(self.speed_samples-2)]
    # speeds = [0.3, 0.6, 1.0]
    rotations = np.linspace(-1.0 * radian, radian, radian_samples)
    # action_space = [[0, 0, 0, 0]]
    action_space = []

    # action_space += [ActionXY(0, 1) if holonomic else ActionRot(1, 0)]
    for rotation, speed in itertools.product(rotations, speeds):
        action_space.append([speed * np.cos(rotation), speed * np.sin(rotation), 0, rotation])
       
    best_actions = np.zeros([B,4]) #[vx, vy, 0, rotation]
    new_optimal_cost = torch.zeros([B,1])
    min_diff_optimal_cost = 1e10 * torch.ones([B,1])
    for i,item in enumerate(action_space):
        # Refer to the maximum speed of autonomous vehicel in the city is 50km/h,
        #  then v_pref = 1.5m/s can be as the maximum motion distance
        # According to the Dijestra method, to make the vechicle move to the destination,
        #  so the motion cost is given by the following: sld = 1.5 - v
        sld_cost = 1.5 - np.sqrt(item[0]**2 + item[1]**2)

        np_mask = np.zeros([B, N])
        for i in range(B):
            pt_map = HPR(np_pc2[i, :, :], item[:3], 3.0)
            np_mask[i, pt_map] = 1
        mask = torch.tensor(np_mask, dtype=torch.float32).unsqueeze(1).to(device)
        new_mask = pointutils.grouping_operation(mask, idx).squeeze(1) #[B,C,N,S]
        new_idx = torch.mul(new_mask, idx).contiguous()
        new_pc1_set_cost = torch.mul(new_mask, pc1_set_cost)
        new_cnt = new_mask.sum(-1).cpu()

        # _, _, K = cnt.shape

        new_bacth_max_cnt, _ = torch.max(new_cnt.view(B, N), dim=-1, keepdim=True)
        new_pc1_set_cost = ComputeNewPointCost(pc2, feats2, new_idx)
        new_total_cost = torch.sum(new_pc1_set_cost.view(B, N*K), dim=-1, keepdim=True)
        
        for i in range(B):
            new_optimal_cost[i, 0] = new_total_cost[i,0] / HarmonicSeries(new_bacth_max_cnt[i,0])
            value = torch.abs(new_optimal_cost[i, 0] - optimal_cost[i, 0])
            value = value.detach().cpu().numpy() + sld_cost
            if min_diff_optimal_cost[i, 0] >= value:
                best_actions[i, :] = item
                min_diff_optimal_cost[i, 0] == value

    return best_actions


if __name__ == "__main__":
    pass

        
        




