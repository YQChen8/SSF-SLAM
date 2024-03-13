import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum
from time import time
import numpy as np
from lib import pointnet2_utils as pointutils


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


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


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_point(k, pos1, pos2):
    '''
    Input:
        k: int32, number of k in k-nn search
        pos1: (batch_size, ndataset, c) float32 array, input points
        pos2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    B, N, C = pos1.shape
    M = pos2.shape[1]
    pos1 = pos1.view(B, 1, N, -1).repeat(1, M, 1, 1)
    pos2 = pos2.view(B, M, 1, -1).repeat(1, 1, N, 1)
    dist = torch.sum(-(pos1 - pos2) ** 2, -1)
    val, idx = dist.topk(k=k, dim=-1)
    return torch.sqrt(-val), idx


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
    group_idx[sqrdists > radius ** 2] = nsample
    mask = group_idx != nsample
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == nsample
    cnt = mask.sum(dim=-1)
    group_idx[mask] = group_first[mask]
    return group_idx.contiguous(), cnt


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx, _ = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, use_instance_norm=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # TODO：
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            if use_instance_norm:
                self.mlp_bns.append(nn.InstanceNorm2d(out_channel, affine=True))
            else:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # if group_all:
        #     self.queryandgroup = pointutils.GroupAll()
        # else:
        #     self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

    def forward(self, xyz, points, fps_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        # if points is not None:
        #     points = points.permute(0, 2, 1).contiguous()

        # 选取邻域点
        # if self.group_all == False:
        if fps_idx is None:
            fps_idx = pointutils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
        
        new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, N]
        _, knn_idx = pointutils.knn(self.nsample, new_xyz.permute(0, 2, 1).contiguous(), xyz_t)
        # new_points = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points)  # [B, 3+C, N, S]
        xyz_grouped = pointutils.grouping_operation(xyz, knn_idx)
        pos_diff = xyz_grouped - new_xyz.unsqueeze(-1) # [B,3,N1,S]
        points_grouped = pointutils.grouping_operation(points, knn_idx)
        new_points = torch.cat([pos_diff, points_grouped], dim=1)
        # else:
        #     new_xyz = xyz
        #     fps_idx = None
        #     new_points = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points)  # [B, 3+C, N, S]
        #     new_points = new_points.permute(0,1,3,2).contiguous()

        # new_xyz: sampled points position data, [B, C, npoint]
        # new_points: sampled points data, [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0]
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

    def forward(self, pos1, pos2, feature1, feature2, sf1=None, sf2=None):
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

        if sf1 is None and sf2 is None:
            pos2_grouped = pointutils.grouping_operation(pos2.float(), idx)
            pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)  # [B,3,N1,S]
        else:
            pos2_grouped = pointutils.grouping_operation(sf2.float(), idx)
            pos_diff = pos2_grouped - sf1.view(B, -1, N, 1)  # [B,3,N1,S]

        feat2_grouped = pointutils.grouping_operation(feature2.float(), idx)
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


class MaskType(IntEnum):
    FPS = 0
    RANDOM = 1


class FeaturesCouplingConv(nn.Module):
        """ An implementation of a coupling layer
        from RealNVP (https://arxiv.org/abs/1605.08803).
        """
        def __init__(self, num_inputs, num_hidden, mask, num_cond_inputs=None, s_act='tanh', t_act='relu'):
            super(FeaturesCouplingConv, self).__init__()
            self.num_inputs = num_inputs
            self.mask = mask

            activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
            s_act_func = activations[s_act]
            t_act_func = activations[t_act]

            if num_cond_inputs is not None:
                total_inputs = num_inputs + num_cond_inputs
            else:
                total_inputs = num_inputs

            self.scale_net = nn.Sequential(
                nn.Linear(total_inputs, num_hidden), s_act_func(),
                nn.Linear(num_hidden, num_hidden), s_act_func(),
                nn.Linear(num_hidden, num_inputs))
            self.translate_net = nn.Sequential(
                nn.Linear(total_inputs, num_hidden), t_act_func(),
                nn.Linear(num_hidden, num_hidden), t_act_func(),
                nn.Linear(num_hidden, num_inputs))

            def init(m):
                if isinstance(m, nn.Linear):
                    m.bias.data.fill_(0)
                    nn.init.orthogonal_(m.weight.data)

        def forward(self, inputs, cond_inputs=None, mode='direct'):
            mask = self.mask

            masked_inputs = inputs * mask
            if cond_inputs is not None:
                masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

            if mode == 'direct':
                log_s = self.scale_net(masked_inputs) * (1 - mask)
                t = self.translate_net(masked_inputs) * (1 - mask)
                s = torch.exp(log_s)
                return inputs * s + t, log_s.sum(-1, keepdim=True)
            else:
                log_s = self.scale_net(masked_inputs) * (1 - mask)
                t = self.translate_net(masked_inputs) * (1 - mask)
                s = torch.exp(-log_s)
                return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class Mix(nn.Module):
    def __init__(self, nsample, radius, knn=True, use_mix=False):
        super(Mix, self).__init__()
        self.nsample = nsample
        self.radius = radius
        self.knn = knn
        self.use_mix = use_mix

    def forward(self, pos1, pos2, feats1, feats2, factor):
        '''
        :param pos1: [B,3,N]
        :param pos2: [B,3,N]
        :param feats1: [B,C,N]
        :param feats2: [B,C,N]
        :return:
        '''
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, C, N = pos1.shape
        # if factor == self.nsample:
        #     self.use_mix = False

        if self.knn:
            _, idx_intra = pointutils.knn(self.nsample, pos1_t, pos2_t)
            # if self.use_mix:
            _, idx = pointutils.knn(self.nsample, pos1_t, pos1_t)
        else:
            idx_intra, _ = query_ball_point(self.radius, self.nsample, pos2_t, pos1_t)
            idx, _ = query_ball_point(self.radius, self.nsample, pos1_t, pos1_t)

        pos2_grouped = pointutils.grouping_operation(pos2, idx_intra)
        pos1_grouped = pointutils.grouping_operation(pos1, idx)

        feats2_grouped = pointutils.grouping_operation(feats2, idx_intra)
        # feats2_grouped = pointutils.grouping_operation(feats2, idx)
        if self.use_mix:
            feats1_grouped = pointutils.grouping_operation(feats1, idx)

        mix_factor = self.nsample - factor
        new_pos1 = torch.cat([pos1_grouped[ :, :, :, :factor], pos2_grouped[ :, :, :, :mix_factor]], dim=-1)
        if self.use_mix:
            new_feats1 = torch.cat([feats1_grouped[:, :, :, :factor], feats2_grouped[:, :, :, :mix_factor]], dim=-1)
        else:
            new_feats1 = feats2_grouped

        return new_pos1, new_feats1


class PointConvFlow(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, bn=True):
        super(PointConvFlow, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.flow_convs = nn.ModuleList()
        if bn:
            self.flow_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.flow_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.flow_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.corr_convs = nn.ModuleList()
        if bn:
            self.corr_bns = nn.ModuleList()
        for out_channel in mlp:
            self.corr_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.corr_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos2_grouped, feats1, feats2_grouped):
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
        B, _, N, S = pos2_grouped.shape
        _, C, _, _ = feats2_grouped.shape
        xyz1 = pos1.contiguous()
        pos2_grouped = pos2_grouped.contiguous()
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)  # [B, 3, N, S]
        feats_diff = torch.cat([feats2_grouped, feats1.view(B, -1, N, 1).repeat(1, 1, 1, self.nsample)], dim=1)

        feat1_new = torch.cat([pos_diff, feats_diff], dim=1)  # [B, 2*C+3,N,S]
        for i, conv in enumerate(self.flow_convs):
            bn = self.flow_bns[i]
            feat1_new = F.relu(bn(conv(feat1_new)))

        for i, conv in enumerate(self.corr_convs):
            bn = self.corr_bns[i]
            feats_diff = F.relu(bn(conv(feats_diff)))

        feat1_new = torch.max(feat1_new, -1)[0]  # [B, mlp[-1], npoint]
        new_pos1 = torch.mean(pos2_grouped, dim=-1, keepdim=False)
        delta_flow = new_pos1 - pos1
        corr_weight = torch.max(feats_diff, -1)[0]

        return new_pos1, corr_weight, feat1_new, delta_flow


class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()
        self.conv_corr = nn.Conv1d(128, 128, 1)
        self.conv_flow = nn.Conv1d(3, 128, 1)
        self.conv = nn.Conv1d(128+128, 128-3, 1)

    def forward(self, flow, corr):
        cor = F.relu(self.conv_corr(corr))
        flo = F.relu(self.conv_flow(flow))
        cor_flo = torch.cat([cor, flo], dim=1)
        out_conv = F.relu(self.conv(cor_flo))
        out = torch.cat([out_conv, flow], dim=1)
        return out


class ConvGRU(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convr = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convq = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        rh_x = torch.cat([r*h, x], dim=1)
        q = torch.tanh(self.convq(rh_x))

        h = (1 - z) * h + z * q
        return h


class ConvRNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(ConvRNN, self).__init__()
        self.convx = nn.Conv1d(input_dim, hidden_dim, 1)
        self.convh = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, h, x):
        xt = self.convx(x)
        ht = self.convh(h)

        h = torch.tanh(xt + ht)
        return h


class UpdateBlock(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(UpdateBlock, self).__init__()
        self.motion_encoder = MotionEncoder()
        self.gru = ConvGRU(input_dim=input_dim, hidden_dim=hidden_dim)
        self.flow_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 3, kernel_size=1, bias=True)
        )

    def forward(self, net, inp, corr, flow):
        motion_features = self.motion_encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)  # 128d
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, delta_flow

class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1).contiguous() # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1).contiguous() # B S 3
        dists, idx = pointutils.three_nn(xyz, sparse_xyz)
        dists[dists < 1e-10] = 1e-10
        weight = 1.0 / dists
        interpolated_feat = torch.sum(pointutils.grouping_operation(sparse_flow, idx) * weight.view(B, 1, N, 3),
                                      dim=-1)  # [B,C,N,3]

        return interpolated_feat


class FlowEmbedding(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, pooling='max', corr_func='concat', knn=True, use_instance_norm=False):
        super(FlowEmbedding, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.pooling = pooling
        self.corr_func = corr_func
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if corr_func == 'concat':
            last_channel = in_channel * 2 + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            if use_instance_norm:
                self.mlp_bns.append(nn.InstanceNorm2d(out_channel, affine=True))
            else:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
        Input:
            xyz1: (batch_size, 3, npoint)
            xyz2: (batch_size, 3, npoint)
            feat1: (batch_size, channel, npoint)
            feat2: (batch_size, channel, npoint)
        Output:
            xyz1: (batch_size, 3, npoint)
            feat1_new: (batch_size, mlp[-1], npoint)
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, N, C = pos1_t.shape
        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)
        else:
            # If the ball neighborhood points are less than nsample,
            # than use the knn neighborhood points
            idx, cnt = query_ball_point(self.radius, self.nsample, pos2_t, pos1_t)
            # 利用knn取最近的那些点
            _, idx_knn = pointutils.knn(self.nsample, pos1_t, pos2_t)
            cnt = cnt.view(B, -1, 1).repeat(1, 1, self.nsample)
            idx = idx_knn[cnt > (self.nsample - 1)]

        pos2_grouped = pointutils.grouping_operation(pos2, idx)  # [B, 3, N, S]
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)  # [B, 3, N, S]

        feat2_grouped = pointutils.grouping_operation(feature2, idx)  # [B, C, N, S]
        if self.corr_func == 'concat':
            feat_diff = torch.cat([feat2_grouped, feature1.view(B, -1, N, 1).repeat(1, 1, 1, self.nsample)], dim=1)

        feat1_new = torch.cat([pos_diff, feat_diff], dim=1)  # [B, 2*C+3,N,S]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat1_new = F.relu(bn(conv(feat1_new)))

        feat1_new = torch.max(feat1_new, -1)[0]  # [B, mlp[-1], npoint]
        return pos1, feat1_new


class PointNetFeaturePropogation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropogation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, C, N = pos1.shape

        # dists = square_distance(pos1, pos2)
        # dists, idx = dists.sort(dim=-1)
        # dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
        dists, idx = pointutils.three_nn(pos1_t, pos2_t)
        dists[dists < 1e-10] = 1e-10
        weight = 1.0 / dists
        weight = weight / torch.sum(weight, -1, keepdim=True)  # [B,N,3]
        interpolated_feat = torch.sum(pointutils.grouping_operation(feature2, idx) * weight.view(B, 1, N, 3),
                                      dim=-1)  # [B,C,N,3]

        if feature1 is not None:
            feat_new = torch.cat([interpolated_feat, feature1], 1)
        else:
            feat_new = interpolated_feat

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat_new = F.relu(bn(conv(feat_new)))
        return feat_new

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    input = torch.randn(4, 3, 2048).cuda()
    label = torch.randn(4, 16)
    # model = RealNVP().cuda()
    # device_ids = [0]
    # if len(device_ids) == 0:
    #     net = nn.DataParallel(model)
    # else:
    #     net = nn.DataParallel(model, device_ids=device_ids)
    # print("Let's use ", len(device_ids), " GPUs!")