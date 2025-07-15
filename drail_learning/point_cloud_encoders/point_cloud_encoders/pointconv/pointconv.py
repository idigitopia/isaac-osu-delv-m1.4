"""
Code taken from PointConv [Wu et al. 2019]:
https://github.com/DylanWusee/pointconv_pytorch/blob/master/utils/pointconv_util.py

Modified by Wesley Khademi
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from point_cloud_encoders.pointconv.utils import grouping_operation, torch_utils


def group(xyz, points, new_xyz, nn_idx=None):
    """
    Input:
        xyz: input points position data, [B, S, C]
        points: input points data, [B, S, D]
        valid_xyz: tensor indicating valid 'xyz' points (non-filler points)
        new_xyz: query points position data, [B, N, C]
        nsample: number of nearest neighbors to sample
        valid_new_xyz: tensor indicating valid 'new_xyz' points (non-filler points)
    Return:
        grouped_xyz_norm: relative point positions [B, 3, nsample, N]
        new_points: sampled points data, [B,  C+D, nsample, N]
        valid_knn: valid nearest neighbors, [B, nsample, N]
    """
    device = xyz.device
    B, C, N = new_xyz.shape
    _, D, _ = points.shape

    if nn_idx is None:
        # _, idx = knn(xyz, new_xyz)  # [B, npoint, nsample]
        # idx = idx.permute(0, 2, 1).contiguous()
        xyz_flipped = xyz.permute(0, 2, 1).contiguous()
        new_xyz_flipped = new_xyz.permute(0, 2, 1).contiguous()
        _, idx = torch_utils.knn(new_xyz_flipped, xyz_flipped, nsample=16)
    else:
        idx = nn_idx

    grouped_xyz = grouping_operation(xyz, idx)  # [B, C, npoint, nsample]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, C, N, 1)

    if points is not None:
        new_points = grouping_operation(points, idx)
    else:
        new_points = grouped_xyz_norm

    new_points = new_points.permute(0, 1, 3, 2)
    grouped_xyz_norm = grouped_xyz_norm.permute(0, 1, 3, 2)

    return new_points, grouped_xyz_norm


class WeightNet(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Linear(in_channel, out_channel))
            self.mlp_bns.append(nn.LayerNorm(out_channel))
        else:
            self.mlp_convs.append(nn.Linear(in_channel, hidden_unit[0]))
            self.mlp_bns.append(nn.LayerNorm(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Linear(hidden_unit[i - 1], hidden_unit[i]))
                self.mlp_bns.append(nn.LayerNorm(hidden_unit[i]))
            self.mlp_convs.append(nn.Linear(hidden_unit[i], out_channel))
            self.mlp_bns.append(nn.LayerNorm(out_channel))

    def forward(self, localized_xyz):
        weights = localized_xyz.permute(0, 2, 3, 1)
        for i, conv in enumerate(self.mlp_convs):
            weights = conv(weights)
            bn = self.mlp_bns[i]
            weights = F.leaky_relu(bn(weights), 0.2)
        weights = weights.permute(0, 3, 1, 2)

        return weights


class PointConv(nn.Module):
    def __init__(
        self,
        npoint,
        nsample,
        point_dim,
        in_channel,
        out_channel,
        mlp=[],
        c_mid=16,
        final_norm="batch",
        final_act="relu",
    ):
        super(PointConv, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.final_norm = final_norm
        self.final_act = final_act

        last_ch = point_dim + in_channel

        self.weightnet = WeightNet(npoint, nsample, point_dim, c_mid)
        self.linear = nn.Linear(c_mid * last_ch, out_channel)
        if self.final_norm == "batch":
            self.norm_linear = nn.LayerNorm(out_channel)

    def forward(self, xyz, points, new_xyz, nn_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
            valid_xyz: indicates valid xyz points [B, N]
            new_xyz: sampled points position data [B, C, S]
            valid_new_xyz: indicates valid new_xyz poins [B, S]
        Return:
            new_points: sample points feature data, [B, D', S]
        """
        B, _, N = xyz.shape
        _, _, S = new_xyz.shape

        new_points, grouped_xyz_norm = group(xyz, points, new_xyz, nn_idx)
        new_points = torch.cat([new_points, grouped_xyz_norm], dim=1)

        weights = self.weightnet(grouped_xyz_norm)
        new_points = torch.matmul(
            input=new_points.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)
        ).view(B, S, -1)

        new_points = self.linear(new_points)

        if self.final_norm is not None:
            new_points = self.norm_linear(new_points)

        if self.final_act == "relu":
            new_points = F.leaky_relu(new_points, 0.2)

        new_points = new_points.permute(0, 2, 1).contiguous()

        return new_points


class BottleneckStridedPointConv(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        nsample,
        point_dim,
        in_channel,
        out_channel,
        mlp=[],
        bottleneck=4,
        c_mid=16,
    ):
        super(BottleneckStridedPointConv, self).__init__()

        self.reduce = nn.Sequential(
            nn.Linear(in_channel, in_channel // bottleneck),
            nn.LayerNorm(in_channel // bottleneck),
            nn.LeakyReLU(0.2),
        )

        self.pointconv = PointConv(
            npoint=nout,
            nsample=nsample,
            point_dim=point_dim,
            in_channel=in_channel // bottleneck,
            out_channel=out_channel // bottleneck,
            c_mid=c_mid,
        )

        self.expand = nn.Sequential(
            nn.Linear(out_channel // bottleneck, out_channel), nn.LayerNorm(out_channel)
        )

        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, xyz, points, new_xyz, nn_idx=None):
        if nn_idx is None:
            xyz_flipped = xyz.permute(0, 2, 1).contiguous()
            new_xyz_flipped = new_xyz.permute(0, 2, 1).contiguous()
            _, nn_idx = torch_utils.knn(new_xyz_flipped, xyz_flipped, nsample=16)

        reduced_points = points.permute(0, 2, 1)
        reduced_points = self.reduce(reduced_points)
        reduced_points = reduced_points.permute(0, 2, 1).contiguous()

        new_points = self.pointconv(xyz, reduced_points, new_xyz, nn_idx=nn_idx)

        new_points = new_points.permute(0, 2, 1)
        new_points = self.expand(new_points)
        new_points = new_points.permute(0, 2, 1)

        group_points = grouping_operation(points, nn_idx)
        points = torch.mean(group_points.permute(0, 1, 3, 2), dim=2)
        shortcut = self.residual_layer(points)

        new_points = F.leaky_relu(new_points + shortcut, 0.2)

        return new_points.contiguous()
