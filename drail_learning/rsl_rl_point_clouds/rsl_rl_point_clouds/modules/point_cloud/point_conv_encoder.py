import torch.nn as nn
from einops import rearrange
from point_cloud_encoders.pointconv.pointconv import (
    BottleneckStridedPointConv,
    PointConv,
)
from point_cloud_encoders.pointconv.utils.torch_utils import downsample_point_cloud


class PointConvEncoder(nn.Module):
    def __init__(self, base_feature_dim=16, in_channel_dim=3, downsample_factor=2, downsample_steps=2, pts=1024):
        super().__init__()

        self.downsample_factor = downsample_factor
        self.downsample_steps = downsample_steps
        self._pts = pts
        # point wise mlp
        self.pw_mlp = nn.Sequential(
            nn.Linear(in_channel_dim, base_feature_dim),
            nn.LayerNorm(base_feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(base_feature_dim, base_feature_dim),
            nn.LayerNorm(base_feature_dim),
            nn.LeakyReLU(0.2),
        )
        # point conv layers

        self.pc_layers = nn.ModuleList(
            [
                PointConv(
                    npoint=pts,
                    nsample=16,
                    point_dim=3,
                    in_channel=base_feature_dim,
                    out_channel=base_feature_dim * self.downsample_factor,
                ),
                BottleneckStridedPointConv(
                    nin=pts,
                    nout=pts // self.downsample_factor,
                    nsample=16,
                    point_dim=3,
                    in_channel=base_feature_dim * self.downsample_factor,
                    out_channel=base_feature_dim * self.downsample_factor * 4,
                    bottleneck=4,
                ),
            ]
        )
        self.outdim = base_feature_dim * self.downsample_factor * 8

    def forward(self, point_cloud):
        # Add more point clouds by downsampling the point cloud
        x_sets = [point_cloud]
        for i in range(self.downsample_steps):
            x_sets.append(downsample_point_cloud(x_sets[-1], self.downsample_factor ** (i + 1)))

        # point wise features
        pw_features = self.pw_mlp(rearrange(x_sets[0], "b xyz n -> b n xyz"))
        features = rearrange(pw_features, "b n bfd -> b bfd n").contiguous()

        # point conv layers
        r = len(x_sets) - 1 if self._pts == 1024 else len(x_sets) - 2
        for idx in range(r):
            features = self.pc_layers[idx](xyz=x_sets[idx], points=features, new_xyz=x_sets[idx + 1])
        return features.mean(-1)
