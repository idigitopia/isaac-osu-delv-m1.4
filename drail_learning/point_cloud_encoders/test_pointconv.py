import time
import torch
import torch.nn as nn
import numpy as np
from point_cloud_encoders.pointconv.pointconv import PointConv, BottleneckStridedPointConv
from pointnet2_ops.pointnet2_utils import gather_operation, furthest_point_sample

from einops import rearrange


class PointConvEncoder(nn.Module):
    def __init__(self, base_feature_dim=16, in_channel_dim=3, factor=2, pts=1024):
        super(PointConvEncoder, self).__init__()

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
        self.pc1 = PointConv(
            npoint=pts,
            nsample=16,
            point_dim=3,
            in_channel=base_feature_dim,
            out_channel=base_feature_dim * factor,
        ).cuda()
        self.pc2 = BottleneckStridedPointConv(
            nin=pts,
            nout=pts // factor,
            nsample=16,
            point_dim=3,
            in_channel=base_feature_dim * factor,
            out_channel=base_feature_dim * factor * 4,
            bottleneck=4,
        ).cuda()
        self.pc2 = BottleneckStridedPointConv(
            nin=pts,
            nout=pts // factor,
            nsample=16,
            point_dim=3,
            in_channel=base_feature_dim * factor,
            out_channel=base_feature_dim * factor * 4,
            bottleneck=4,
        ).cuda()

        self.pc_layers = [self.pc1, self.pc2]
        self.outdim = base_feature_dim * factor * 8

    def forward(self, x_sets):
        for xs in x_sets:
            assert len(xs.shape) == 3, print(
                f"Shape of input expected to be (b, d, n) but got {xs.shape}"
            )
        pts = [xs.shape[-1] for xs in x_sets]
        assert pts == sorted(pts, reverse=True), print(
            f"The pts were of the order: {pts}, but wanted : {sorted(pts, reverse=True)}"
        )

        feature_set = []
        # point wise features
        pw_features = self.pw_mlp(rearrange(x_sets[0], "b xyz n -> b n xyz"))
        features = rearrange(pw_features, "b n bfd -> b bfd n").contiguous()

        # point conv layers
        r = len(x_sets) - 1 if self._pts == 1024 else len(x_sets) - 2
        for idx in range(r):
            print(idx)
            features = self.pc_layers[idx](xyz=x_sets[idx], points=features, new_xyz=x_sets[idx + 1])
            feature_set.append(features)
            print(f"Feature shape after pc{idx + 1}: {features.shape}")
        return feature_set[-1]



def main():
    B = 4096
    NUM_PTS = 64 
    DOWNSAMPLE_FACTOR = 4
    
    input_point = torch.rand((B, 3, NUM_PTS)).cuda()

    points = [input_point]

    # downsample input point cloud
    for i in range(2):
        factor = DOWNSAMPLE_FACTOR ** (i + 1)
        fps_idx = furthest_point_sample(rearrange(points[-1], "b d n -> b n d").contiguous(), points[-1].shape[-1] // factor)
        downsampled_x = gather_operation(points[-1], fps_idx)
        print(f"Downsampled point cloud shape: {downsampled_x.shape}")
        points.append(downsampled_x)


    model = PointConvEncoder(factor=DOWNSAMPLE_FACTOR).cuda()

    # get number of trainable parameters in model
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(f"Number of Trainable Parameters: {sum([np.prod(p.size()) for p in parameters])}")

    # forward pass
    features = model(points)
    print(features.shape)

if __name__ == "__main__":
    main()