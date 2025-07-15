import os
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from point_cloud_encoders.pointconv.utils import gather_operation, grouping_operation

fast_operations = True

if fast_operations:
    # This uses pykeops and cpp to speed up knn and fps operations.
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as furthest_point_sample, gather_operation
    from pykeops.torch import LazyTensor

    def knn(pts1, pts2, nsample, sorted=False):
        """
        Input:
            pts1: query point set, [B, S, C]
            pts2: other point set, [B, N, C]
            nsample: number of nearest neighbors to sample
            sorted: whether to sort by nearest to farthest
        Return:
            nn_dists: nearest neigbor distances, [B, S, nsample]
            nn_idx: grouped points index, [B, S, nsample]
        """
        B, S, C = pts1.shape
        _, N, _ = pts2.shape

        x_i = LazyTensor(pts1.view(B, S, 1, C))
        y_j = LazyTensor(pts2.view(B, 1, N, C))

        D_ij = ((x_i - y_j)**2).sum(-1)**0.5
        #distances_i = D_ij.Kmin(nsample, dim=2)
        distances_i, indices_i = D_ij.Kmin_argKmin(nsample, dim=2)

        return distances_i, indices_i.int()
else:
    # This is the slow implementation of knn without using pykeops and cpp. Enable this block for onnx export.
    def knn(pts1, pts2, nsample, sorted=False):
        """
        Input:
            pts1: query point set, [B, S, C]
            pts2: other point set, [B, N, C]
            nsample: number of nearest neighbors to sample
            sorted: whether to sort by nearest to farthest
        Return:
            nn_dists: nearest neighbor distances, [B, S, nsample]
            nn_idx: grouped points index, [B, S, nsample]
        """
        B, S, C = pts1.shape
        _, N, _ = pts2.shape

        # Compute pairwise distances
        pts1_expanded = pts1.unsqueeze(2).expand(-1, -1, N, -1)
        pts2_expanded = pts2.unsqueeze(1).expand(-1, S, -1, -1)
        dists = torch.norm(pts1_expanded - pts2_expanded, dim=-1)

        # Find the k smallest distances and their indices
        distances_i, indices_i = torch.topk(-dists, nsample, dim=2)
        distances_i = -distances_i  # Negate to get the smallest distances

        return distances_i, indices_i.int()

    def furthest_point_sample(xyz, npoint):
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance.

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        B, N, _ = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
        distance = torch.ones(B, N, device=xyz.device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
        batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids

    def gather_operation(points, idx):
        """
        Gathers points from the input tensor based on the provided indices.

        Parameters
        ----------
        points : torch.Tensor
            (B, C, N) tensor where B is the batch size, C is the number of channels, and N is the number of points.
        idx : torch.Tensor
            (B, npoint) tensor containing the indices of points to gather.

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor containing the gathered points.
        """
        B, C, N = points.shape
        _, npoint = idx.shape

        # Expand idx to match the dimensions of points
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)

        # Gather points based on idx
        gathered_points = torch.gather(points, 2, idx_expanded)

        return gathered_points


def downsample_point_cloud(point_cloud, factor):
    fps_idx = furthest_point_sample(
        rearrange(point_cloud, "b d n -> b n d").contiguous(),
        point_cloud.shape[-1] // factor,
    )
    return gather_operation(point_cloud, fps_idx)


def count_parameters(model, logger):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                param_count = name + ': ' + 'x'.join(str(x) for x in list(param.size())) + ' = ' + str(num_param)
                logger.info(param_count)
            else:
                logger.info('%s: %d'%(name, num_param))
            total_param += num_param

    logger.info('Number of trainable parameters: %d'%total_param)


def save(state, logger, checkpoint_dir, model_name, epoch, update_best=False, metrics=None):
    save_path = os.path.join(checkpoint_dir, '%s_ckpt_epoch_%04d.pth'%(model_name, epoch))
    logger.info('Saving model weights in %s'%save_path)
    torch.save(state, save_path)
    save_path = os.path.join(checkpoint_dir, '%s_ckpt_last.pth'%model_name)
    torch.save(state, save_path)

    if update_best:
        save_path = os.path.join(checkpoint_dir, '%s_ckpt_best_%f_%f_%f.pth'%(model_name, metrics[0], metrics[1], metrics[2]))
        logger.info('Saving model weights in %s'%save_path)
        torch.save(state, save_path)

    # only save at most two weights files to prevent taking up a ton of storage
    weights_files = [f for f in os.listdir(checkpoint_dir) if (f.endswith('.pth')
                        and 'best' not in f and 'last' not in f and model_name in f)]
    if len(weights_files) > 2:
        num_epoch = min([int(weights_file[-8:-4]) for weights_file in weights_files])
        os.remove(os.path.join(checkpoint_dir, '%s_ckpt_epoch_%04d.pth'%(model_name, num_epoch)))


def create_optimizer(parameters, optimization_config):
    if optimization_config.type == 'Adam':
        optimizer = torch.optim.Adam(parameters, **optimization_config.kwargs)
    if optimization_config.type == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, **optimization_config.kwargs)
    elif optimization_config.type == 'SGD':
        optimizer = torch.optim.SGD(parameters, **optimization_config.kwargs)

    return optimizer


def build_lambda_sche(opti, config):
    if config.get('decay_step') is not None:
        lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd)
    else:
        raise NotImplementedError()

    return scheduler


def create_scheduler(optimizer, scheduler_config):
    if scheduler_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_config.kwargs)
    elif scheduler_config.type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config.kwargs)
    elif scheduler_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, scheduler_config.kwargs)

    return scheduler


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 1e10
    #farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = torch.zeros(B, dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def masked_dist(pts1, valid1, pts2, valid2, compute_mode=None):
    device = pts1.device
    B, S, _ = pts1.shape
    _, N, _ = pts2.shape

    if compute_mode is None:
        sqrdists = torch.cdist(pts1, pts2, compute_mode='donot_use_mm_for_euclid_dist')  # [B, S, N]
    elif compute_mode == 'mm':
        sqrdists = torch.cdist(pts1, pts2)

    # set distance to invalid points to infinity to avoid selecting them as a nearest neighbor
    valid_nn = valid2.view(B, 1, N).expand(-1, S, -1)
    sqrdists = valid_nn * sqrdists + (1 - valid_nn) * 1e10

    return sqrdists





def old_masked_knn(pts1, valid1, pts2, valid2, nsample=8):
    """
    Input:
        pts1: query point set, [B, S, C]
        valid1: valid points, [B, S]
        pts2: other point set, [B, N, C]
        valid2: valid points, [B, N]
        nsample: number of nearest neighbors to sample
        sorted: whether to sort by nearest to farthest
    Return:
        nn_dists: nearest neigbor distances, [B, S, nsample]
        nn_idx: grouped points index, [B, S, nsample]
    """
    B, _, S = pts1.shape
    _, _, N = pts2.shape

    with torch.no_grad():
        pts1 = pts1.permute(0, 2, 1)
        pts2 = pts2.permute(0, 2, 1)

        sqrdists = masked_dist(pts1, valid1, pts2, valid2)

        nn_dists, nn_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)

        nn_idx = nn_idx * valid1.view(B, S, 1)
        nn_dists = nn_dists * valid1.view(B, S, 1)

    return nn_dists, nn_idx.to(torch.int32)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = -1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).expand(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def masked_knn(pts1, valid1, pts2, valid2, knn):
    """
    Input:
        pts1: query point set, [B, S, C]
        valid1: valid points, [B, S]
        pts2: other point set, [B, N, C]
        valid2: valid points, [B, N]
        nsample: number of nearest neighbors to sample
        sorted: whether to sort by nearest to farthest
    Return:
        nn_dists: nearest neigbor distances, [B, S, nsample]
        nn_idx: grouped points index, [B, S, nsample]
    """
    with torch.no_grad():
        nn_dists, nn_idx = knn(pts1, valid1, pts2, valid2)
        nn_dists = nn_dists.permute(0, 2, 1).contiguous()
        nn_idx = nn_idx.permute(0, 2, 1). contiguous()

    return nn_dists, nn_idx


def masked_gather(points, idx, valid):
    new_points = gather_operation(points, idx)
    new_points = new_points * valid.unsqueeze(dim=1)

    return new_points


def masked_group(points, idx, valid):
    B, N = valid.shape

    new_points = grouping_operation(points, idx)
    new_points = new_points * valid.view(B, 1, N, 1)

    return new_points


def combine(partial, sampled, num_partial, num_sampled):
    combined = [torch.cat([partial[b:b+1,:,:n_p], sampled[b:b+1,:,:n_s]], dim=-1)
                            for b, (n_p, n_s) in enumerate(zip(num_partial, num_sampled))]
    combined = torch.cat(combined, dim=0)

    return combined


class MaskedLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=False,
                 device=None, dtype=None):
        super(MaskedLayerNorm, self).__init__(normalized_shape, eps,
                                              elementwise_affine)

    def forward(self, input, valid):
        '''
        input: [N, *, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]
        valid: [N, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]
        '''
        # make sure input dims match normalized shape dims...
        assert len(input.shape) > len(self.normalized_shape)
        #assert list(input.shape[-len(self.normalized_shape):]) == list(self.normalized_shape)

        device = input.device
        N = input.shape[0]
        dims = tuple(range(-len(self.normalized_shape), 0))
        extra_dims = len(input.shape) - len(valid.shape)
        valid = valid.float()

        # create probability mask that gives equal probability to valid
        # points and 0 probability to invalid points
        num_valid_points = torch.sum(valid, dim=dims, keepdims=True)
        #probs = valid / num_valid_points
        #probs = probs.view(N, *(1,)*extra_dims, *self.normalized_shape).expand_as(input)
        mask = valid.view(N, *(1,)*extra_dims, *self.normalized_shape).expand_as(input)

        # compute mean over last D dims
        mean = torch.sum(input * mask, dim=dims, keepdims=True) / num_valid_points

        # compute variance over last D dims
        #var = torch.sum((input**2) * probs, dim=dims, keepdims=True) - (mean**2)
        var = torch.sum(((input - mean) * mask)**2, dim=dims, keepdims=True) / num_valid_points

        # normalize input
        output = (input - mean) / torch.sqrt(var + self.eps)

        # apply learned affine transformation
        if self.elementwise_affine:
            weight = self.weight.view(*(1,)*(extra_dims+1), *self.normalized_shape).to(device)
            bias = self.bias.view(*(1,)*(extra_dims+1), *self.normalized_shape).to(device)
            output = output * weight + bias

        # zero out masked regions that may have been shifted by bias
        output = output * mask

        return output
