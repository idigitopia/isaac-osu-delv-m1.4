import torch

# TODO: Careful about these definitions, since we do * imports in the __init__
# fmt: off
GO2_OBS_MIRROR_INDICES = {
    "policy": [
        -0.01, 1, -2,  # base angular velocity
        3, -4, 5,  # projected gravity
        6, -7, -8,  # velocity cmd
        -12, 13, 14, -9, 10, 11, -18, 19, 20, -15, 16, 17,  # joint positions
        -24, 25, 26, -21, 22, 23, -30, 31, 32, -27, 28, 29,  # joint velocities
        -36, 37, 38, -33, 34, 35, -42, 43, 44, -39, 40, 41  # joint actions
    ],
    "critic": [
        0.01, -1, 2,  # base linear velocity
        -3, 4, -5,  # base angular velocity
        6, -7, 8,  # projected gravity
        9, -10, -11,  # velocity cmd
        -15, 16, 17, -12, 13, 14, -21, 22, 23, -18, 19, 20,  # joint positions
        -27, 28, 29, -24, 25, 26, -33, 34, 35, -30, 31, 32,  # joint velocities
        -39, 40, 41, -36, 37, 38, -45, 46, 47, -42, 43, 44  # joint actions
        ]
}
GO2_ACTION_MIRROR_INDICES = [-3, 4, 5, -0.01, 1, 2, -9, 10, 11, -6, 7, 8]
# fmt: on


def mirror_tensor(tensor: torch.Tensor, indices: torch.Tensor):
    """
    Mirrors a tensor according to the information provided in 'indices.' The 'indices' list will
    specify which indices of the tensor need to be moved and/or negated. For example, if the
    mirror vector is [0.1, 2, -1], the mirror state will be will keep the first (0th) element in
    place, then swap the 2nd and 3rd elements while negating the 2nd element. So the state [1, 2, 3]
    would become [1, 3, -2].

    Note that due to how this implemented (need to get sign as well), the zeroth element is index as
    0.1, since 0 and -0 are the same. So this way allows one to specify either 0.1 or -0.1, if you
    need to negate the first element as well.

    Args:
        tensor (torch.Tensor): tensor to be mirrored. Shape (..., N)
        indices (torch.Tensor): tensor to use as mirror indices. Shape (N,)
    """
    sign = torch.sign(indices).to(tensor.device)
    indices = indices.long().abs().to(tensor.device)
    return sign * torch.index_select(tensor, -1, indices)


# TODO: Can probably make this more generic by actually using the env and looking at the observation/action
# manager, get the function names/order, and flip accordingly. For now, just hardcode for set Go2 task.
def data_augmentation_func_go2(obs, actions, env, obs_type="policy"):
    if obs is None:
        return_obs = None
    else:
        if obs_type == "critic":
            return_obs = torch.cat((obs, mirror_tensor(obs, torch.tensor(GO2_OBS_MIRROR_INDICES["critic"]))), dim=0)
        elif obs_type == "policy":
            return_obs = torch.cat((obs, mirror_tensor(obs, torch.tensor(GO2_OBS_MIRROR_INDICES["policy"]))), dim=0)
        else:
            raise ValueError(f"Unknown observation type: {obs_type}. Expected 'policy' or 'critic'.")

    if actions is None:
        return_actions = None
    else:
        return_actions = torch.cat((actions, mirror_tensor(actions, torch.tensor(GO2_ACTION_MIRROR_INDICES))), dim=0)

    return return_obs, return_actions
