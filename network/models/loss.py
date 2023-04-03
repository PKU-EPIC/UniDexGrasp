import torch


def min_distance_from_m_to_n(m, n):
    """
    :param m: [..., M, 3]
    :param n: [..., N, 3]
    :return: [..., M]
    """
    m_num = m.shape[-2]
    n_num = n.shape[-2]

    # m_: [..., M, N, 3]
    # n_: [..., M, N, 3]
    m_ = m.unsqueeze(-2)  # [..., M, 1, 3]
    n_ = n.unsqueeze(-3)  # [..., 1, N, 3]

    m_ = torch.repeat_interleave(m_, n_num, dim=-2)  # [..., M, N, 3]
    n_ = torch.repeat_interleave(n_, m_num, dim=-3)  # [..., M, N, 3]

    # [..., M, N]
    pairwise_dis = torch.sqrt(((m_ - n_) ** 2).sum(dim=-1))

    ret_dis = torch.min(pairwise_dis, dim=-1)[0]  # [..., M]

    return ret_dis


def soft_distance(distance):
    """
    :param distance: [..., M]
    :return: [..., M]
    """
    sigmoid = torch.nn.Sigmoid()
    normalize_factor = 60  # decided by visualization
    return 1 - 2 * (sigmoid(normalize_factor * distance) - 0.5)


def contact_map_of_m_to_n(m, n):
    """
    :param m: [..., M, 3]
    :param n: [..., N, 3]
    :return: [..., M]
    """
    distances = min_distance_from_m_to_n(m, n)  # [..., M]
    distances = soft_distance(distances)
    return distances

def discretize_gt_cm(contact_map, num_bins=10):
    """
    :param contact_map: [B, N], with values within [0, 1]
    :return: [B, N, num_bins]
    """
    bin_boundaries = [i * (1 / num_bins) for i in range(num_bins + 1)]
    bins = []
    contact_map = contact_map.unsqueeze(-1)  # [B, N, 1]
    for i in range(num_bins):
        bins.append(torch.logical_and(contact_map >= bin_boundaries[i],
                                      contact_map < bin_boundaries[i + 1]))
    one_hot = torch.cat(bins, dim=-1).float()  # [B, N, num_bins]
    return one_hot