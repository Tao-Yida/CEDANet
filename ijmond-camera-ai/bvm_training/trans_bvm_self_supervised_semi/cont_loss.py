import torch
import torch.nn.functional as F


def cos_sim(vec_a, vec_b, temp_fac=0.1):
    """
    Compute the cosine similarity between two sets of vectors.
    Args:
        vec_a (torch.Tensor): First set of vectors, shape (N, D) where N is the number of samples and D is the dimension.
        vec_b (torch.Tensor): Second set of vectors, shape (Z, D) where Z is the number of classes and D is the dimension.
        temp_fac (float): Temperature factor to scale the cosine similarity.
    Returns:
        torch.Tensor: Cosine similarity scores, shape (N, Z) where N is the number of samples and Z is the number of classes.
    """
    # Compute the cosine similarity score of the input 2 vectors scaled by temperature factor
    # L2 normalization of vectors
    # vec_a: (N,D), vec_b: (ZxD)
    norm_vec_a = F.normalize(vec_a, dim=1)
    norm_vec_b = F.normalize(vec_b, dim=1)
    # Cosine similarity calculation 100x16
    cos_sim_val = torch.matmul(norm_vec_a, norm_vec_b.transpose(-1, -2)) / temp_fac  # (NxD) (DxZ) -> NxZ
    return cos_sim_val  # NxZ  N-samples Z-num of classes


def class_center_contrastive_loss(target_samples, target_center, other_centers, temp_fac=0.1, epsilon=1e-8):
    """
    计算目标类样本与类中心的对比损失，鼓励样本接近自己的类中心，远离其他类中心。

    Args:
        target_samples (torch.Tensor): 目标类的样本特征，形状为 (D, N)，D为特征维度，N为样本数量。
        target_center (torch.Tensor): 目标类的中心向量，形状为 (D, 1)。
        other_centers (torch.Tensor): 其他类的中心向量，形状为 (D, Z)，Z为其他类的数量。
        temp_fac (float): 温度因子，用于缩放余弦相似度，默认0.1。
        epsilon (float): 小值常数，避免除零错误，默认1e-8。

    Returns:
        torch.Tensor: 对比损失值，标量。
    """
    # 转置所有输入以适应cos_sim函数的输入格式
    target_samples = target_samples.T  # 形状: (D, N) -> (N, D)，例如(100, 16)
    target_center = target_center.T  # 形状: (D, 1) -> (1, D)
    other_centers = other_centers.T  # 形状: (D, Z) -> (Z, D)

    # 计算每个目标样本与目标类中心的相似度
    sim_with_target = cos_sim(target_samples, target_center, temp_fac)  # 形状: (N, 1)

    # 计算每个目标样本与所有其他类中心的相似度总和
    sim_with_others = cos_sim(target_samples, other_centers, temp_fac).sum(dim=-1, keepdim=True)  # 形状: (N, Z) -> (N, 1)，求和后

    # 对相似度值进行截断，确保非负
    sim_with_target = torch.clamp(sim_with_target, min=epsilon)
    sim_with_others = torch.clamp(sim_with_others, min=epsilon)

    # 计算对比损失: -log(正例相似度/(正例相似度+负例相似度))
    contrast_loss = -torch.log(sim_with_target / (sim_with_target + sim_with_others))  # 形状: (N, 1)

    # 返回所有样本的平均损失
    return contrast_loss.mean()


def intra_inter_contrastive_loss(features, masks, num_samples=100, margin=1.0, inter=True):
    """
    计算图内(intra)或图间(inter)对比损失，区分烟雾和背景两个类别。

    参数:
        features (Tensor): 模型输出特征图，形状为 (B, D, H, W)，B为批量大小，D为特征维度，H和W为空间尺寸。
        masks (Tensor): 真值掩码，形状为 (B, H, W)，其中1表示烟雾，0表示背景。
        num_samples (int): 每类采样的像素点数量，默认100。
        margin (float): 对比损失的边界参数，默认1.0。
        inter (bool): 如果为True，计算图间对比损失；如果为False，计算图内对比损失。

    返回:
        loss (Tensor): 计算得到的对比损失值。
    """
    batch_size, feature_dim, height, width = features.size()
    total_loss = 0.0

    for i in range(batch_size):
        # 每次取出一个样本的特征图和对应的掩码
        # features: (B, D, H, W) -> (D, H, W) for each sample
        # masks: (B, H, W) -> (1, H, W) for each sample
        feature_map = features[i]  # D=16 x H=352 x W=352
        mask = masks[i]  # 1 x H=352 x W=352,

        # Separate features into smoke and background based on mask
        smoke_features = feature_map[:, mask.squeeze(0) > 0].view(feature_dim, -1)  # D x N_smoke
        background_features = feature_map[:, mask.squeeze(0) == 0].view(feature_dim, -1)  # D x N_background
        # print(background_features.shape, smoke_features.shape)

        # 对每列（同类像素向量）求均值，得到该图内烟雾/背景的平均特征向量
        # Compute mean feature vectors for smoke and background within the same image
        mean_smoke = smoke_features.mean(dim=1, keepdim=True) if smoke_features.size(1) > 0 else None  # 16 x 1
        mean_background = background_features.mean(dim=1, keepdim=True) if background_features.size(1) > 0 else None  # 16 x 1
        # print(mean_background.shape, mean_smoke.shape)
        if mean_smoke is None or mean_background is None:
            batch_size -= 1  # 修正后续平均分母
            continue

        # Normalize mean smoke and mean background
        # 沿通道维度做L2归一化，使每个均值向量的模长为1
        mean_smoke = F.normalize(mean_smoke, dim=0)
        mean_background = F.normalize(mean_background, dim=0)

        # Sample features from each class within the same image
        # 如果烟雾特征数量大于num_samples，则随机采样num_samples个像素特征
        if smoke_features.size(1) > num_samples:
            smoke_samples = smoke_features[:, torch.randperm(smoke_features.size(1))[:num_samples]]  # 16 x 100
        else:
            smoke_samples = smoke_features

        # 再做一次归一化
        smoke_features = F.normalize(smoke_features, dim=0)
        # print(smoke_samples.shape)

        # if background_features.size(1) > num_samples:
        #     background_samples = background_features[:, torch.randperm(background_features.size(1))[:num_samples]] # 16 x 100
        # else:
        #     background_samples = background_features
        # print(background_samples.shape)

        # Intra-image contrastive loss
        if not inter:
            # Compute positive and negative losses within the same image
            # 1. For each smoke sample compute the loss
            smoke_loss = class_center_contrastive_loss(smoke_samples, mean_smoke, mean_background, temp_fac=0.1)
            # print(smoke_loss)
            # 2. For each background sample compute the loss
            # (Maybe it doesn't make sense to calcualte the loss for the background samples as it's not a class.)
            # background_loss = class_center_contrastive_loss(background_samples, mean_background, mean_smoke, temp_fac=0.1)
            # loss = torch.div(torch.add(smoke_loss, background_loss), 2)
            # print(loss)
            total_loss += smoke_loss

    return total_loss / batch_size


################################################################
# 以下为废弃的老版本实现代码，保留作为参考
# for sample in smoke_samples.T:
#     smoke_loss += -torch.log( cos_sim(sample.unsqueeze(1), mean_smoke)
#                 / (cos_sim(sample.unsqueeze(1), mean_smoke) + cos_sim(sample.unsqueeze(1), mean_background)))
# smoke_loss /= smoke_samples.size(1)

# for sample in target_samples.T:
#     extra_mean = 0
#     for mean_other in other_centers:
#         extra_mean += cos_sim(sample.unsqueeze(1), mean_other, temp_fac)
#     contrast_loss += -torch.log( cos_sim(sample.unsqueeze(1), target_center, temp_fac)
#                         / (cos_sim(sample.unsqueeze(1), target_center, temp_fac) + extra_mean))
# return contrast_loss / target_samples.size(1)
