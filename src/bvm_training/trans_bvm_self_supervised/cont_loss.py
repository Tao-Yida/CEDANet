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
    Calculate contrastive loss between target class samples and class centers, encouraging samples to be close to their own class center and far from other class centers.

    Args:
        target_samples (torch.Tensor): Target class sample features, shape (D, N), where D is feature dimension and N is number of samples.
        target_center (torch.Tensor): Target class center vector, shape (D, 1).
        other_centers (torch.Tensor): Other class center vectors, shape (D, Z), where Z is the number of other classes.
        temp_fac (float): Temperature factor for scaling cosine similarity, default 0.1.
        epsilon (float): Small constant to avoid division by zero, default 1e-8.

    Returns:
        torch.Tensor: Contrastive loss value, scalar.
    """
    # Transpose all inputs to fit the input format of cos_sim function
    target_samples = target_samples.T  # Shape: (D, N) -> (N, D), e.g. (100, 16)
    target_center = target_center.T  # Shape: (D, 1) -> (1, D)
    other_centers = other_centers.T  # Shape: (D, Z) -> (Z, D)

    # Calculate similarity between each target sample and target class center
    sim_with_target = cos_sim(target_samples, target_center, temp_fac)  # Shape: (N, 1)

    # Calculate sum of similarities between each target sample and all other class centers
    sim_with_others = cos_sim(target_samples, other_centers, temp_fac).sum(dim=-1, keepdim=True)  # Shape: (N, Z) -> (N, 1), after summing

    # Clip similarity values to ensure non-negative
    sim_with_target = torch.clamp(sim_with_target, min=epsilon)
    sim_with_others = torch.clamp(sim_with_others, min=epsilon)

    # Calculate contrastive loss: -log(positive similarity / (positive similarity + negative similarity))
    contrast_loss = -torch.log(sim_with_target / (sim_with_target + sim_with_others))  # Shape: (N, 1)

    # Return average loss of all samples
    return contrast_loss.mean()


def intra_inter_contrastive_loss(features, masks, num_samples=100, margin=1.0, inter=True):
    """
    Calculate intra or inter image contrastive loss, distinguishing between smoke and background classes.

    Args:
        features (Tensor): Model output feature map, shape (B, D, H, W), where B is batch size, D is feature dimension, H and W are spatial dimensions.
        masks (Tensor): Ground truth masks, shape (B, H, W), where 1 represents smoke and 0 represents background.
        num_samples (int): Number of pixel points sampled per class, default 100.
        margin (float): Margin parameter for contrastive loss, default 1.0.
        inter (bool): If True, calculate inter-image contrastive loss; if False, calculate intra-image contrastive loss.

    Returns:
        loss (Tensor): Calculated contrastive loss value.
    """
    batch_size, feature_dim, height, width = features.size()
    total_loss = 0.0

    for i in range(batch_size):
        # Extract feature map and corresponding mask for each sample
        # features: (B, D, H, W) -> (D, H, W) for each sample
        # masks: (B, H, W) -> (1, H, W) for each sample
        feature_map = features[i]  # D=16 x H=352 x W=352
        mask = masks[i]  # 1 x H=352 x W=352,

        # Separate features into smoke and background based on mask
        smoke_features = feature_map[:, mask.squeeze(0) > 0].view(feature_dim, -1)  # D x N_smoke
        background_features = feature_map[:, mask.squeeze(0) == 0].view(feature_dim, -1)  # D x N_background
        # print(background_features.shape, smoke_features.shape)

        # Average each column (same-class pixel vectors) to get the average feature vector for smoke/background within the image
        # Compute mean feature vectors for smoke and background within the same image
        mean_smoke = smoke_features.mean(dim=1, keepdim=True) if smoke_features.size(1) > 0 else None  # 16 x 1
        mean_background = background_features.mean(dim=1, keepdim=True) if background_features.size(1) > 0 else None  # 16 x 1
        # print(mean_background.shape, mean_smoke.shape)
        if mean_smoke is None or mean_background is None:
            batch_size -= 1  # Adjust subsequent average denominator
            continue

        # Normalize mean smoke and mean background
        # L2 normalize along channel dimension, making each mean vector's magnitude equal to 1
        mean_smoke = F.normalize(mean_smoke, dim=0)
        mean_background = F.normalize(mean_background, dim=0)

        # Sample features from each class within the same image
        # If smoke features count > num_samples, randomly sample num_samples pixel features
        if smoke_features.size(1) > num_samples:
            smoke_samples = smoke_features[:, torch.randperm(smoke_features.size(1))[:num_samples]]  # 16 x 100
        else:
            smoke_samples = smoke_features

        # Normalize again
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
# The following is deprecated legacy implementation code, kept for reference
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
