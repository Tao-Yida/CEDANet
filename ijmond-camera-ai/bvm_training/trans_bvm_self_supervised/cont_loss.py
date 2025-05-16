import torch
import torch.nn.functional as F
import numpy as np

def cos_sim(vec_a, vec_b, temp_fac=0.1):
    # Compute the cosine similarity score of the input 2 vectors scaled by temperature factor
    # L2 normalization of vectors
    # vec_a: (N,D), vec_b: (ZxD)
    norm_vec_a = F.normalize(vec_a, dim=1) 
    norm_vec_b = F.normalize(vec_b, dim=1)
    # Cosine similarity calculation 100x16 
    cos_sim_val = torch.matmul(norm_vec_a, norm_vec_b.transpose(-1, -2)) / temp_fac # (NxD) (DxZ) -> NxZ
    return cos_sim_val # NxZ  N-samples Z-num of classes

def region_loss(class1_samples, mean_class1, mean_other_classes, temp_fac=0.1, epsilon=1e-8):
    # class1_samples: (D, N), mean_class1: (Dx1), mean_other_classes: (DxZ)
    class1_samples = class1_samples.T  # Shape: (N, D) = (100, 16)
    mean_class1 = mean_class1.T  # 16x1 -> 1x16
    mean_other_classes = mean_other_classes.T # 16xZ -> Zx16
    smoke_loss = 0
    # Compute cosine similarities
    sim_with_class1 = cos_sim(class1_samples, mean_class1, temp_fac)  # Shape: (N, 1)
    sim_with_others = cos_sim(class1_samples, mean_other_classes, temp_fac).sum(dim=-1, keepdim=True)  # Shape: (N, Z) -> (N, 1), after sum. Z is the number of other classes
    # Calculate smoke loss
    # To avoid the negative in the similarity we clamp that vectors
    sim_with_class1 = torch.clamp(sim_with_class1, min=epsilon)
    sim_with_others = torch.clamp(sim_with_others, min=epsilon)
    smoke_loss = - torch.log(sim_with_class1 / (sim_with_class1 + sim_with_others)) # (N, 1)
    # Return the mean loss
    return smoke_loss.mean()

def intra_inter_contrastive_loss(features, masks, num_samples=100, margin=1.0, inter=True):
    """
    Compute intra- and inter-image contrastive loss for two classes (smoke and background).
    
    Args:
        features (Tensor): Model output features of shape (B, D, H, W).
        masks (Tensor): Ground truth masks of shape (B, H, W), with 1 for smoke and 0 for background.
        num_samples (int): Number of pixels to sample per class.
        margin (float): Margin parameter for contrastive loss.
        inter (bool): If True, calculate inter-image contrastive loss; if False, calculate intra-image contrastive loss.
        
    Returns:
        loss (Tensor): Calculated contrastive loss.
    """
    batch_size, feature_dim, height, width = features.size()
    total_loss = 0.0
    for i in range(batch_size):
        feature_map = features[i]  # D=16 x H=352 x W=352
        mask = masks[i]  # 1 x H=352 x W=352,

        # Separate features into smoke and background based on mask
        smoke_features = feature_map[:, mask.squeeze(0) > 0].view(feature_dim, -1)  # D x N_smoke 
        background_features = feature_map[:, mask.squeeze(0) == 0].view(feature_dim, -1)  # D x N_background 
        # print(background_features.shape, smoke_features.shape)

        # Compute mean feature vectors for smoke and background within the same image
        mean_smoke = smoke_features.mean(dim=1, keepdim=True) if smoke_features.size(1) > 0 else None # 16 x 1 
        mean_background = background_features.mean(dim=1, keepdim=True) if background_features.size(1) > 0 else None # 16 x 1
        # print(mean_background.shape, mean_smoke.shape)
        if mean_smoke is None or mean_background is None:
            batch_size -= 1
            continue
        # Normalize mean smoke and mean background
        mean_smoke = F.normalize(mean_smoke, dim=0)
        mean_background = F.normalize(mean_background, dim=0)

        # Sample features from each class within the same image
        if smoke_features.size(1) > num_samples:
            smoke_samples = smoke_features[:, torch.randperm(smoke_features.size(1))[:num_samples]] # 16 x 100
        else:
            smoke_samples = smoke_features
        
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
            smoke_loss = region_loss(smoke_samples, mean_smoke, mean_background, temp_fac=0.1)
            # print(smoke_loss)
            # 2. For each background sample compute the loss 
            # (Maybe it doesn't make sense to calcualte the loss for the background samples as it's not a class.)
            # background_loss = region_loss(background_samples, mean_background, mean_smoke, temp_fac=0.1)
            # loss = torch.div(torch.add(smoke_loss, background_loss), 2)
            # print(loss)
            total_loss += smoke_loss


    return total_loss / batch_size

################################################################
            # for sample in smoke_samples.T:
            #     smoke_loss += -torch.log( cos_sim(sample.unsqueeze(1), mean_smoke) 
            #                 / (cos_sim(sample.unsqueeze(1), mean_smoke) + cos_sim(sample.unsqueeze(1), mean_background)))
            # smoke_loss /= smoke_samples.size(1)

                # for sample in class1_samples.T:
        #     extra_mean = 0
    #     for mean_other in mean_other_classes:
    #         extra_mean += cos_sim(sample.unsqueeze(1), mean_other, temp_fac)
    #     smoke_loss += -torch.log( cos_sim(sample.unsqueeze(1), mean_class1, temp_fac) 
    #                         / (cos_sim(sample.unsqueeze(1), mean_class1, temp_fac) + extra_mean))
    # return smoke_loss / class1_samples.size(1)