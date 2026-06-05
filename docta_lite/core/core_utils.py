import torch
import torch.nn.functional as F


def cosDistance(features):
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    distance_matrix = 1.0 - similarity_matrix
    return distance_matrix


def cosDistance_chunked(features, all_features, chunk_size=1024):
    features = F.normalize(features, dim=1)
    features = features.to('cuda').half()
    num_samples = features.size(0)
    all_num_samples = all_features.size(0)
    distance_matrix = torch.zeros(num_samples, all_num_samples, device=features.device, dtype=features.dtype)

    similarity_matrix = torch.matmul(features, all_features.T)
    distance_matrix = 1.0 - similarity_matrix

    return distance_matrix


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
