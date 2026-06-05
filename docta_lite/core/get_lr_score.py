"""Long-tail diversity score for each sample."""

import math
import torch
import numpy as np
from collections import Counter
from .hoc import get_consensus_patterns
from docta_lite.datasets.customize import CustomizedDataset


def lt_score(data, feature_type, k=10):
    return score_from_embedding(data, k)


def score_from_embedding(data, k):
    sample = np.arange(len(data))
    print("Getting consensus patterns for diversity scoring...")
    _, values = get_consensus_patterns(data, sample, k=k)
    np_values = values.numpy()
    mean_dist = np.mean(np_values, 1)
    lt_scores = []
    for i in range(mean_dist.shape[0]):
        tmp = np.round((2.0 / (1 + math.exp(-mean_dist[i]))) - 1.0, 4)
        lt_scores.append(tmp)
    return lt_scores
