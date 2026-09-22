# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

"""K-means clustering + Sinkhorn-Knopp entropic OT helpers.

Vendored verbatim from CHaRS (`/home/dixon/projects/CHaRS/tox_and_t2i/chars/utils/nonparam.py`,
"Concept Heterogeneity-aware Representation Steering") to make `act.hooks.chars_transport`'s
`GaussianOTClustHook`/`GaussianOTPCAHook` importable -- `chars/hooks/transport.py` itself does
`from act.utils.nonparam import *`, i.e. it already expects this module to live here; it just
wasn't part of upstream `ml-act`. No changes from the CHaRS source.
"""

from sklearn.cluster import KMeans
import numpy as np
import torch

def get_clusters(acts_normed, k):
    assert len(acts_normed.shape) == 3, f"Invalid Shape: {acts_normed.shape}"
    X = acts_normed
    acts_normed_clusters = np.ones((*X.shape[:1], k, X.shape[-1]))
    acts_normed_labels = np.ones(X.shape[:2])
    inertia = 0
    for i in range(X.shape[0]):
        xi = X[i].reshape(-1, X.shape[-1]) # dont reshape should still be the same
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(xi)
        inertia += kmeans.inertia_
        acts_normed_clusters[i] = kmeans.cluster_centers_
        acts_normed_labels[i] = kmeans.labels_
    return acts_normed_clusters, acts_normed_labels, inertia


# Compute Cluster Weights (Marginals)
def compute_cluster_weights(acts_normed_labels, k):
    X = acts_normed_labels
    acts_normed_cluster_weights = np.ones((*X.shape[:1], k)) # also equals to acts_normed_clusters.shape[:2]
    for i in range(X.shape[0]):
        acts_normed_cluster_weights[i] = np.array([np.sum(X[i] == cl) / len(X[i]) for cl in range(k)])
    return acts_normed_cluster_weights


def calculate_cost_matrix(harmful_acts_normed_clusters, harmless_acts_normed_clusters):
    return np.sum((harmful_acts_normed_clusters[:, :, None, :] - harmless_acts_normed_clusters[:, None, :, :]) ** 2, axis=-1)


def calculate_kernel_matrix(cost_matrix, eps=0.1):
    return np.exp(-cost_matrix / eps)

def calculate_kernel_matrix_adaptive(cost_matrix):
    medians = np.median(cost_matrix, axis = (-1, -2), keepdims=True)
    return np.exp(-cost_matrix / medians)


def sinkhorn_knopp_single(harmful_acts_normed_cluster_weights_sliced, harmless_acts_normed_cluster_weights_sliced, kernel_matrix_sliced, k, max_iter=1000, tau=1e-6):
    u = np.ones(k) # Row scaling vector
    v = np.ones(k) # Column scaling vector

    for t in range(max_iter):
        v_prev = v.copy()
        u = harmful_acts_normed_cluster_weights_sliced / (kernel_matrix_sliced @ v) # Update u: Row normalization
        v = harmless_acts_normed_cluster_weights_sliced / (kernel_matrix_sliced.T @ u) # Update v: Column normalization

        if np.linalg.norm(v - v_prev, ord=1) < tau:
            break

    P_star = np.diag(u) @ kernel_matrix_sliced @ np.diag(v)
    return P_star

def sinkhorn_knopp_multi(harmful_acts_normed_cluster_weights, harmless_acts_normed_cluster_weights, kernel_matrix, k, max_iter=1000, tau=1e-6):
    P_star = np.ones((*harmful_acts_normed_cluster_weights.shape[:1], k, k))
    for i in range(harmful_acts_normed_cluster_weights.shape[0]):
        P_star[i] = sinkhorn_knopp_single(
            harmful_acts_normed_cluster_weights[i],
            harmless_acts_normed_cluster_weights[i],
            kernel_matrix[i],
            k,
            max_iter=max_iter,
            tau=tau
        )
    return torch.Tensor(P_star)


##########################################################################################################################################################################

def calculate_similarity2_torch(x, acts_normed_clusters, similarity_kernel):
    if len(x.shape) == 3:
        B, L, D = x.shape[0], x.shape[1], x.shape[2]
        BL = B*L
    elif len(x.shape) == 2:
        B, L = None, None
        BL, D = x.shape[0], x.shape[1]
    else:
        raise NotImplementedError

    # Similarity Kernel
    if similarity_kernel == "gaussian":
        similarity_hs_src = torch.exp(- (torch.linalg.norm(x.reshape(BL, D)[:,None,:]-acts_normed_clusters[None,:,:], dim=-1) ** 2) / (D ** 0.5) )
        similarity_hs_src = similarity_hs_src / torch.sum(similarity_hs_src, dim = -1, keepdim = True)
    elif similarity_kernel == "adaptive_gaussian":
        similarity_hs_src = (torch.linalg.norm(x.reshape(BL, D)[:,None,:]-acts_normed_clusters[None,:,:], dim=-1) ** 2)
        similarity_hs_src = torch.exp(- (similarity_hs_src / torch.median(similarity_hs_src, dim = -1, keepdim=True).values))
        similarity_hs_src = similarity_hs_src / torch.sum(similarity_hs_src, dim = -1, keepdim = True)
    else:
        raise NotImplementedError

    return similarity_hs_src

def get_steering_direction2_torch(harmful_acts_normed_clusters, harmless_acts_normed_clusters, P_star, x, sim):
    # Compute the kernel matrix for the current point x
    similarity_x_harmful = calculate_similarity2_torch(x, harmful_acts_normed_clusters, sim)

    # Compute the steering direction
    num = P_star[None,:,:] * similarity_x_harmful[:,:,None]
    num = num[:,:,:,None] * (harmless_acts_normed_clusters[None,None,:,:] - harmful_acts_normed_clusters[None,:,None,:])
    num = num.sum(axis=-3).sum(axis=-2)
    denom = P_star[None,:,:] * similarity_x_harmful[:,:,None]
    denom = denom.sum(axis=-2).sum(axis=-1)+1e-8
    steering_direction = num / denom[:,None]

    return steering_direction # 36, 2, 2048

def pca_get_steering_direction2_torch(harmful_acts_normed_clusters, harmless_acts_normed_clusters, P_star, v_bar, pc_scores, top_k_pc, x, sim):
    # Compute the kernel matrix for the current point x
    similarity_x_harmful = calculate_similarity2_torch(x, harmful_acts_normed_clusters, sim)

    # Compute the steering direction
    num = P_star[None, :, :] * similarity_x_harmful[:, :, None]
    # L tokens, multiply with pc scores: "L i j, k i j -> L k i j"
    num = num[:, None, :, :] * pc_scores[None, :, :, :]
    num = num.sum(axis = -2).sum(axis = -1) # L k i j -> L k
    denom = P_star[None, :, :] * similarity_x_harmful[:, :, None]
    denom = denom.sum(axis = -2).sum(axis = -1) # L,
    pc_coeff = num / denom[:, None] # L k, k -> L, k
    scaled_pc = pc_coeff[:, :, None] * top_k_pc[None, :, :] # L k, k D -> L, k, D
    scaled_pc = scaled_pc.sum(axis = -2) # L, k, D -> L, D
    feature_direction = v_bar[None, :] + scaled_pc

    return feature_direction
