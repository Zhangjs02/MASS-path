# -*- coding: utf-8 -*-
"""GPU-accelerated ManiFeSt feature selection algorithm"""

import cupy as cp
from cupy.linalg import eigh, svd  
from cuml.metrics import pairwise_distances

def construct_kernel_gpu(X, y, kernel_scale_factor, percentile=50):
    label = cp.unique(y)
    x_1 = X[cp.where(y == label[0])[0], :].T  
    x_2 = X[cp.where(y == label[1])[0], :].T 

    K1_dis = pairwise_distances(x_1, metric='euclidean')
    K2_dis = pairwise_distances(x_2, metric='euclidean')

    mask1 = ~cp.eye(K1_dis.shape[0], dtype=bool)
    mask2 = ~cp.eye(K2_dis.shape[0], dtype=bool)
    
    epsilon1 = kernel_scale_factor * cp.percentile(K1_dis[mask1], percentile)
    epsilon2 = kernel_scale_factor * cp.percentile(K2_dis[mask2], percentile)

    K1 = cp.exp(-(K1_dis ** 2) / (2 * epsilon1 ** 2))
    K2 = cp.exp(-(K2_dis ** 2) / (2 * epsilon2 ** 2))
    return K1, K2

def calc_tol_gpu(matrix, var_type=cp.float32, energy_tol=0):
    tol = cp.max(matrix) * len(matrix) * cp.finfo(var_type).eps
    tol2 = cp.sqrt(cp.sum(matrix ** 2) * energy_tol)
    return cp.maximum(tol, tol2)

def spsd_geodesics_gpu(G1, G2, p=0.5, r=None, eigVecG1=None, eigValG1=None, eigVecG2=None, eigValG2=None):
    if eigVecG1 is None:
        eigValG1, eigVecG1 = eigh(G1)
    if eigVecG2 is None:
        eigValG2, eigVecG2 = eigh(G2)

    if r is None:
        tol = calc_tol_gpu(eigValG1)
        rank_G1 = len(cp.abs(eigValG1)[cp.abs(eigValG1) > tol])
        tol = calc_tol_gpu(eigValG2)
        rank_G2 = len(cp.abs(eigValG2)[cp.abs(eigValG2) > tol])
        r = min(rank_G1, rank_G2)

    maxIndciesG1 = cp.flip(cp.argsort(cp.abs(eigValG1))[-r:], 0)
    V1 = eigVecG1[:, maxIndciesG1]
    lambda1 = eigValG1[maxIndciesG1]

    maxIndciesG2 = cp.flip(cp.argsort(cp.abs(eigValG2))[-r:], 0)
    V2 = eigVecG2[:, maxIndciesG2]
    lambda2 = eigValG2[maxIndciesG2]

    O2, sigma, O1T = svd(V2.T @ V1, full_matrices=False)
    O1 = O1T.T

    sigma = cp.clip(sigma, -1, 1)
    theta = cp.arccos(sigma)

    U1 = V1 @ O1
    R1 = O1.T @ cp.diag(lambda1) @ O1

    U2 = V2 @ O2
    R2 = O2.T @ cp.diag(lambda2) @ O2

    tol = calc_tol_gpu(sigma)
    valid_ind = cp.where(cp.abs(sigma - 1) > tol)
    pinv_sin_theta = cp.zeros(theta.shape)
    pinv_sin_theta[valid_ind] = 1 / cp.sin(theta[valid_ind])

    UG1G2 = U1 @ cp.diag(cp.cos(theta * p)) + (cp.eye(G1.shape[0]) - U1 @ U1.T) @ U2 @ cp.diag(pinv_sin_theta) @ cp.diag(cp.sin(theta * p))
    return UG1G2, R1, R2, O1, lambda1

def get_operators_gpu(K1, K2, use_spsd=True):
    if use_spsd:
        eigValK1, eigVecK1 = eigh(K1)
        tol = calc_tol_gpu(eigValK1)
        rank_K1 = len(eigValK1[cp.abs(eigValK1) > tol])

        eigValK2, eigVecK2 = eigh(K2)
        tol = calc_tol_gpu(eigValK2)
        rank_K2 = len(eigValK2[cp.abs(eigValK2) > tol])

        min_rank = min(rank_K1, rank_K2)
        UK1K2, RK1, RK2, OK1, lambdaK1 = spsd_geodesics_gpu(K1, K2, p=0.5, r=min_rank, 
                                                          eigVecG1=eigVecK1, eigValG1=eigValK1,
                                                          eigVecG2=eigVecK2, eigValG2=eigValK2)

        RK1PowerHalf = OK1.T @ cp.diag(cp.sqrt(lambdaK1)) @ OK1
        RK1PowerMinusHalf = OK1.T @ cp.diag(1 / cp.sqrt(lambdaK1)) @ OK1
        e, v = eigh(RK1PowerMinusHalf @ RK2 @ RK1PowerMinusHalf)
        e = cp.maximum(e, 0)
        RK1K2 = RK1PowerHalf @ v @ cp.diag(cp.sqrt(e)) @ v.T @ RK1PowerHalf

        M = UK1K2 @ RK1K2 @ UK1K2.T

        eigValM, eigVecM = eigh(M)
        tol = calc_tol_gpu(eigValM)
        rank_M = len(eigValM[cp.abs(eigValM) > tol])

        min_rank = min(rank_K1, rank_M)
        UMK1, RM, RK1, OM, lambdaM = spsd_geodesics_gpu(M, K1, p=1, r=min_rank,
                                                      eigVecG1=eigVecM, eigValG1=eigValM,
                                                      eigVecG2=eigVecK1, eigValG2=eigValK1)

        RMPowerHalf = OM.T @ cp.diag(cp.sqrt(lambdaM)) @ OM
        RMPowerMinusHalf = OM.T @ cp.diag(1 / cp.sqrt(lambdaM)) @ OM
        e, v = eigh(RMPowerMinusHalf @ RK1 @ RMPowerMinusHalf)
        e = cp.maximum(e, tol)
        logarithmic = RMPowerHalf @ v @ cp.diag(cp.log(e)) @ v.T @ RMPowerHalf

        D = UMK1 @ logarithmic @ UMK1.T
    else:
        K1 += cp.eye(K1.shape[0]) * cp.finfo(cp.float32).eps * 2
        K2 += cp.eye(K2.shape[0]) * cp.finfo(cp.float32).eps * 2

        eigValK1, eigVecK1 = eigh(K1)
        K1PowerHalf = eigVecK1 @ cp.diag(cp.sqrt(eigValK1)) @ eigVecK1.T
        K1PowerMinusHalf = eigVecK1 @ cp.diag(1 / cp.sqrt(eigValK1)) @ eigVecK1.T
        e, v = eigh(K1PowerMinusHalf @ K2 @ K1PowerMinusHalf)
        e = cp.maximum(e, 0)
        M = K1PowerHalf @ v @ cp.diag(cp.sqrt(e)) @ v.T @ K1PowerHalf

        eigValM, eigVecM = eigh(M)
        MPowerHalf = eigVecM @ cp.diag(cp.sqrt(eigValM)) @ eigVecM.T
        MPowerMinusHalf = eigVecM @ cp.diag(1 / cp.sqrt(eigValM)) @ eigVecM.T
        e, v = eigh(MPowerMinusHalf @ K1 @ MPowerMinusHalf)
        e = cp.maximum(e, calc_tol_gpu(e))
        D = MPowerHalf @ v @ cp.diag(cp.log(e)) @ v.T @ MPowerHalf

    return M, D

def compute_manifest_score_gpu(D):
    eigValD, eigVecD = eigh(D)
    eigVec_norm = eigVecD ** 2
    return eigVec_norm @ cp.abs(eigValD)

def ManiFeSt_gpu(X, y, kernel_scale_factor=1, use_spsd=True, percentile=50):
    K1, K2 = construct_kernel_gpu(X, y, kernel_scale_factor, percentile)
    M, D = get_operators_gpu(K1, K2, use_spsd)
    score = compute_manifest_score_gpu(D)
    return score

