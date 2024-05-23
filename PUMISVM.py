import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib

from random import random, sample
import math
from copy import deepcopy
import numpy as np
from cvxopt import solvers as cvxopt_solvers
from cvxopt import matrix
import utils
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import lines
from qpsolvers import solve_qp
import statistics
from numpy import linalg as LA
from scipy.linalg import solve_sylvester

class PUMISVM:

    def __init__(self, gammas= None, kernel_method = "linear", kernel_kwargs = None, scale_factor = "auto", version = 6, normal_int_ratio = None, thres= None, num_planes = 1, PU_kwargs = None):
        """
        Parameters
        ----------
        scale_factor : float 
            It is used for numerical scalability and it solves rank mismatch error in cvxopt_solvers.qp(P, q, G, h, A, b).
        """

        self.version = version
        if self.version == 0:
            self.gammas = deepcopy(gammas) if gammas is not None else [10., 0.1, 1., 1., 0.95]
        elif self.version == 1:
            self.gammas = deepcopy(gammas) if gammas is not None else [1.5, 1., 1., 0.1, 0.95]
        elif self.version == 4:
            self.gammas = deepcopy(gammas) if gammas is not None else [1., 1., None, 0.1, None]
        elif self.version == 6:
            self.gammas = deepcopy(gammas) if gammas is not None else [1., 1., 1., 1., 1.]
        elif self.version == "PU":
            self.gammas = deepcopy(gammas) if gammas is not None else [1., 1., 1., 1., 1., 0.1, 0.1]
        else:
            raise Exception(NotImplementedError)
        
        if self.version not in [4, 6]: assert(0. <= self.gammas[4] <= 1.)
        self.kernel_method = kernel_method
        self.kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {}
        self.scale_factor = scale_factor
        self.normal_int_ratio = normal_int_ratio if normal_int_ratio is not None else [-1.6, 1.6]
        self.thres = thres
        self.num_planes = num_planes

        self.PU_kwargs = PU_kwargs
        if self.version in ["PU"]:
            assert(self.PU_kwargs is not None)
            self.num_planes = 1
            # self.kernel_method = "linear"
    
    def kernel(self, *args, if_mix_PU = True, **kwargs):
        if self.version in ["PU"] and if_mix_PU:
            result_PU = self.kernel_inner(*args, if_use_H= True, kernel_method= "linear", **kwargs)
            result_original = self.kernel_inner(*args, if_use_H= False, kernel_method= self.PU_kwargs["original_data_kernel"]["kernel_method"], kernel_kwargs= self.PU_kwargs["original_data_kernel"]["kernel_kwargs"], **kwargs)
            assert(0 <= self.gammas[6] <= 1)
            if False:
                return self.gammas[6] * result_PU / (np.mean(np.abs(result_PU))) + (1. - self.gammas[6]) * result_original / (np.mean(np.abs(result_original)))
            else:
                return self.gammas[6] * result_PU + (1. - self.gammas[6]) * result_original 
        else:
            return self.kernel_inner(*args, **kwargs)

    def kernel_inner(self, X_target, kernel_method = None, kernel_kwargs = None, if_use_H = True):
        """Calculate inner products for K(X_target.T @ X_train)"""
        if kernel_method is None: 
            kernel_method = self.kernel_method
        if kernel_kwargs is None:
            kernel_kwargs = self.kernel_kwargs

        if self.version in ["PU"] and if_use_H:
            X_target_loc = self.H @ X_target
            X_concat_train_loc = self.H @ self.X_concat_train
        else:
            X_target_loc = X_target
            X_concat_train_loc = self.X_concat_train 

        if kernel_method == "linear":
            result1 = X_concat_train_loc.T @ X_target_loc
        elif kernel_method == "poly":
            result1 = (kernel_kwargs["c"] + X_concat_train_loc.T @ X_target_loc) ** kernel_kwargs["degree"]
        elif kernel_method == "rbf":
            if False:
                result = np.zeros((X_concat_train_loc.shape[1], X_target_loc.shape[1]))
                for it in range(X_concat_train_loc.shape[1]):
                    for ip in range(X_target_loc.shape[1]):
                        result[it, ip] = np.exp(-(1 / kernel_kwargs["sigma"] ** 2) * np.linalg.norm())
                return result
            else:
                result1 = np.exp(-(1 / kernel_kwargs["sigma"] ** 2) * np.linalg.norm(X_concat_train_loc.T[:, np.newaxis] - X_target_loc.T[np.newaxis, :], axis=2) ** 2)
                if True:
                    assert(np.allclose(np.exp(- (1 / kernel_kwargs["sigma"] ** 2) * np.sum((X_target_loc.T - X_concat_train_loc.T[:,np.newaxis])**2,axis=-1)), result1))
        return result1.T


    def fit(self, X, y, similarities = None, is_PL= None):
        """
        Example
        -------
        X = [np.array([[1.2, 3.5], [0.6, -1.2], [3.6, -2.4]]).T, np.array([[0.7, -1.1], [1.2, 4.5], [2.0, 4.8], [9.2, -4.5]]).T, ...], (2 features: 3 instances, 4 instances, ...), len(X) = number of bags.
        y = ["unlabeled", "A", ...], len(y) = number of bags.
        is_PL = [1, 0, 1, 1, 0, ...], len(is_PL) = number of bags.
        """

        if self.version in ["PU"]: 
            assert(is_PL is not None)
        else:
            assert(is_PL is None)
            is_PL = [1 for i in range(len(y))]
        assert(len(X) == len(y) == len(is_PL))

        self.Ns_all = len(X)
        self.Ns_class_acc = [0]
        self.Ns_all_labeled = 0
        self.nis_all = sum([x.shape[1] for x in X])
        self.nis_bag = []
        self.nis_class_acc_bag = []
        self.nis_class_acc = [0]

        self.N_factor = 1.0 ## self.Ns_all
        self.unique_labels = list(set(y))

        X_reorder = []
        X_L = []
        X_U = []
        is_PL_reorder = []
        self.start_end_idx_acc = []
        idx_acc = 0
        for label in self.unique_labels:
            nis = []
            self.nis_class_acc_bag.append(([0] if len(self.nis_class_acc_bag) == 0 else [self.nis_class_acc_bag[-1][-1]]))
            for i in range(len(X)):
                if y[i] == label:
                    X_reorder.append(X[i])
                    nis.append(X[i].shape[1])
                    self.start_end_idx_acc.append((idx_acc, idx_acc + X[i].shape[1]))
                    idx_acc += X[i].shape[1]
                    self.nis_class_acc_bag[-1].append(self.nis_class_acc_bag[-1][-1] + X[i].shape[1])
                    if is_PL[i] == 1:
                        X_L.append(X[i])
                    else:
                        X_U.append(X[i])
                    is_PL_reorder.append(is_PL[i])
            self.nis_bag.append(nis)
            self.nis_class_acc.append(self.nis_class_acc[-1] + sum(nis))
            assert(self.nis_class_acc[-1] - self.nis_class_acc[-2] == self.nis_class_acc_bag[-1][-1] - self.nis_class_acc_bag[-1][0])
            self.Ns_class_acc.append(self.Ns_class_acc[-1] + len(nis))
            if label != "unlabeled":
                self.Ns_all_labeled += len(nis)
        self.X_concat_train = np.concatenate(X_reorder, axis= 1)
        assert(self.X_concat_train.shape[1] == self.nis_all)

        if self.version not in ["PU"]:
            iters = 1
        else:
            X_L = np.concatenate(X_L, axis= 1)
            X_U = np.concatenate(X_U, axis= 1)
            iters = self.PU_kwargs["iters"]
            self.Z_L = np.random.random((self.PU_kwargs["d_z"], X_L.shape[1]))
            # self.G = np.random.random((self.PU_kwargs["d_z"], self.nis_all))
            self.Lambda_list = [np.random.random((self.PU_kwargs["d_z"], self.PU_kwargs["d_z"]))]
            self.mu_list = deepcopy(self.PU_kwargs["mus_list"])
            self.D_list = [np.zeros((X_U.shape[1], X_U.shape[1]))]
            self.F = np.random.random((self.X_concat_train.shape[0], self.PU_kwargs["d_z"]))
            self.H = np.random.random((self.PU_kwargs["d_z"], self.X_concat_train.shape[0]))
            lambdas = np.random.random((2 * self.nis_all))
        
        for it in range(iters):
            if self.version in ["PU"]: ## PU learning variables
                Z_U_T = (self.H @ X_U).T
                idx_acc = 0
                for i in range(len(is_PL_reorder)):
                    if is_PL_reorder[i] == 0:
                        idx_cut = (idx_acc, idx_acc + self.start_end_idx_acc[i][1] - self.start_end_idx_acc[i][0])
                        if it == 0: assert(not np.any(self.D_list[0][idx_cut[0] : idx_cut[1], idx_cut[0] : idx_cut[1]]))
                        value = ((LA.norm(Z_U_T[idx_cut[0] : idx_cut[1], :], 'fro') + self.PU_kwargs["delta"]) ** (-0.5)) / 2
                        for j in range(idx_cut[0], idx_cut[1]):
                            self.D_list[0][j, j] = value
                        idx_acc += self.start_end_idx_acc[i][1] - self.start_end_idx_acc[i][0]
                assert(self.D_list[0][-1, -1] != 0)
                self.Z_L = LA.inv(self.F.T @ self.F) @ self.F.T @ X_L
                self.F = solve_sylvester(self.mu_list[0] * self.H.T @ self.H, 2 * self.gammas[4] * self.Z_L @ self.Z_L.T, 2 * self.gammas[4] * X_L @ self.Z_L.T + self.mu_list[0] * self.H.T - self.H.T @ self.Lambda_list[0])

                lambdas_epsilon = lambdas[0:self.nis_all]
                lambdas_delta = lambdas[self.nis_all:2*self.nis_all]
                self.H = (np.outer(self.H @ self.X_concat_train @ (lambdas_epsilon - lambdas_delta), (lambdas_epsilon - lambdas_delta).T) @ self.X_concat_train.T + self.mu_list[0] * self.F.T - self.Lambda_list[0] @ self.F.T) @ LA.inv(self.mu_list[0] * self.F @ self.F.T + 2 * self.gammas[5] * X_U @ self.D_list[0] @ X_U.T)

                self.Lambda_list[0] = self.mu_list[0] * (np.identity(self.PU_kwargs["d_z"]) - self.H @ self.F)
                self.mu_list[0] *= self.PU_kwargs["rho"]
            if self.version in ["PU"] and it < iters - 1:
                XTX = self.kernel(self.X_concat_train, kernel_method = "linear", if_mix_PU = False)
            else:
                XTX = self.kernel(self.X_concat_train)

            self.lambdas_list = []
            self.lambdas_epsilon_list = []
            self.lambdas_delta_list = []
            self.avg_response_train_list = []

            self.h_is_k_list = []
            self.h_std_k_list = []
            self.h_k_list = []


            for g in range(self.num_planes):
                C1 = np.zeros((self.nis_all, self.nis_all))
                C2 = np.zeros(self.nis_all)
                C3 = 0.

                inv_S_k = []
                M_k = []
                P_k = []
                U_k = []
                const_base_k = []

                if self.version == 1:
                    M_L = np.zeros((self.Ns_all_labeled, self.nis_all))
                    S_L = np.zeros((self.Ns_all_labeled, self.Ns_all_labeled))
                
                elif self.version == 0:
                    for k in range(len(self.unique_labels)):
                        ## Create M_k, P_k, U_k
                        size_Ik = self.Ns_class_acc[k + 1] - self.Ns_class_acc[k]

                        if self.unique_labels[k] == "unlabeled":
                            assert(not np.any(C1[self.nis_class_acc[k]:self.nis_class_acc[k + 1], self.nis_class_acc[k]:self.nis_class_acc[k + 1]]))
                            assert(not np.any(C2[self.nis_class_acc[k]:self.nis_class_acc[k + 1]]))
                            # self.C1[self.nis_class_acc[k]:self.nis_class_acc[k + 1], self.nis_class_acc[k]:self.nis_class_acc[k + 1]] = np.zeros((sum(self.nis_bag[k]), sum(self.nis_bag[k])))
                            # self.C2[self.nis_class_acc[k]:self.nis_class_acc[k + 1]] = np.zeros(sum(self.nis_bag[k]))
                            # self.C3 += 0.

                            M_k.append(None)
                            inv_S_k.append(None)
                            P_k.append(None)
                            U_k.append(None)
                            const_base_k.append(None)
                        else:
                            M_k_this = np.zeros((size_Ik, sum(self.nis_bag[k])))
                            P_k_this = np.zeros((sum(self.nis_bag[k]), sum(self.nis_bag[k])))
                            S_k_this = np.zeros((size_Ik, size_Ik))
                            if False:
                                if similarities is None:
                                    for ia in range(size_Ik):
                                        for ib in range(size_Ik):
                                            if ia == ib:
                                                S_k_this[ia, ib] = 0.
                                            else:
                                                S_k_this[ia, ib] = 1. / max(np.mean(np.abs(np.mean(X_reorder[self.Ns_class_acc[k] + ia], axis = 1) - np.mean(X_reorder[self.Ns_class_acc[k] + ib], axis = 1))), 1e-16)
                                else:
                                    raise Exception(NotImplementedError)
                                if k == len(self.unique_labels) - 1: assert(self.Ns_class_acc[k] + ia == self.Ns_all - 1 and S_k_this[-1, -2] != 0)

                                if False:
                                    S_k_this = (S_k_this - np.min(S_k_this)) / (np.max(S_k_this) - np.min(S_k_this))
                                else:
                                    for i in range(S_k_this.shape[0]):
                                        S_k_this[i, :] = S_k_this[i, :] / np.sum(S_k_this[i, :])
                            else:
                                if False:
                                    S_k_this = S_k_this + (1 / (size_Ik - 1))
                                    S_k_this = S_k_this - np.identity(size_Ik) * (1 / (size_Ik - 1))
                                else:
                                    S_k_this = S_k_this + (1 / size_Ik)
                            if True:
                                if False:
                                    inv_S_k_this = np.linalg.inv(S_k_this - np.identity(size_Ik))
                                else:
                                    print(f"Exact inverse of inv_S_k_this for k = {k} does not exits.")
                                    inv_S_k_this = np.linalg.pinv(S_k_this - np.identity(size_Ik))
                                    inv_S_k_this = (inv_S_k_this + inv_S_k_this.T) / 2.
                            inv_gap = np.abs((S_k_this - np.identity(size_Ik)) @ inv_S_k_this - np.identity(size_Ik))
                            print(f"Inv gap is {np.mean(inv_gap)}.")
                            inv_S_k.append(inv_S_k_this)

                            D_sk = np.zeros((size_Ik, size_Ik))
                            s_Ik = np.zeros(self.nis_class_acc[k + 1] - self.nis_class_acc[k])
                            for i in range(size_Ik):
                                D_sk[i, i] = 1 / (2 * self.gammas[2] * (S_k_this[i, i] - 1))
                                assert(not np.any(s_Ik[self.nis_class_acc_bag[k][i]:self.nis_class_acc_bag[k][i+1]]))
                                s_Ik[self.nis_class_acc_bag[k][i]:self.nis_class_acc_bag[k][i+1]] = (1 / (2 * self.gammas[2] * (S_k_this[i, i] - 1))) ** 2
                            const_base_k.append(D_sk)

                            U_k_this = inv_S_k_this @ const_base_k[k]

                            ni_idx = 0
                            for ini in range(len(self.nis_bag[k])):
                                M_k_this[ini, ni_idx:(ni_idx + self.nis_bag[k][ini])] = 1.
                                P_k_this[ni_idx:(ni_idx + self.nis_bag[k][ini]), ni_idx:(ni_idx + self.nis_bag[k][ini])] = (1 / (2 * self.gammas[2] * (S_k_this[ini, ini] - 1))) ** 2
                                ni_idx = ni_idx + self.nis_bag[k][ini]
                            
                            assert(not np.any(C1[self.nis_class_acc[k]:self.nis_class_acc[k + 1], self.nis_class_acc[k]:self.nis_class_acc[k + 1]]))
                            C1[self.nis_class_acc[k]:self.nis_class_acc[k + 1], self.nis_class_acc[k]:self.nis_class_acc[k + 1]] = - M_k_this.T @ U_k_this @ M_k_this + self.gammas[2] * P_k_this

                            assert(not np.any(C2[self.nis_class_acc[k]:self.nis_class_acc[k + 1]]))
                            C2[self.nis_class_acc[k]:self.nis_class_acc[k + 1]] = - 2 * self.gammas[2] * self.gammas[3] * s_Ik.T ## + 2 * self.gammas[3] * np.ones(size_Ik) @ U_k_this @ M_k_this term has been removed.
                            # C3 += - np.sum(U_k_this) * (self.gammas[3] ** 2) + (const_base_k[k] ** 2) * self.gammas[2] * size_Ik * (self.gammas[3] ** 2) ## We don't need C3 which is constant.
                        
                            M_k.append(M_k_this)
                            P_k.append(P_k_this)
                            U_k.append(U_k_this)
                elif self.version in [6, "PU"]:
                    I_const = np.concatenate([np.identity(self.nis_all), - np.identity(self.nis_all)], axis= 1)
                    assert(I_const.shape[1] == 2 * self.nis_all)
                    U_L = np.zeros((2 * self.nis_all, 2 * self.nis_all))
                    for k in range(len(self.unique_labels)):
                        if self.unique_labels[k] != "unlabeled":
                            u_is= []
                            u_i_sum = 0.
                            for i in range(len(self.nis_class_acc_bag[k]) - 1):
                                n_i = self.nis_class_acc_bag[k][i + 1] - self.nis_class_acc_bag[k][i]
                                slice_n_i = slice(self.nis_class_acc_bag[k][i], self.nis_class_acc_bag[k][i + 1])
                                c1 = np.ones(2 * self.nis_all)
                                c1[slice_n_i] = 1 / (2 * self.gammas[0] * n_i)
                                c1[self.nis_class_acc_bag[k][i] + self.nis_all : self.nis_class_acc_bag[k][i + 1] + self.nis_all] = - 1 / (2 * self.gammas[1] * n_i)
                                u_i = c1 + (1 / (n_i * self.N_factor)) * np.ones(n_i).T @ XTX[slice_n_i, :] @ I_const
                                u_is.append(u_i)
                                u_i_sum = u_i_sum + u_i
                            assert(c1[-1] != 0)
                            u_i_sum = u_i_sum / (len(self.nis_class_acc_bag[k]) - 1)
                            for i in range(len(self.nis_class_acc_bag[k]) - 1):
                                U_iK = np.outer(u_i_sum - u_is[i], u_i_sum - u_is[i])
                                assert(U_iK.shape[1] == 2 * self.nis_all)
                                U_L = U_L + U_iK
                    assert(U_L.shape[1] == 2 * self.nis_all)

                
                if self.version not in [4, 6, "PU"]:
                    A = np.zeros((self.Ns_all - self.Ns_all_labeled, 2 * self.nis_all))
                else:
                    A = np.zeros((self.Ns_all, 2 * self.nis_all))
                instances_acc = 0
                bags_acc = 0
                sum_S_L = 0
                for k in range(len(self.unique_labels)):
                    size_Ik = self.Ns_class_acc[k + 1] - self.Ns_class_acc[k]
                    if self.version == 1 and self.unique_labels[k] != "unlabeled": ## S_L is normalized below
                        if False:
                            if similarities is None:
                                for ia in range(bags_acc, bags_acc + len(self.nis_bag[k])):
                                    for ib in range(bags_acc, bags_acc + len(self.nis_bag[k])):
                                        if ia == ib:
                                            self.S_L[ia, ib] = 0.
                                        else:
                                            self.S_L[ia, ib] = 1. / max(np.mean(np.abs(np.mean(X_reorder[ia], axis = 1) - np.mean(X_reorder[ib], axis = 1))), 1e-16)
                            else:
                                raise Exception(NotImplementedError)
                        else:
                            bags_slice = slice(bags_acc, bags_acc + self.Ns_class_acc[k+1] - self.Ns_class_acc[k])
                            if True:
                                S_L[bags_slice, bags_slice] = 1 / (size_Ik - 1)
                                S_L[bags_slice, bags_slice] += - np.identity(self.Ns_class_acc[k+1] - self.Ns_class_acc[k]) * 1 / (size_Ik - 1)
                                sum_S_L += (size_Ik - 1) * size_Ik / (size_Ik - 1)
                            else:
                                S_L[bags_slice, bags_slice] = 1 / (size_Ik)
                                sum_S_L += size_Ik


                    # assert(np.sum(self.S_L) == sum([(size_Ik ** 2 / (size_Ik - 1)) for ]))
                    for i in range(len(self.nis_bag[k])):
                        if self.unique_labels[k] == "unlabeled" and self.version not in [4, 6, "PU"]:
                            A[i, instances_acc:instances_acc+self.nis_bag[k][i]] = 1.
                            A[i, self.nis_all+instances_acc:self.nis_all+instances_acc+self.nis_bag[k][i]] = -1.
                        else:
                            if self.version == 1:
                                M_L[bags_acc, instances_acc:instances_acc+self.nis_bag[k][i]] = 1.
                            bags_acc += 1
                        instances_acc += self.nis_bag[k][i]
                    if self.version == 1:
                        k = self.unique_labels.index("unlabeled")
                        assert(not np.any(M_L[:, self.nis_class_acc[k]:self.nis_class_acc[k+1]]))

                if self.version == 1:
                    # assert(sum_S_L - 0.0001 < np.sum(S_L) < sum_S_L + 0.0001)   
                    assert((self.unique_labels[0] == "unlabeled" or S_L[0, 1] != 0) and (self.unique_labels[-1] == "unlabeled" or S_L[-1, -2] != 0))
                    assert(bags_acc == self.Ns_all_labeled and ((self.unique_labels[-1] == "unlabeled" and M_L[0, 0] == 1) or M_L[-1, -1] == 1.))
                    if False:
                        assert(self.unique_labels[-1] == "unlabeled" or S_L[-1, -2] == 1. / max(np.mean(np.abs(np.mean(X_reorder[self.Ns_all_labeled - 1], axis = 1) - np.mean(X_reorder[self.Ns_all_labeled - 2], axis = 1))), 1e-16))

                    if False:
                        for i in range(self.Ns_all_labeled):
                            if np.sum(S_L[i]) != 0:
                                S_L[i] = S_L[i] / np.sum(S_L[i])
                    else: ## This will alleviate the very large h_k problem. The magnitude of S_L or S_k heavily impacts the result.
                        S_L = (S_L - np.min(S_L)) / (np.max(S_L) - np.min(S_L))
                    try:
                        S_LT_inv = np.linalg.inv(np.identity(self.Ns_all_labeled) - S_L.T)
                    except:
                        print(f"Exact inverse of S_LT does not exits.")
                        S_LT_inv  = np.linalg.pinv(np.identity(self.Ns_all_labeled) - S_L.T)
                    try:
                        S_L_inv = np.linalg.inv(np.identity(self.Ns_all_labeled) - S_L)
                    except:
                        print(f"Exact inverse of S_L does not exits.")
                        S_L_inv  = np.linalg.pinv(np.identity(self.Ns_all_labeled) - S_L)
                    
                    inv_gap = np.abs((np.identity(self.Ns_all_labeled) - S_L.T) @ S_LT_inv - np.identity(self.Ns_all_labeled))
                    print(f"Inv gap is {np.mean(inv_gap)}.")
                    S_Lp = S_LT_inv.T @ S_LT_inv
                    assert(np.allclose(S_Lp, S_Lp.T))
                    assert(S_L[1, 0] > 0 and S_L[-2, -1] > 0)

                    M_L_concat = np.concatenate([- M_L, M_L], axis= 1)
                
                if self.version in [4, 6, "PU"]:
                    for bi in range(len(self.start_end_idx_acc)):
                        assert(not np.any(A[:, self.start_end_idx_acc[bi][0]:self.start_end_idx_acc[bi][1]]))
                        assert(not np.any(A[:, self.nis_all+self.start_end_idx_acc[bi][0]:self.nis_all+self.start_end_idx_acc[bi][1]]))
                        A[bi, self.start_end_idx_acc[bi][0] : self.start_end_idx_acc[bi][1]] = 1.
                        A[bi, self.nis_all + self.start_end_idx_acc[bi][0] : self.nis_all + self.start_end_idx_acc[bi][1]] = - 1.

                        # if is_PL_reorder[bi] == 1:
                        #     A[bi, self.start_end_idx_acc[bi][0]:self.start_end_idx_acc[bi][1]] = self.gammas[3]
                        #     A[bi, self.nis_all + self.start_end_idx_acc[bi][0] : self.nis_all + self.start_end_idx_acc[bi][1]] = - self.gammas[3]
                        # else:
                        #     A[bi, self.start_end_idx_acc[bi][0]:self.start_end_idx_acc[bi][1]] = 0.
                        #     A[bi, self.nis_all + self.start_end_idx_acc[bi][0] : self.nis_all + self.start_end_idx_acc[bi][1]] = 0.
                    assert(A[-1, -1] == - 1.)

                K = (- 1 / (2 * self.N_factor)) * XTX
                if self.version in [0, 4, 6, "PU"]:
                    if self.version == 0:
                        K = K + C1
                    P = np.zeros((2 * self.nis_all, 2 * self.nis_all))
                    P[0:self.nis_all, 0:self.nis_all] = K - (1 / (4 * self.gammas[0])) * np.identity(self.nis_all)
                    P[0:self.nis_all, self.nis_all:2*self.nis_all] = -K
                    P[self.nis_all:2*self.nis_all, 0:self.nis_all] = -K
                    P[self.nis_all:2*self.nis_all, self.nis_all:2*self.nis_all] = K - (1 / (4 * self.gammas[1])) * np.identity(self.nis_all)
                    # P = (P + P.T) / 2.
                    if self.version == 0:
                        q = np.concatenate([C2, -C2])
                    elif self.version in [6, "PU"]:
                        P = P - self.gammas[2] * U_L
                        if g > 0:
                            LX_prev = + (4 * self.gammas[4] / self.num_planes) * (self.lambdas_epsilon_list[-1] - self.lambdas_delta_list[-1]) @ K ## - sign is already included in K.
                            q = np.concatenate([LX_prev, -LX_prev])
                elif self.version == 1:
                    P = (- 1 / (8 * self.gammas[2])) * M_L_concat.T @ S_Lp @ M_L_concat
                    P[0:self.nis_all, 0:self.nis_all] += K - (1 / (4 * self.gammas[0])) * np.identity(self.nis_all)
                    P[0:self.nis_all, self.nis_all:2*self.nis_all] += -K
                    P[self.nis_all:2*self.nis_all, 0:self.nis_all] += -K
                    P[self.nis_all:2*self.nis_all, self.nis_all:2*self.nis_all] += K - (1 / (4 * self.gammas[1])) * np.identity(self.nis_all)
                    q = - (self.gammas[3] / (4 * self.gammas[2])) * np.ones(self.Ns_all_labeled) @ S_Lp @ M_L_concat
                assert(np.allclose(P, P.T))

                h = np.zeros(2 * self.nis_all)
                G = - np.identity(2 * self.nis_all)
                if self.version not in [4, 6, "PU"]:
                    b = np.ones(self.Ns_all - self.Ns_all_labeled) * self.gammas[3]
                else:
                    b = np.ones(self.Ns_all) * self.gammas[3]
                    if self.version in ["PU"]:
                        for i in range(len(is_PL_reorder)):
                            if is_PL_reorder[i] == 0:
                                b[i] = 0.
                                # b[i] = - self.gammas[3]

                ## put - to convert maximization to minimization problem. Multiply * 2 to match the format in https://cvxopt.org/userguide/coneprog.html#quadratic-programming
                if self.scale_factor == "auto":
                    self.scale_factor = 1. / np.mean(np.abs(P))
                P = self.scale_factor * utils.get_near_psd(-P * 2)
                if self.version not in [4, 6, "PU"] or (self.version in [6, "PU"] and g > 0):
                    q = -self.scale_factor * q
                else:
                    q = np.zeros(2 * self.nis_all)

                if False:
                    G = self.scale_factor * G
                    h = self.scale_factor * h
                    A = self.scale_factor * A
                    b = self.scale_factor * b
                if True:
                    P = matrix(P)
                    q = matrix(q)
                    G = matrix(G)
                    h = matrix(h)
                    A = matrix(A)
                    b = matrix(b)
                    cvxopt_solvers.options['show_progress'] = True
                    # cvxopt_solvers.options['abstol'] = 1e-10
                    # cvxopt_solvers.options['reltol'] = 1e-10
                    # cvxopt_solvers.options['feastol'] = 1e-10

                    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
                    if False:
                        lambdas = np.array(sol['x'])[:, 0]
                    else:
                        lambdas = np.ravel(sol['x'])
                    assert(abs(0.5 * lambdas.T @ np.array(P) @ lambdas + np.ravel(q) @ lambdas - sol['dual objective']) < abs(sol['dual objective'] * 1e-2))
                else:
                    lambdas = solve_qp(P, q, G, h, A, b, solver= "osqp") ## Set the `solver` keyword argument to one of the available solvers in ['cvxopt', 'ecos', 'osqp', 'quadprog']
                self.lambdas_list.append(lambdas)
                lambdas_epsilon = lambdas[0:self.nis_all]
                lambdas_delta = lambdas[self.nis_all:2*self.nis_all]
                # X_lambda_eps_minus_delta = self.X_concat @ (lambdas_epsilon - lambdas_delta)
                self.lambdas_epsilon_list.append(lambdas_epsilon)
                self.lambdas_delta_list.append(lambdas_delta)

                avg_response_train = {}
                for k in range(len(self.unique_labels)):
                    avg_response_train[self.unique_labels[k]] = np.mean(self.kernel(self.X_concat_train[:, self.nis_class_acc[k]:self.nis_class_acc[k+1]]) @ ((1 / self.N_factor) * (lambdas_epsilon - lambdas_delta)))

                ## Calculate h_k
                ## h_U for unlabeled X
                if self.version == 1:
                    h_L = (1 / (4 * self.gammas[2])) * S_L_inv @ S_LT_inv @ (- M_L @ (lambdas_epsilon - lambdas_delta) + self.gammas[3] * np.ones(self.Ns_all_labeled))
                h_U_k_sum = {}
                h_is_k = {label: [] for label in self.unique_labels}
                h_std_k = {}
                for k in range(len(self.unique_labels)):
                    h_U_k_sum_this = 0
                    assert(len(self.nis_class_acc_bag[k]) - 1 == self.Ns_class_acc[k + 1] - self.Ns_class_acc[k])
                    for i in range(len(self.nis_class_acc_bag[k]) - 1):
                        h_i = 0.
                        for j in range(self.nis_class_acc_bag[k][i], self.nis_class_acc_bag[k][i+1]):
                            h_i += (1. / self.N_factor) * XTX[j, :] @ (lambdas_epsilon - lambdas_delta)
                            epsilon = (1 / (2 * self.gammas[0])) * lambdas_epsilon[j]
                            delta = (1 / (2 * self.gammas[1])) * lambdas_delta[j]
                            if False:
                                if epsilon > delta:
                                    h_i += epsilon
                                else:
                                    h_i += -delta
                            else:
                                h_i += epsilon - delta
                        h_i = (1. / (self.nis_class_acc_bag[k][i+1] - self.nis_class_acc_bag[k][i])) * h_i
                        h_U_k_sum_this += h_i
                        h_is_k[self.unique_labels[k]].append(h_i)
                    h_U_k_sum[k] = h_U_k_sum_this
                    h_std_k[self.unique_labels[k]] = statistics.stdev(h_is_k[self.unique_labels[k]])
                
                bags_acc = 0
                h_k = {}
                h_k["unlabeled"] = (1. / self.Ns_all) * sum(list(h_U_k_sum.values()))
                for k in range(len(self.unique_labels)):
                    size_Ik = self.Ns_class_acc[k + 1] - self.Ns_class_acc[k]
                    if self.unique_labels[k] != "unlabeled":
                        if self.version == 0:
                            if True:
                                h_k[self.unique_labels[k]] =  (1 - self.gammas[4]) * np.mean(inv_S_k[k] @ const_base_k[k] @ (np.ones(size_Ik) * self.gammas[3] - M_k[k] @ (lambdas_epsilon - lambdas_delta)[self.nis_class_acc[k]:self.nis_class_acc[k + 1]])) + self.gammas[4] * (1. / (self.Ns_class_acc[k + 1] - self.Ns_class_acc[k])) * h_U_k_sum[k]
                            else:
                                h_k[self.unique_labels[k]] = h_U_k_sum[k] / (self.Ns_class_acc[k + 1] - self.Ns_class_acc[k])
                        elif self.version == 1:
                            h_k[self.unique_labels[k]] = (1. - self.gammas[4]) * np.mean(h_L[bags_acc: bags_acc + self.Ns_class_acc[k+1] - self.Ns_class_acc[k]]) + self.gammas[4] * (h_U_k_sum[k] / size_Ik if False else (1. / self.Ns_all) * sum(list(h_U_k_sum.values())))
                            bags_acc += self.Ns_class_acc[k+1] - self.Ns_class_acc[k]
                        else:
                            h_k[self.unique_labels[k]] = (1. / size_Ik) * h_U_k_sum[k]
                
                if False:
                    for k in range(len(self.unique_labels)):
                        print("Debugging..")
                        h_k[self.unique_labels[k]] = avg_response_train[self.unique_labels[k]]
                # h_k["unlabeled"] = (h_U + sum([(self.Ns_class_acc[k + 1] - self.Ns_class_acc[k]) * h_k[self.unique_labels[k]] for k in range(len(self.unique_labels)) if self.unique_labels[k] != "unlabeled"])) / self.Ns_all
                if self.version == 1: assert(bags_acc == self.Ns_all_labeled)

                self.avg_response_train_list.append(avg_response_train)
                self.h_is_k_list.append(h_is_k)
                self.h_std_k_list.append(h_std_k)
                self.h_k_list.append(h_k)
    
    def predict(self, X, y):
        y_pred = []
        responses= [[] for i in range(len(X))]
        anomaly_scores = [[] for i in range(len(X))]
        for g in range(self.num_planes):
            const_vec = (1 / self.N_factor) * (self.lambdas_epsilon_list[g] - self.lambdas_delta_list[g])
            for i in range(len(X)):
                response = np.mean(self.kernel(X[i]) @ const_vec)
                label = y[i] if y[i] in self.unique_labels else "unlabeled"
                threshold = self.h_k_list[g][label]
                if self.version not in [4, 6, "PU"]:
                    anomaly_scores[i].append(response - threshold)
                else:
                    anomaly_scores[i].append((response - threshold) / self.h_std_k_list[g][label])
                responses[i].append(response)
            if False:
                print(y_pred)
                # print(responses)
        
        anomaly_scores_mean = [np.mean(scores_G) for scores_G in anomaly_scores]
        responses_mean = [np.mean(response_G) for response_G in responses]
        for i in range(len(anomaly_scores_mean)):
            label = y[i] if y[i] in self.unique_labels else "unlabeled"
            if self.version not in [4, 6, "PU"]:
                y_pred.append(int(anomaly_scores_mean[i] > 0) * 2 - 1)
            else:
                y_pred.append(int(self.normal_int_ratio[0] < anomaly_scores_mean[i] < self.normal_int_ratio[1]) * 2 - 1)
        if self.thres is not None:
            y_pred = self.thres.eval([- score for score in responses_mean]) ## All threshold functions return a binary array where inliers and outliers are represented by a 0 and 1 respectively., https://github.com/KulikDM/pythresh#ocsvm -> may be wrong??
            # y_pred = np.where(y_pred == 1, -1, 1)
            y_pred = np.where(y_pred == 1, 1, -1)
        return y_pred
    
    def get_abnormality_response(self, X, y):
        abnormality_bags= []
        for i in range(len(X)):
            abnormality_instances = [[] for ii in range(X[i].shape[1])]
            for g in range(self.num_planes):
                responses_instances = self.kernel(X[i]) @ ((1 / self.N_factor) * (self.lambdas_epsilon_list[g] - self.lambdas_delta_list[g]))
                label = y[i] if y[i] in self.unique_labels else "unlabeled"
                threshold = self.h_k_list[g][label]
                for ii in range(responses_instances.shape[0]):
                    if self.version not in [4, 6, "PU"]:
                        abnormality_instances[ii].append(max(responses_instances[ii] - threshold, 0))
                    else:
                        abnormality_instances[ii].append(abs(responses_instances[ii] - threshold) / self.h_std_k_list[g][label])
            abnormality_bags.append([np.mean(abnorm_planes) for abnorm_planes in abnormality_instances])
        return abnormality_bags

    def plot_decision_boundary(self, X, ys_aux, ys_positive, class_to_color, circles_dicts = None, min_max_dict = None, is_upper = True, lines_lst = None, xlim_set= None, ylim_set= None, path_to_save = None):
        assert(X.shape[1] == ys_aux.shape[0] == ys_positive.shape[0])
        if_imshow = False

        cls_list = list(set(ys_aux.tolist()))
        cls_list = sorted(cls_list, key=lambda x: self.h_k_list[0][x])
        colors = np.array([class_to_color[ys_aux[i]] for i in range(ys_aux.shape[0])])
        plt.scatter(X[0, ys_positive == 1], X[1, ys_positive == 1], c=colors[ys_positive == 1], s=50, alpha=.5, marker= "o")
        plt.scatter(X[0, ys_positive == -1], X[1, ys_positive == -1], c=colors[ys_positive == -1], s=50, alpha=.5, marker= "^")
        ax = plt.gca()
        
        if circles_dicts is not None and not if_imshow:
            for circle_info in circles_dicts: 
                if circle_info is not None:
                    # circle_info = {"xy": , "radius": , "color": , }
                    e = Circle(**circle_info) ## https://stackoverflow.com/questions/4143502/how-to-do-a-scatter-plot-with-empty-circles-in-python
                    # ax.add_artist(e)
                    ax.add_patch(e)
        
        if lines_lst is not None and not if_imshow:
            for line in lines_lst:
                ax.add_line(line)
        
        if xlim_set is not None:
            ax.set_xlim(*xlim_set)
        if ylim_set is not None:
            ax.set_ylim(*ylim_set)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 120)
        yy = np.linspace(ylim[0], ylim[1], 120)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()])
        if min_max_dict is not None:
            xy = utils.min_max_scale(xy, min_max_dict = min_max_dict, if_reverse= False)

        Z = (self.kernel(xy) @ ((1 / self.N_factor) * (self.lambdas_epsilon_list[0] - self.lambdas_delta_list[0]))).T.reshape(XX.shape)
        # Z = self._decision_function(xy).reshape(XX.shape)

        # plot decision boundary and margins
        h_k_int = max(list(self.h_k_list[0].values())) - min(list(self.h_k_list[0].values()))
        colors = []
        levels = []
        for cls in cls_list:
            max_hi = self.h_k_list[0][cls] + self.normal_int_ratio[1] * self.h_std_k_list[0][cls]
            min_hi = self.h_k_list[0][cls] + self.normal_int_ratio[0] * self.h_std_k_list[0][cls]
            colors.append(class_to_color[cls])
            
            if True:
                if is_upper: levels.append(max_hi)
                else: levels.append(min_hi)
            else:
                colors.append(class_to_color[cls])
                levels.append(max_hi)
                levels.append(min_hi)
        
        levels_rank = np.argsort(levels)
        colors = [colors[i] for i in levels_rank]
        levels = [levels[i] for i in levels_rank]

        # colors = [class_to_color[cls] for cls in cls_list]
        # levels = [self.h_k_list[0][cls] for cls in cls_list]
        if False:
            # colors += ["green", "brown", "yellow"]
            # levels += [max(list(self.h_k_list[0].values())) + h_k_int * (0.2 * (i + 1)) for i in range(3)]
            colors = ["green", "brown", "yellow"] + colors
            levels = [min(list(self.h_k_list[0].values())) - h_k_int + h_k_int * (0.33 * i) for i in range(3)] + levels
        if not if_imshow:
            ax.contour(XX, YY, Z, colors= colors, levels= levels, alpha=0.5,
                    linestyles=['--' for cls in cls_list], linewidths=[2.0 for cls in cls_list])
        else:
            plt.imshow(Z, cmap= "viridis")
            plt.colorbar()

        # highlight the support vectors
        # ax.scatter(X[:, 0][self.alpha > 0.], X[:, 1][self.alpha > 0.], s=50, linewidth=1, facecolors='none', edgecolors='k')

        if not path_to_save:
            plt.show()
        else:
            plt.savefig(path_to_save)
        plt.clf()

