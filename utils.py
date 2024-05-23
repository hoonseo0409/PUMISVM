import numpy as np
import numpy.linalg as la
from random import random, shuffle, randint
import pharmpy
from tensorflow.keras.datasets import mnist
from sklearn.datasets import make_moons, make_blobs
from collections import OrderedDict
import mahotas
import cv2
from copy import deepcopy
from skimage.feature import hog
import itertools
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from kenchi.datasets import load_pima

import os, sys
from random import sample
from pandas import get_dummies

import glob
import imageio
import mahotas
from PIL import Image, ImageDraw

from sklearn.feature_extraction import image  # Used to extract patches from image

def get_points_inside_circle(center_xy, radius_max, num_points, radius_min = None):
    num_points = max(1, round(num_points))
    points = []
    num_points = round(num_points)
    for i in range(num_points):
        passed = False
        for j in range(1000):
            sampled_xy = []
            for axis in [0, 1]:
                sampled_xy.append(radius_max * (random() - 0.5) * 2)
            distance_sqed = sampled_xy[0] ** 2. + sampled_xy[1] ** 2.
            if distance_sqed <= radius_max ** 2. and (radius_min is None or distance_sqed >= radius_min ** 2.):
                points.append(np.array([center_xy[0] + sampled_xy[0], center_xy[1] + sampled_xy[1]]))
                passed = True
                break
        assert(passed)
    return np.array(points).T

        # radius_sampled = random() * radius
        # theta_sampled = random() * math.pi * 2.
        # points.append([center_xy[0] + radius_sampled * math.sin(theta_sampled),])

def get_points_region(region_funct, region_xyxy, num_points, max_tries = 1000):
    num_points = max(1, round(num_points))
    points = []
    for pi in range(num_points):
        for ti in range(max_tries):
            x, y = region_xyxy[0] + random() * (region_xyxy[2] - region_xyxy[0]), region_xyxy[1] + random() * (region_xyxy[3] - region_xyxy[1])
            if region_funct(x, y):
                points.append([x, y])
                break
    if len(points) != num_points:
        print(f"Requested number of points is {num_points}, but {len(points)} points have been generated.")
    return np.array(points).T

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        test1 = True
    except la.LinAlgError:
        test1 = False
    print(f"PD test 1 = {test1}")
    return test1
    eigval, eigvec = np.linalg.eig(B)
    test2 = np.all(eigval.real >= 0)
    print(f"PD test 2 = {test2}")
    return test1 and test2
    

def get_near_psd(M, niter = 1):
    assert(np.allclose(M, M.T))
    # M = (M + M.T) / 2.
    if isPD(M):
        return M
    else:
        print(f"WARNING: Given matrix is not positive semi-definite, so the objective may not be convex.")
        # return M
    if True:
        
        for it in range(niter):
            eigval, eigvec = np.linalg.eig(M)
            if np.all(eigval.real >= 0):
                return M
            
            eigval = eigval.real
            eigvec = eigvec.real
            eigval[eigval < 0] = 0

            if False:
                eigval_filtered = []
                for i in range(eigval.shape[0]):
                    eigval_this = eigval[i].real
                    if eigval_this < 0: eigval_this = 0
                    eigval_filtered.append(eigval_this)
                eigval_filtered = np.array(eigval_filtered)

            M = eigvec.dot(np.diag(eigval)).dot(eigvec.T)
            assert(np.allclose(M, M.T))
            # M = (M + M.T) / 2.
        return M
    elif True:
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """
        A = M

        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if isPD(A3):
            return A3

        spacing = np.spacing(la.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3

def min_max_scale(bag, min_max_dict, if_reverse = False):
    bag_scaled = deepcopy(bag)
    for axis in range(bag_scaled.shape[0]):
        if if_reverse:
            if False:
                bag_scaled[axis, :] = (bag_scaled[axis, :] / 2) + 0.5
            bag_scaled[axis, :] = bag_scaled[axis, :] * (min_max_dict[axis]["max"] - min_max_dict[axis]["min"]) + min_max_dict[axis]["min"]
        else:
            bag_scaled[axis, :] = (bag_scaled[axis, :] - min_max_dict[axis]["min"]) / (min_max_dict[axis]["max"] - min_max_dict[axis]["min"])
            if False:
                bag_scaled[axis, :] = (bag_scaled[axis, :] - 0.5) * 2
    return bag_scaled

def normalize_similarity_matrix(S):
    pass

def dict_to_list(d):
    return [d[k] for k in sorted(d)] ## https://stackoverflow.com/questions/22856319/convert-dictionary-to-list-of-values-sorted-by-keys

def filter_cls(cls_list, cls_limit = 10):
    count_dict = {}
    cls_list = deepcopy(cls_list)
    cls_minor = []
    for cls in cls_list:
        if cls not in count_dict.keys():
            count_dict[cls] = 1
        else:
            count_dict[cls] += 1
    for cls in count_dict.keys():
        if count_dict[cls] <= cls_limit:
            cls_minor.append(cls)
    for i in range(len(cls_list)):
        if cls_list[i] in cls_minor:
            cls_list[i] = "unlabeled"
    return cls_list

def min_max_scale_bags(bags, minv = -1., maxv = 1.):
    bags_idc_acc = [0]
    for bag in bags:
        bags_idc_acc.append(bags_idc_acc[-1] + bag.shape[1])
    bags_concat = np.concatenate(bags, axis= 1)
    bags_scaled = []
    for fi in range(bags_concat.shape[0]):
        maxf, minf = np.max(bags_concat[fi, :]), np.min(bags_concat[fi, :])
        if maxf > minf:
            bags_concat[fi, :] = ((bags_concat[fi, :] - minf) / (maxf - minf)) * (maxv - minv) + minv
        else:
            bags_concat[fi, :] = (maxv - minv) / 2.
    for bi in range(len(bags)):
        bags_scaled.append(bags_concat[:, bags_idc_acc[bi] : bags_idc_acc[bi + 1]])
    assert(np.allclose(bags_scaled[-1][:, -1], bags_concat[:, -1]))
    return bags_scaled

def chest_dataset_output(bags, target, image_path, x, y, M_cut_idx, bags_M, similarities, cls_bags):
    def cls_bag_filter(elem):
        if False: ## RT_PCR_positive
            if np.all(np.isnan(elem)):
                return "unlabeled"
            else:
                for i in range(len(elem)):
                    elem_2 = elem[len(elem) - (i + 1)]
                    if not np.isnan(elem_2):
                        if elem_2 == 1.:
                            return "Y"
                        elif elem_2 == 0.:
                            return "N"
                        else:
                            raise Exception(NotImplementedError)
        else:
            if np.all([elem_2 in ["Unknown", "todo"] for elem_2 in elem]):
                return "unlabeled"
            else:
                for i in range(len(elem)):
                    elem_2 = elem[len(elem) - (i + 1)]
                    if elem_2 not in ["Unknown", "todo"]:
                        return elem_2
    cls_bags = filter_cls([cls_bag_filter(elem) for elem in cls_bags])
    bags = [np.stack(bag).T for bag in bags]
    is_normal = [not any([tee == "Y" for tee in te]) for te in target]
    is_normal = [1 if is_normal[i] else -1 for i in range(len(is_normal))]
    return min_max_scale_bags(bags), cls_bags, is_normal

def train_test_split_wrap(is_normal, train_norm_proportion = 0.8, int_of_target = 1):
    splits = []
    positive_idcs = []
    negative_idcs = []
    for i in range(len(is_normal)):
        if is_normal[i] == int_of_target:
            positive_idcs.append(i)
        else:
            negative_idcs.append(i)
    shuffle(positive_idcs)
    shuffle(negative_idcs)

    for si in range(int(1 // (1. - train_norm_proportion))):
        positive_idcs_train, positive_idcs_test = train_test_split(positive_idcs, test_size= 1. - train_norm_proportion)
        splits.append({"train": positive_idcs_train, "test": positive_idcs_test + negative_idcs})
    return splits
                    
def plot_images(imgs_arr_list, selected_patches, path_to_save, color_not_selected_instances = None, color_rotation = None, subtitles= None):
    ## Among the positions of patches in (x_pos and y_pos), we plot the patches of selected_patches with the special colors specified by patches.

    num_identified_patches = len(selected_patches) if isinstance(selected_patches, list) else selected_patches.shape[0]
    ## Default Params
    if color_rotation is None: color_rotation = ["r", "g", "b"]
    idx_color_dict = {selected_patches[rank]: color_rotation[(len(color_rotation) + rank) % len(color_rotation)] for rank in range(num_identified_patches)}

    ## Plot the patches
    # selected_patches = selected_patches[0] ## selected_patches is list of just a single index
    num_patches = len(imgs_arr_list)
    # print(f"imgs_path: {imgs_path}, selected_patches: {selected_patches}, path_to_save: {path_to_save}, x_pos: {x_pos}, y_pos: {y_pos}, width_height_patches: {width_height_patches}")
    fig, ax = plt.subplots(ncols= num_patches)

    ## Plot patch in the bag
    for i in range(num_patches):
        if num_patches > 1:
            current_ax = ax[i]
        else:
            current_ax = ax
        img_arr_patch = imgs_arr_list[i]
        # print(f"x_pos: {x_pos}, y_pos: {y_pos}, width_height_patch: {width_height_patch}, x_pos[i]: {x_pos[i]}, (x_pos[i] + width_height_patch[0]): {(x_pos[i] + width_height_patch[0])}")

        # Display the image
        current_ax.imshow(img_arr_patch, cmap = 'gray')
        if subtitles is not None: current_ax.title.set_text(subtitles[i]) #  fontdict= dict(fontsize= 10)
        current_ax.set_xticks([])
        current_ax.set_yticks([])
        current_ax.axis("off")
        
        if i in selected_patches:
            ## Create a Rectangle patch
            rect = patches.Rectangle((0,0), img_arr_patch.shape[1] - 1, img_arr_patch.shape[0] - 1, linewidth= 2, edgecolor= idx_color_dict[i], facecolor='none') ## Input array is transposed.

            # Add the patch to the Axes
            current_ax.add_patch(rect)
        else:
            if color_not_selected_instances is not None:
                ## Create a Rectangle patch
                rect = patches.Rectangle((0,0), img_arr_patch.shape[1] - 1, img_arr_patch.shape[0] - 1, linewidth= 2, edgecolor= color_not_selected_instances, facecolor='none') ## Input array is transposed.
                # Add the patch to the Axes
                current_ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(path_to_save, bbox_inches='tight', pad_inches=0) ## .pdf omitted from path_to_save.

def plot_patches(imgs_path, selected_patches, path_to_save, x_pos, y_pos, width_height_patches, color_not_selected_instances = None, color_rotation = None):
    ## Among the positions of patches in (x_pos and y_pos), we plot the patches of selected_patches with the special colors specified by patches.

    num_identified_patches = len(selected_patches) if isinstance(selected_patches, list) else selected_patches.shape[0]
    ## Default Params
    if color_rotation is None: color_rotation = ["r", "g", "b"]
    idx_color_dict = {selected_patches[rank]: color_rotation[(len(color_rotation) + rank) % len(color_rotation)] for rank in range(num_identified_patches)}

    ## Plot the patches
    # selected_patches = selected_patches[0] ## selected_patches is list of just a single index
    num_patches = len(x_pos)
    assert(len(x_pos) == len(y_pos))
    # print(f"imgs_path: {imgs_path}, selected_patches: {selected_patches}, path_to_save: {path_to_save}, x_pos: {x_pos}, y_pos: {y_pos}, width_height_patches: {width_height_patches}")
    img_arr_whole = np.transpose(plt.imread(imgs_path), (1, 0, 2)) ## In matplotlib, width goes to y axis somehow..
    fig, ax = plt.subplots(ncols= num_patches)

    ## Plot patch in the bag
    for i in range(num_patches):
        width_height_patch = width_height_patches[i][0]
        if num_patches > 1:
            current_ax = ax[i]
        else:
            current_ax = ax
        img_arr_patch = img_arr_whole[x_pos[i]:(x_pos[i] + width_height_patch[0]), y_pos[i]:(y_pos[i] + width_height_patch[1])]
        # print(f"x_pos: {x_pos}, y_pos: {y_pos}, width_height_patch: {width_height_patch}, x_pos[i]: {x_pos[i]}, (x_pos[i] + width_height_patch[0]): {(x_pos[i] + width_height_patch[0])}")

        # Display the image
        current_ax.imshow(img_arr_patch, cmap = 'gray')
        current_ax.set_xticks([])
        current_ax.set_yticks([])
        current_ax.axis("off")
        
        if i in selected_patches:
            ## Create a Rectangle patch
            rect = patches.Rectangle((0,0), width_height_patch[1] - 1, width_height_patch[0] - 1, linewidth= 2, edgecolor= idx_color_dict[i], facecolor='none') ## Input array is transposed.

            # Add the patch to the Axes
            current_ax.add_patch(rect)
        else:
            if color_not_selected_instances is not None:
                ## Create a Rectangle patch
                rect = patches.Rectangle((0,0), width_height_patch[1] - 1, width_height_patch[0] - 1, linewidth= 2, edgecolor= color_not_selected_instances, facecolor='none') ## Input array is transposed.
                # Add the patch to the Axes
                current_ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(path_to_save + "_patches.pdf", bbox_inches='tight', pad_inches=0) ## .pdf omitted from path_to_save.

    ## Plot patch in the global img
    fig, current_ax = plt.subplots(ncols= 1)
    current_ax.imshow(img_arr_whole, cmap = 'gray')
    current_ax.set_xticks([])
    current_ax.set_yticks([])
    current_ax.axis("off")
    for i in selected_patches:
        # Create a Rectangle patch
        rect = patches.Rectangle((y_pos[i], x_pos[i]), width_height_patches[i][0][1], width_height_patches[i][0][0], linewidth= 2, edgecolor= idx_color_dict[i], facecolor='none')
        # Add the patch to the Axes
        current_ax.add_patch(rect)
    if color_not_selected_instances is not None:
        for i in [j for j in range(num_patches) if j not in selected_patches]:
            # Create a Rectangle patch
            rect = patches.Rectangle((y_pos[i], x_pos[i]), width_height_patches[i][0][1], width_height_patches[i][0][0], linewidth= 2, edgecolor= color_not_selected_instances, facecolor='none')
            # Add the patch to the Axes
            current_ax.add_patch(rect)
    plt.axis('off')
    plt.savefig(path_to_save + "_whole.pdf", bbox_inches='tight', pad_inches=0) ## .pdf omitted from path_to_save.

def getExecPath():
    '''
        ref: https://stackoverflow.com/questions/606561/how-to-get-filename-of-the-main-module-in-python
    '''
    try:
        sFile = os.path.abspath(sys.modules['__main__'].__file__)
    except:
        sFile = sys.executable
    return os.path.dirname(sFile)

def getNewDirectoryName(parentDir, newDir, root_dir = None):
    '''
        To get new directory name to save results while avoiding duplication
    '''

    if root_dir is None:
        root_dir = getExecPath()
    if parentDir[0] != '/':
        parentDir = '/' + parentDir
    if parentDir[-1] != '/':
        parentDir = parentDir + '/'

    assert(root_dir + parentDir)

    duplicatedNameNum = 0
    while(os.path.isdir(root_dir + parentDir + newDir + str(duplicatedNameNum)) and duplicatedNameNum < 1000):
        duplicatedNameNum = duplicatedNameNum + 1
    newDir = newDir + str(duplicatedNameNum)

    return newDir

def load_synthetic_mi(class_radius, class_abnormal_threshold, num_points_factor = 1.0):
    aux_classes = class_radius.keys()
    for dct in [class_abnormal_threshold]:
        for cls in aux_classes:
            assert(cls in dct.keys())

    bags, cls_bags, is_normal, cls_bags_known = [], [], [], []

    circles_dicts = []
    for cls in aux_classes:
        ## Labeled Normal, Unlabeled Normal, Labeled Abnormal, Unlabeled Abnormal Bags
        for normality, bool_labeled, centers_of_bags in zip([1, 1, -1, -1], [True, False, True, False], [
            get_points_inside_circle(center_xy= [0., 0.], radius_max= class_abnormal_threshold[cls] + 3, num_points= 150, radius_min= class_abnormal_threshold[cls]),
            get_points_inside_circle(center_xy= [0., 0.], radius_max= class_abnormal_threshold[cls] + 3, num_points= 100, radius_min= class_abnormal_threshold[cls]),
            get_points_inside_circle(center_xy= [0., 0.], radius_max= class_abnormal_threshold[cls], num_points= 50, radius_min= 0),
            get_points_inside_circle(center_xy= [0., 0.], radius_max= class_abnormal_threshold[cls], num_points= 20, radius_min= 0)
            ]):
            for i in range(centers_of_bags.shape[1]):
                bags.append(get_points_inside_circle(center_xy= [centers_of_bags[0, i], centers_of_bags[1, i]], radius_max= class_radius[cls], num_points= sample(list(range(3, 7)), 1)[0] * num_points_factor))
                cls_bags_known.append(cls)
                if bool_labeled:
                    cls_bags.append(cls)
                else:
                    cls_bags.append("unlabeled")
                is_normal.append(normality)
                circles_dicts.append(dict(xy= [centers_of_bags[0, i], centers_of_bags[1, i]], radius= class_radius[cls], edgecolor= None, fill= False, linestyle= (0, (1, 1)), linewidth= None))

    min_max_dict = [{"min": 1e+8, "max": -1e+8} for i in range(2)]
    for i in range(len(bags)):
        for axis in [0, 1]:
            if min_max_dict[axis]["max"] < np.max(bags[i][axis, :]):
                min_max_dict[axis]["max"] = np.max(bags[i][axis, :])
            if min_max_dict[axis]["min"] > np.min(bags[i][axis, :]):
                min_max_dict[axis]["min"] = np.min(bags[i][axis, :])        

    for i in range(len(bags)):
        bags[i] = min_max_scale(bags[i], min_max_dict= min_max_dict, if_reverse= False)
    
    return bags, cls_bags, is_normal, circles_dicts, min_max_dict, cls_bags_known
