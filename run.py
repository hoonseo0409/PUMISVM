import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib

import PUMISVM
import utils
from sklearn import metrics
import numpy as np
import cv2
import os
import json
import misvm
from sklearn.svm import OneClassSVM
import other_models
from random import sample
from copy import deepcopy
import kenchi.outlier_detection
from kenchi.datasets import load_pima, load_pendigits, load_wdbc, load_wilt
import pythresh.thresholds.filter

if True:
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

def ours_data_process(X, y, phase):
    return {"X": X, "y": y}

def misvm_data_process(X, y, phase):
    output = {}
    output["bags"] = [x.T for x in X]
    if phase == "train":
        output["y"] = [1. for ye in y]
    return output

int_of_target = 1
models_dict = {"ours": {"model": PUMISVM.PUMISVM, "data_process": ours_data_process, "init_kwargs": dict(kernel_method= "rbf", kernel_kwargs= {"sigma": 1.5}, normal_int_ratio= [-1.0, 1.0], thres= None, version= 6, 
                                                                                        PU_kwargs= {"iters": 1, "d_z": 5, "mus_list": [1.], "delta": 1e-8, "rho": 1.1,    "original_data_kernel": dict(kernel_method= "rbf", kernel_kwargs= {"sigma": 1.5})})}}

if True:
    from kenchi.outlier_detection import OCSVM, FastABOD, PCA, KNN, LOF, MiniBatchKMeans, KDE
    models_dict["OCSISVM"] = {"model": other_models.WrapOneClassSVM, "data_process": ours_data_process, "init_kwargs": dict()}

    # FastABOD(novelty=True, n_jobs=-1), OCSVM(), MiniBatchKMeans(), LOF(novelty=True, n_jobs=-1), KNN(novelty=True, n_jobs=-1), PCA(), KDE()
    for name, kwargs in zip(["OCSVM", "LOF"], [dict(base= OCSVM), dict(base= LOF, novelty=True, n_jobs=-1)]):
        models_dict[name] = {"model": other_models.WrapKenchi, "data_process": ours_data_process, "init_kwargs": kwargs}
if True:
    from pyod.models.lunar import LUNAR
    from pyod.models.deep_svdd import DeepSVDD
    for name, kwargs in zip(["LUNAR", "DeepSVDD"], [dict(base= LUNAR), dict(base= DeepSVDD)]):
        models_dict[name] = {"model": other_models.WrapPyod, "data_process": ours_data_process, "init_kwargs": kwargs}
if True:
    import pulearn
    from sklearn.svm import SVC
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    for name, kwargs in zip(["Elkanoto", "BaggingPuClassifier"], [dict(base= pulearn.WeightedElkanotoPuClassifier, estimator=svc, labeled=10, unlabeled=20, hold_out_ratio=0.2), dict(base= pulearn.BaggingPuClassifier, base_estimator=svc, n_estimators=15)]):
        models_dict[name] = {"model": other_models.WrapPulearn, "data_process": ours_data_process, "init_kwargs": kwargs, "if_PU": True}

results_dict = {model: {} for model in models_dict.keys()}
dataset_kind = "synthetic_mi" ## 

if dataset_kind == "chest-xray":
    models_dict["ours"]["init_kwargs"]["normal_int_ratio"] = [-1.2, 1.2]
    models_dict["ours"]["init_kwargs"]["kernel_kwargs"]["sigma"] = 1.2
    dataset_raw =utils.load_chest(chest_path= "path/to/chest_dataset")
    image_path = dataset_raw[2]
    dataset = utils.chest_dataset_output(*dataset_raw)
    bags, cls_bags, is_normal = dataset[0], dataset[1], dataset[2]
else:
    # models_dict["ours"]["init_kwargs"]["normal_int_ratio"] = [-1.6, 1.6]
    class_radius = {"A": 1.5, "B": 3}
    class_abnormal_threshold = {"A": 4, "B": 14}
    class_to_color = {"A": "green", "B": "brown"}
    lines_gt, xlim_set, ylim_set = None, None, None

    circles_dicts_class = [dict(xy= [0, 0], radius= class_abnormal_threshold[cls], edgecolor= class_to_color[cls], fill= False, linestyle= "-", linewidth= None) for cls in class_abnormal_threshold.keys()]

    bags, cls_bags, is_normal, circles_dicts, min_max_dict, cls_bags_known = utils.load_synthetic_mi(class_radius, class_abnormal_threshold, num_points_factor = 1.0)
splits = utils.train_test_split_wrap(is_normal, int_of_target= int_of_target)

output_path = "/Users/projects/python/PUMISVM/output"
folder_name = utils.getNewDirectoryName(output_path + "/", f"{dataset_kind}_", root_dir= "")
output_path = output_path + f"/{folder_name}"
os.mkdir(output_path)

for si in range(len(splits)):
    split = splits[si]
    for model_name in models_dict.keys():
        print(f"Prediction with {model_name}.")
        bags_loc = bags
        model_inst = models_dict[model_name]["model"](**models_dict[model_name]["init_kwargs"])
        if_PU = False if "if_PU" not in models_dict[model_name] else models_dict[model_name]["if_PU"]
        if not if_PU:
            model_inst.fit(**models_dict[model_name]["data_process"](X= [bags_loc[i] for i in split["train"]], y= [cls_bags[i] for i in split["train"]], phase= "train"))
        else:
            split_indices = split["train"] + split["test"]
            model_inst.fit(**models_dict[model_name]["data_process"](X= [bags_loc[i] for i in split_indices], y= [cls_bags[i] for i in split_indices], phase= "train"), is_PL = [(1 if is_normal[i] == int_of_target and i in split["train"] else 0) for i in split_indices])

        if model_name.startswith("ours"):
            print(f"h_k is {model_inst.h_k_list}.")
            print(f"Response average of training set is {model_inst.avg_response_train_list}.")
        y_pred = model_inst.predict(**models_dict[model_name]["data_process"](X= [bags_loc[i] for i in split["test"]], y= [cls_bags[i] for i in split["test"]], phase= "test"))

        y_true = []
        for i in split["test"]:
            if is_normal[i] == int_of_target:
                y_true.append(1)
            else:
                y_true.append(-1)
        
        if True:
            idcs = sample(range(len(y_true)), min(len(y_true), 10))
            print(f"(true, pred) = {[(y_true[i], y_pred[i]) for i in idcs]}.")

        results_dict[model_name][si] = {}
        results_dict[model_name][si]["accuracy"] = metrics.accuracy_score(y_pred, y_true)
        results_dict[model_name][si]["precision"] = metrics.precision_score(y_pred, y_true)
        results_dict[model_name][si]["recall"] = metrics.recall_score(y_pred, y_true)
        results_dict[model_name][si]["f1"] = metrics.f1_score(y_pred, y_true)
        results_dict[model_name][si]["bacc"] = metrics.balanced_accuracy_score(y_pred, y_true)
        for metric in ["precision", "recall", "f1", "accuracy", "bacc"]:
            print(f"{metric}: {results_dict[model_name][si][metric]}.")    

        if model_name.startswith("ours"):
            for pi in range(3):
                num_plots = min(10, len(split["test"]))
                indices_in_splits= sample(list(range(len(split["test"]))), k= num_plots)
                indices_in_bags = [split["test"][i] for i in indices_in_splits]
                circles_dicts_this_split = [deepcopy(circles_dicts[i]) for i in indices_in_bags]
                X_plot, ys_aux, ys_positive = [], [], []
                for i in range(num_plots):
                    circles_dicts_this_split[i]["edgecolor"] = "blue" if y_pred[indices_in_splits[i]] == 1 else "red"
                    X_plot.append(deepcopy(utils.min_max_scale(bags[indices_in_bags[i]], min_max_dict, if_reverse= True)))
                    for j in range(bags[indices_in_bags[i]].shape[1]):
                        ys_aux.append(cls_bags_known[indices_in_bags[i]])
                        ys_positive.append((1 if is_normal[indices_in_bags[i]] == int_of_target else -1))
                
                circles_dicts_this_split = circles_dicts_this_split + deepcopy(circles_dicts_class)
                model_inst.plot_decision_boundary(np.concatenate(X_plot, axis= 1), np.array(ys_aux), np.array(ys_positive), class_to_color, circles_dicts = circles_dicts_this_split, min_max_dict= min_max_dict, lines_lst= lines_gt, xlim_set= xlim_set, ylim_set= ylim_set, path_to_save= f"{output_path}/points_plot_{model_name}_{si}_{pi}.pdf")

with open(f'{output_path}/scores.txt', 'w') as convert_file:
     convert_file.write(json.dumps(results_dict))
    
print(f"Finished, results are saved in {output_path}")