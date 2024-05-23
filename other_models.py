from sklearn.svm import OneClassSVM
import numpy as np
from kenchi.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class WrapOneClassSVM:

    def __init__(self, *args, **kwargs):
        self.OneClassSVM_inst = OneClassSVM(*args, **kwargs)
    
    def fit(self, X, y= None):
        X_concat = [x.T for x in X]
        X_concat = np.concatenate(X_concat)
        self.OneClassSVM_inst.fit(X_concat)

    def predict(self, X, y= None):
        y_pred = []

        for bi in range(len(X)):
            y_pred_e = np.mean(self.OneClassSVM_inst.predict(X[bi].T))
            y_pred_e = 1 if y_pred_e > 0 else -1
            y_pred.append(y_pred_e)
        
        return y_pred

class WrapKenchi:
    def __init__(self, base, *args, **kwargs):
        self.base_inst = base(*args, **kwargs)
        self.scaler = StandardScaler()
    
    def fit(self, X, y= None):
        X_concat = [x.T for x in X]
        X_concat = np.concatenate(X_concat)
        # self.base_inst._fit(X_concat)
        self.pipeline = make_pipeline(self.scaler, self.base_inst).fit(X_concat)
        self.threshold = self.base_inst._get_threshold()

    def predict(self, X, y= None):
        y_pred = []

        for bi in range(len(X)):
            anomaly_score_mean = np.mean(self.pipeline.anomaly_score(X[bi].T))
            y_pred_e = -1 if self.threshold < anomaly_score_mean else 1
            y_pred.append(y_pred_e)
        
        return y_pred

class WrapPyod:
    def __init__(self, base, *args, **kwargs):
        self.base_inst = base(*args, **kwargs)
    
    def fit(self, X, y= None):
        X_concat = [x.T for x in X]
        X_concat = np.concatenate(X_concat)
        self.base_inst.fit(X_concat)
    
    def predict(self, X, y= None):
        y_pred = []

        for bi in range(len(X)):
            anomaly_score_mean = np.mean(self.base_inst.decision_function(X[bi].T))
            y_pred_e = -1 if self.base_inst.threshold_ < anomaly_score_mean else 1
            y_pred.append(y_pred_e)
        
        return y_pred

class WrapPulearn:
    def __init__(self, base, *args, **kwargs):
        self.base_inst = base(*args, **kwargs)
    
    def fit(self, X, is_PL, y= None,):
        is_PL_loc = []
        for bi in range(len(X)):
            for ii in range(X[bi].shape[1]):
                is_PL_loc.append(-1 if is_PL[bi] == 0 else 1)
        X_concat = [x.T for x in X]
        X_concat = np.concatenate(X_concat)
        self.base_inst.fit(X_concat, np.array(is_PL_loc))
    
    def predict(self, X, y= None):
        y_pred = []

        for bi in range(len(X)):
            y_preds_mean = np.mean(self.base_inst.predict(X[bi].T))
            y_pred_e = 1 if y_preds_mean > 0.5 else -1
            y_pred.append(y_pred_e)
        
        return y_pred
