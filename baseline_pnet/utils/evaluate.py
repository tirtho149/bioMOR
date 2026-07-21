import logging

# from lifelines.utils import concordance_index
from sklearn import metrics
from sklearn.metrics import accuracy_score


def evalualte(y_test, y_pred, y_pred_score=None):
    accuracy = accuracy_score(y_test, y_pred)
    # score_train = accuracy_score(y_train, y_pred_train)
    if y_pred_score is None:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # auc2 = metrics.roc_auc_score(y_test, y_pred_score)
    f1 = metrics.f1_score(y_test, y_pred)
    f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    percision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    logging.info(metrics.classification_report(y_test, y_pred))
    from sklearn.metrics import average_precision_score
    aupr = average_precision_score(y_test, y_pred_score)
    logging.info(
        '--accuracy: {0:.2f} precision: {1:.2f} auc: {2:.2f} f1: {3:.2f} f1_macro: {4:.2f} aupr {5:.2f}'.format(
            accuracy, percision, auc, f1, f1_macro, aupr))
    score = {}
    score['accuracy'] = accuracy
    score['precision'] = percision
    score['auc'] = auc
    score['f1'] = f1
    score['f1_macro'] = f1_macro
    score['aupr'] = aupr
    score['recall'] = recall
    # logging.info(score)
    # score['aupr'] = aupr
    return score


def evalualte_classification_binary(y_test, y_pred, y_pred_score=None):
    print(y_test.shape, y_pred.shape)
    accuracy = accuracy_score(y_test, y_pred)
    if y_pred_score is None:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    f1 = metrics.f1_score(y_test, y_pred)
    f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    logging.info(metrics.classification_report(y_test, y_pred))
    from sklearn.metrics import average_precision_score
    aupr = average_precision_score(y_test, y_pred_score)
    score = {}
    score['accuracy'] = accuracy
    score['precision'] = precision
    score['auc'] = auc
    score['f1'] = f1
    score['f1_macro'] = f1_macro
    score['aupr'] = aupr
    score['recall'] = recall
    return score


def evalualte_classification_multiclass(y_test, y_pred, y_pred_score=None):
    """Multi-class classification evaluator.

    y_test       : (N,) int class labels
    y_pred       : (N,) int predicted class (argmax)
    y_pred_score : (N, K) softmax probabilities

    Reports macro-averaged AUC/AUPR (OvR) and macro/weighted F1 on top of
    accuracy/precision/recall. Same metric KEYS as the binary evaluator
    so the downstream CV pipeline + post-processor work unchanged.
    """
    import numpy as np
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)

    y_test = np.asarray(y_test).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    n_classes = int(max(y_test.max(), y_pred.max(),
                        y_pred_score.shape[1] - 1 if y_pred_score is not None else 0)) + 1

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    auc = float('nan')
    aupr = float('nan')
    if y_pred_score is not None:
        y_pred_score = np.asarray(y_pred_score)
        # one-hot the labels for OvR AUPR
        y_oh = np.zeros((len(y_test), n_classes), dtype=int)
        y_oh[np.arange(len(y_test)), y_test] = 1
        classes_present = np.unique(y_test)
        try:
            if len(classes_present) > 1:
                auc = roc_auc_score(y_test, y_pred_score,
                                    multi_class='ovr', average='macro',
                                    labels=list(range(n_classes)))
            aupr = average_precision_score(y_oh, y_pred_score, average='macro')
        except Exception as e:
            logging.warning("multiclass AUC/AUPR computation failed: %s", e)

    logging.info(metrics.classification_report(y_test, y_pred, zero_division=0))
    score = {
        'accuracy':  accuracy,
        'precision': precision,
        'auc':       auc,
        'f1':        f1_weighted,   # populated so downstream code stays happy
        'f1_macro':  f1_macro,
        'aupr':      aupr,
        'recall':    recall,
    }
    return score


def evalualte_regression(y_true, y_pred, **kwargs):
    var = metrics.explained_variance_score(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    score = {}
    score['explained variance'] = var
    score['precision'] = r2
    score['median_absolute_error'] = median_absolute_error
    score['mean_squared_log_error'] = mean_squared_log_error
    score['mean_squared_error'] = mean_squared_error
    score['mean_absolute_error'] = mean_absolute_error
    return score


def evalualte_survival(y_true, y_pred, **kwargs):
    e = y_true['event']
    t = y_true['time']
    partial_hazards = y_pred
    c_index = concordance_index(t, partial_hazards, e)
    # score={}
    # score['c_index variance'] = c_index
    return c_index


# custom R2-score metrics for keras backend
from tensorflow.keras import backend as K


def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


# def rank_corr(y_true, y_pred):
#     print (y_true)
#     print (y_pred)
#     ret =spearmanr(y_true, y_pred)[0]
#     print(ret)
#     return ret
import tensorflow as tf


def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)
