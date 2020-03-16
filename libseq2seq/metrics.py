import numpy as np
from sklearn import metrics


def accuracy(true_labels, prob_preds):
    pred_labels = np.argmax(prob_preds, axis=1)
    return metrics.accuracy_score(true_labels, pred_labels)


def log_loss(true_labels, prob_preds):
    prob_preds = np.array(prob_preds)
    return metrics.log_loss(
        true_labels,
        prob_preds,
        labels=np.arange(prob_preds.shape[1]),
    )


def precision(true_labels, prob_preds, label):
    pred_labels = np.argmax(prob_preds, axis=1)
    return (
        sum((pred_labels == label) & (true_labels == label))
        * 1.0
        / sum(pred_labels == label)
    )


def recall(true_labels, prob_preds, label):
    pred_labels = np.argmax(prob_preds, axis=1)
    return (
        sum((pred_labels == label) & (true_labels == label))
        * 1.0
        / sum(true_labels == label)
    )


def f1_micro(true_labels, prob_preds):
    pred_labels = np.argmax(prob_preds, axis=1)
    return metrics.f1_score(true_labels, pred_labels, average="micro")


def f1_macro(true_labels, prob_preds):
    pred_labels = np.argmax(prob_preds, axis=1)
    return metrics.f1_score(true_labels, pred_labels, average="macro")


def auc_micro(true_labels, prob_preds):
    n_samples, n_labels = len(prob_preds), len(prob_preds[0])
    true_labels_ohe = np.zeros((n_samples, n_labels), dtype=int)
    for i, label in enumerate(true_labels):
        true_labels_ohe[i][label] = 1
    return metrics.roc_auc_score(true_labels_ohe, prob_preds, average="micro")


def auc_macro(true_labels, prob_preds):
    n_samples, n_labels = len(prob_preds), len(prob_preds[0])
    true_labels_ohe = np.zeros((n_samples, n_labels), dtype=int)
    for i, label in enumerate(true_labels):
        true_labels_ohe[i][label] = 1
    return metrics.roc_auc_score(true_labels_ohe, prob_preds, average="macro")
