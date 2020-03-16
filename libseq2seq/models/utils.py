import os
import csv
import codecs
import yaml
import time
import numpy as np

from sklearn import metrics


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):

    return AttrDict(yaml.load(open(path, "r")))


def read_datas(filename, trans_to_num=False):
    lines = open(filename, "r").readlines()
    lines = list(map(lambda x: x.split(), lines))
    if trans_to_num:
        lines = [list(map(int, line)) for line in lines]
    return lines


def save_datas(data, filename, trans_to_str=False):
    if trans_to_str:
        data = [list(map(str, line)) for line in data]
    lines = list(map(lambda x: " ".join(x), data))
    with open(filename, "w") as f:
        f.write("\n".join(lines))


def logging(file):
    def write_log(s):
        print(s, end="")
        with open(file, "a") as f:
            f.write(s)

    return write_log


def logging_csv(file):
    def write_csv(s):
        with open(file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(s)

    return write_csv


def format_time(t):
    return time.strftime("%Y-%m-%d-%H:%M:%S", t)


def eval_metrics(reference, candidate, params):
    def make_label(l):
        length = params["num_labels"] - 4
        result = np.zeros(length)
        indices = [label for label in l]
        result[indices] = 1
        return result

    def prepare_label(y_list, y_pre_list):
        reference = np.array([make_label(y) for y in y_list])
        candidate = np.array([make_label(y_pre) for y_pre in y_pre_list])
        return reference, candidate

    def get_metrics(y, y_pre):
        hamming_loss = metrics.hamming_loss(y, y_pre)
        macro_f1 = metrics.f1_score(y, y_pre, average="macro")
        macro_precision = metrics.precision_score(y, y_pre, average="macro")
        macro_recall = metrics.recall_score(y, y_pre, average="macro")
        micro_f1 = metrics.f1_score(y, y_pre, average="micro")
        micro_precision = metrics.precision_score(y, y_pre, average="micro")
        micro_recall = metrics.recall_score(y, y_pre, average="micro")
        return (
            hamming_loss,
            macro_f1,
            macro_precision,
            macro_recall,
            micro_f1,
            micro_precision,
            micro_recall,
        )

    def get_accuracy(reference, candidate):
        assert len(reference) == len(candidate)
        cnt_strict = 0
        cnt_soft = 0
        cnt_soft_1 = 0
        for i in range(len(reference)):
            ref = set([elem for elem in reference[i]])
            cand = set([elem for elem in candidate[i]])
            if ref == cand:
                cnt_strict += 1
            if len(ref - cand) == 0:
                cnt_soft += 1
            if len(ref - cand) == 0 and len(cand - ref) <= 1:
                cnt_soft_1 += 1
        return (
            cnt_strict * 1.0 / len(reference),
            cnt_soft * 1.0 / len(reference),
            cnt_soft_1 * 1.0 / len(reference),
        )

    y, y_pre = prepare_label(reference, candidate)
    # print(y[:3])
    # print(y_pre[:3])
    print()

    (
        hamming_loss,
        macro_f1,
        macro_precision,
        macro_recall,
        micro_f1,
        micro_precision,
        micro_recall,
    ) = get_metrics(y, y_pre)
    accuracy_strict, accuracy_soft, accuracy_soft_1 = get_accuracy(
        reference,
        candidate,
    )
    return {
        "hamming_loss": hamming_loss,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "accuracy_strict": accuracy_strict,
        "accuracy_soft": accuracy_soft,
        "accuracy_soft_1": accuracy_soft_1,
    }
