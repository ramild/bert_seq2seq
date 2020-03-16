import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd


def read_rcv1_ids(filepath):
    ids = set()
    with open(filepath) as f:
        new_doc = True
        for line in f:
            line_split = line.strip().split()
            if new_doc and len(line_split) == 2:
                tmp, did = line_split
                if tmp == ".I":
                    ids.add(int(did))
                    new_doc = False
                else:
                    print(line_split)
                    print("maybe error")
            elif len(line_split) == 0:
                new_doc = True
    print("{} samples in {}".format(len(ids), filepath))
    return ids


def read_rcv1(data_path):
    p2c = defaultdict(list)
    id2doc = defaultdict(lambda: defaultdict(list))
    nodes = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(data_path, "rcv1.topics.hier.orig.txt")) as f:
        for line in f:
            start = line.find("parent: ") + len("parent: ")
            end = line.find(" ", start)
            parent = line[start:end]
            start = line.find("child: ") + len("child: ")
            end = line.find(" ", start)
            child = line[start:end]
            start = line.find("child-description: ") + len(
                "child-description: ",
            )
            end = line.find("\n", start)
            child_desc = line[start:end]
            p2c[parent].append(child)
    for label in p2c:
        if label == "None":
            continue
        for children in p2c[label]:
            nodes[label]["children"].append(children)
            nodes[children]["parent"].append(label)

    train_id_set = read_rcv1_ids(
        os.path.join(data_path, "lyrl2004_tokens_train.dat"),
    )
    test_id_set = read_rcv1_ids(
        os.path.join(data_path, "lyrl2004_tokens_test_pt0.dat"),
    )
    test_id_set |= read_rcv1_ids(
        os.path.join(data_path, "lyrl2004_tokens_test_pt1.dat"),
    )
    test_id_set |= read_rcv1_ids(
        os.path.join(data_path, "lyrl2004_tokens_test_pt2.dat"),
    )
    test_id_set |= read_rcv1_ids(
        os.path.join(data_path, "lyrl2004_tokens_test_pt3.dat"),
    )
    np.random.seed(1331)

    with open(os.path.join(data_path, "rcv1-v2.topics.qrels")) as f:
        for line in f:
            cat, doc_id, _ = line.strip().split()
            if int(doc_id) in train_id_set or int(doc_id) in test_id_set:
                id2doc[int(doc_id)]["categories"].append(cat)

    train_ids = []
    test_ids = []
    rcv_df = pd.read_csv(
        os.path.join(data_path, "rcv1_v2_df.csv"),
        encoding="utf-8",
    )
    indexes_to_drop = []
    for i, row in rcv_df.iterrows():
        if not isinstance(row["text"], str):
            rcv_df.iloc[i]["text"] = "-"

    for i, row in rcv_df.iterrows():
        if row["id"] in train_id_set:
            train_ids.append(i)
        if row["id"] in test_id_set:
            test_ids.append(i)

    df_train = rcv_df.iloc[train_ids]
    df_test = rcv_df.iloc[test_ids]
    print()
    print("Objects:   ", len(rcv_df))
    print("Train size:", len(df_train))
    print("Test size: ", len(df_test))

    return rcv_df, df_train, df_test, dict(id2doc), dict(nodes)


if __name__ == "__main__":
    read_rcv1()
