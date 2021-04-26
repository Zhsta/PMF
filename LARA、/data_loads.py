import pandas as pd
import numpy as np

def get_train_data(start, end):
    train = np.array(pd.read_csv('train_data.csv', header=None))
    train_data = train
    real_user_emb = np.array(pd.read_csv('user_emb.csv',header=None))
    '''get train samples'''
    train_batch = train_data[start: end]
    user_batch = [line[0] for line in train_batch]
    item_batch = [line[1] for line in train_batch]
    attr_batch = [line[2][1:-1].split() for line in train_batch]
    real_user_emb_batch = real_user_emb[user_batch]
    return user_batch, item_batch, attr_batch, real_user_emb_batch

def get_neg_data(start, end):
    neg = np.array(pd.read_csv('neg_data.csv', header=None))
    neg_data = neg
    neg_batch = neg_data[start:end]
    real_user_emb = np.array(pd.read_csv('user_emb.csv',header=None))
    user_batch = [line[0] for line in neg_batch]
    item_batch = [line[1] for line in neg_batch]
    attr_batch = [line[2][1:-1].split() for line in neg_batch]
    real_user_emb_batch = real_user_emb[user_batch]
    return user_batch, item_batch, attr_batch, real_user_emb_batch

def get_test_data():
    test_item = np.array(pd.read_csv('test_item.csv', header=None).astype(np.int32))
    test_attribute = np.array(pd.read_csv('test_attribute.csv', header=None).astype(np.int32))
    return test_item, test_attribute


def RR(r, k):
    for i in range(k):
        if r[i] == 1:
            return 1.0 / (i + 1.0)
    return 0


def dcg_at_k(r, k, method=1):

    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):

    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def precision_at_k(r, k):

    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r):

    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):

    return np.mean([average_precision(r) for r in rs])



