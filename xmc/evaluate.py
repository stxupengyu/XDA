import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.sparse import csr_matrix
import os



def evaluate_xml(targets, prediction, args):
    # print(prediction.shape, 'prediction.shape')
    # print(prediction, 'prediction')
    # print(targets.shape, 'targets.shape')
    # print(targets, 'targets')
    mlb = MultiLabelBinarizer(range(targets.shape[1]), sparse_output=True)
    mlb.fit(None)
    result = []
    result.append('P@k')
    for top_K in [1, 3, 5]:
        precision = get_precision(prediction, targets, top_K, mlb)
        result.append(precision)
    inv_w = get_inv_propensity(args)
    result.append('PSP@k')
    for top_K in [1, 3, 5]:
        psp = get_psp(prediction, targets, mlb, inv_w, top_K, args)
        result.append(psp)
    return result

def get_inv_propensity(args):
    a = 0.6
    b = 2.6
    train_labels = np.array(read_txt(os.path.join(args.data_dir, args.train_labels)))
    test_labels = np.array(read_txt(os.path.join(args.data_dir, args.test_labels)))
    train_labels = np.concatenate((train_labels, test_labels), axis=0)
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels)
    # print(mlb.classes_, 'mlb.classes_')
    # print(mlb.classes, 'mlb.classes')
    train_y = mlb.transform(train_labels)  # to one-hot form
    # print(train_labels.shape, 'train_labels.shape')
    # print(train_labels, 'train_labels')
    # print(train_y.shape, 'train_y.shape')
    # print(train_y, 'train_y')
    n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
    # print(number.shape, 'number.shape')
    # print(number, 'number')
    c = (np.log(n) - 1) * ((b + 1) ** a)
    # print('n', n)
    # print('number', number)
    # print(max(number), min(number))
    # print('c', c)
    return 1.0 + c * (number + b) ** (-a)

def get_psp(prediction, targets, mlb, inv_w, top_K, args):
    # print(prediction.shape, 'prediction.shape')
    # print(prediction, 'prediction')
    # print(targets.shape, 'targets.shape')
    # print(targets, 'targets')
    # print(inv_w.shape, 'inv_w.shape')
    # print(inv_w, 'inv_w')
    # print(mlb.transform(prediction[:, :top_K]).shape, 'mlb.transform(prediction[:, :top_K]).shape')
    # print(mlb.transform(prediction[:, :top_K]), 'mlb.transform(prediction[:, :top_K])')
    # print(mlb.transform(prediction[:, :top_K]).A.shape, 'mlb.transform(prediction[:, :top_K]).A.shape')
    # print(mlb.transform(prediction[:, :top_K]).A, 'mlb.transform(prediction[:, :top_K]).A')

    if not isinstance(targets, csr_matrix):
        # targets = mlb.transform(targets) #id to one-hot
        targets = csr_matrix(targets)
    prediction = mlb.transform(prediction[:, :top_K]).multiply(inv_w)
    num = prediction.multiply(targets).sum()
    t, den = csr_matrix(targets.multiply(inv_w)), 0
    for i in range(t.shape[0]):
        den += np.sum(np.sort(t.getrow(i).data)[-top_K:])
    # print('prediction', prediction.shape)
    # print('targets', targets.shape)
    # print('inv_w', inv_w.shape)
    # print('inv_w', inv_w)
    # print(max(inv_w), min(inv_w))
    # print('num', num)
    # print('den', den)
    # print('num', num)
    # print('den', den)
    return round(num / den * 100, 2)

def get_precision(prediction, targets, top_K, mlb):
    targets = sparse.csr_matrix(targets)
    prediction = mlb.transform(prediction[:, :top_K])
    precision = prediction.multiply(targets).sum() / (top_K * targets.shape[0])
    return round(precision * 100, 2)

def read_txt(txt_path):
    f = open(txt_path, 'r')
    lines = []
    count = 0
    for line in f.readlines():
        # process src and trg line
        lines.append(line.strip().split(' '))
    return lines