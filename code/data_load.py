import numpy as np
from sklearn.metrics import roc_curve,average_precision_score,precision_recall_curve,precision_score, recall_score, f1_score
from sklearn.metrics import auc
from sklearn import metrics


def load_data(rna_dis_adj, k_folds):
    test_pos_list = list()
    test_pos_idx = np.array(np.where(rna_dis_adj == 1))
    test_neg_idx = np.array(np.where(rna_dis_adj == 0))
    rng = np.random.default_rng()
    rng.shuffle(test_pos_idx, axis=1)

    for test_pos in np.array_split(test_pos_idx, k_folds, axis=1):
        num_of_samples = len(test_pos[0])

        rng.shuffle(test_neg_idx, axis=1)
        test_neg = test_neg_idx[:, :num_of_samples]
        test_pos_list.append(np.hstack((test_pos, test_neg)))
    return test_pos_list


def normalize_mat(mat):
    assert mat.size != 0, f"Calculating normalized matrix need a non-zero square matrix. matrix size:{mat.shape}"
    mat_size = mat.shape[0]
    diag = np.zeros((mat_size, mat_size))
    np.fill_diagonal(diag, np.power(np.sum(mat, axis=0), -1 / 2))
    ret = diag.dot(mat).dot(diag)
    return ret


def construct_het_graph(rna_dis_mat, dis_mat, rna_mat, miu):
    mat1 = np.hstack((rna_mat * miu, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat * miu))
    ret = np.vstack((mat1, mat2))
    return ret


def construct_adj_mat(training_mask):
    adj_tmp = training_mask.copy()
    adj_tmp = (1 - adj_tmp) * -1e9
    rna_mat = np.zeros((training_mask.shape[0], training_mask.shape[0]))
    dis_mat = np.zeros((training_mask.shape[1], training_mask.shape[1]))

    mat1 = np.hstack((rna_mat, adj_tmp))
    mat2 = np.hstack((adj_tmp.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret


def calculate_auc(rna_dis_adj_mat, pred_adj_mat, testing_data_idx):
    pred_adj_mat = np.reshape(pred_adj_mat, (rna_dis_adj_mat.shape[0], rna_dis_adj_mat.shape[1]))
    row_idx = testing_data_idx[0]
    col_idx = testing_data_idx[1]
    truth_score = rna_dis_adj_mat[row_idx, col_idx]
    pred_score = pred_adj_mat[row_idx, col_idx]


    fpr, tpr, thresholds = roc_curve(truth_score, pred_score)
    test_auc = auc(fpr, tpr)

    test_aupr = average_precision_score(truth_score, pred_score)
    precision, recall, thresholds = precision_recall_curve(truth_score, pred_score)
    pred_list = [int(x.round()) for x in pred_score]

    accuracy_val = metrics.accuracy_score(truth_score, pred_list)
    precision_val = metrics.precision_score(truth_score, pred_list)
    recall_val = metrics.recall_score(truth_score, pred_list)
    f1_val = metrics.f1_score(truth_score, pred_list)

    return test_auc,test_aupr,accuracy_val,precision_val,recall_val,f1_val,fpr,tpr,precision,recall
