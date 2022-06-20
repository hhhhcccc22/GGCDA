import torch
from data_load import *
torch.set_default_tensor_type(torch.DoubleTensor)

def rwr(A, restart_prob):
    n = A.shape[0]
    A = (A + A.T) / 2
    A = A - np.diag(np.diag(A))
    A = A + np.diag(sum(A) == 0)
    P = A / sum(A)
    Q = np.linalg.inv(np.eye(n) - (1 - restart_prob) * P) @ (restart_prob * np.eye(n))
    return Q

def train():
    dis_sim_mat = np.loadtxt("CircR2Disease/d-d.csv", delimiter=",")
    rna_sim_mat = np.loadtxt("CircR2Disease/c-c.csv", delimiter=",")
    rna_dis_adj_mat = np.loadtxt("CircR2Disease/ass matrix.csv", delimiter=",")

    dis_sim_mat = rwr(dis_sim_mat, 0.9)
    rna_sim_mat = rwr(rna_sim_mat, 0.9)
    device = torch.device("cpu")

    k_folds = 5
    final_auc = 0
    final_aupr = 0
    final_acc = 0
    final_precision = 0
    final_recall = 0
    final_f1 = 0

    idx = load_data(rna_dis_adj_mat, k_folds)
    total_auc = 0
    total_aupr = 0
    total_acc = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    fprs = []
    tprs = []
    pres = []
    recs = []
    AUCS = []
    AUPRS = []

    for i in range(k_folds):
        print(f"fold:" + str(i+1))
        model = torch.load('C:\\Users\\hec\\OneDrive\\桌面\\IMCHGAN-main\\GGCDA\\GGCDA.pth')
        training_mat = rna_dis_adj_mat.copy()
        training_mat[tuple(idx[i])] = 0
        het_mat = construct_het_graph(training_mat, dis_sim_mat, rna_sim_mat, 0.6)
        adj_mat = construct_adj_mat(training_mat)
        het_graph = torch.tensor(het_mat).to(device=device)
        adj_graph = torch.tensor(adj_mat).to(device=device)
        graph_data = (het_graph, adj_graph)

        model.eval()
        link_pred = model(graph_data, adj_mat).cpu().detach().numpy()
        test_auc, test_aupr, acc, p, r, f1, fpr_fold, tpr_fold, precision, recall = calculate_auc(rna_dis_adj_mat,
                                                                                                  link_pred, idx[i])

        print(f"AUC:{test_auc} AUPR:{test_aupr} ACC:{acc} PRE:{p} REC:{r} F1-SCORE:{f1} ")
        total_auc += test_auc / k_folds
        total_aupr += test_aupr / k_folds
        total_acc += acc / k_folds
        total_precision += p / k_folds
        total_recall += r / k_folds
        total_f1 += f1 / k_folds
        fprs.append(fpr_fold)
        tprs.append(tpr_fold)
        pres.append(precision)
        recs.append(recall)
        AUCS.append(test_auc)
        AUPRS.append(test_aupr)

    print(f"{k_folds}-folds average auc,aupr,acc,pre,rec,f1:{total_auc, total_aupr, total_acc, total_precision, total_recall, total_f1}")
    final_auc += (total_auc / 10)
    final_aupr += (total_aupr / 10)
    final_acc += (total_acc / 10)
    final_precision += (total_precision / 10)
    final_recall += (total_recall / 10)
    final_f1 += (total_f1 / 10)


if __name__ == '__main__':
    train()
