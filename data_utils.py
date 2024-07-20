import numpy as np


def split_non_iid_dataset(labels, index, alpha, n_clients):
    # labels是整个数据集的labels
    # index是最终需要用到的下标
    n_classes = max(labels) + 1
    index_by_label = [np.argwhere(labels[index] == y).flatten() for y in range(n_classes)]
    dirichlet_distribution = np.random.dirichlet(alpha=[alpha] * n_clients, size=n_classes)
    client_index_list = [[] for _ in range(n_clients)]
    # c为某一类的分布比例
    for frac, y in zip(dirichlet_distribution, index_by_label):
        div = (frac.cumsum() * len(y))[:-1].astype(int)
        for i, lbl in enumerate(np.split(y, div)):
            client_index_list[i] += lbl.tolist()
    return client_index_list
