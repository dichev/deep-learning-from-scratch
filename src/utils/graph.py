import torch

def edge_index_to_adj_list(edge_index, bidirectional=False, num_nodes=None):
    assert edge_index.shape[0] == 2, f'Expected edge index shape [2, E] but got {edge_index.shape}'
    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    adj_list = {i: set() for i in range(num_nodes)}
    for i, j in edge_index.t().tolist():
        adj_list[i].add(j)
        if bidirectional:
            adj_list[j].add(i)
    return adj_list


def edge_index_to_adj_matrix(edge_index, sparse=False, num_nodes=None):
    n, E = edge_index.shape
    assert n == 2, f'Expected edge index shape [2, E] but got {edge_index.shape}'
    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    if sparse:
        values = torch.ones(E, device=edge_index.device)
        A = torch.sparse_coo_tensor(edge_index, values, size=(num_nodes, num_nodes))
    else:
        V, W = edge_index
        A = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        A[V, W] = 1

    return A
