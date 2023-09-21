#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader

class BundleTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, neg_sample=1):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample
        self.bundle_list = np.arange(self.num_bundles)
    def __getitem__(self, index):
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and not i in all_bundles:
                all_bundles.append(i)
                if len(all_bundles) == self.neg_sample+1:
                    break
        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)
    
    # def __getitem__(self, index):
    #     temp = np.zeros(1+self.neg_sample, dtype=np.int32)
    #     user,temp[0]= self.u_b_pairs[index]
    #     pos_index = self.u_b_graph[user].nonzero()[1]
    #     pro = np.ones(self.num_bundles)/(self.num_bundles - len(pos_index))
    #     pro[pos_index] = 0
    #     temp[1:1 + self.neg_sample] = np.random.choice(self.bundle_list, size=self.neg_sample, p=pro)
    #     return torch.LongTensor([user]),torch.Tensor(temp).long()

    def __len__(self):
        return len(self.u_b_pairs)


class BundleTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.train_mask_u_b = u_b_graph_train

        self.num_users = num_users
        self.num_bundles = num_bundles

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(num_bundles, dtype=torch.long)

    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()
        return index, u_b_grd, u_b_mask

    def __len__(self):
        return self.u_b_graph.shape[0]


class Datasets():
    def __init__(self, conf):
        self.path = conf['dataset_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size']
        batch_size_test = conf['batch_size']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_pairs, b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")


        self.bundle_train_data = BundleTrainDataset(
            conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, conf["neg_num"])
        self.bundle_val_data = BundleTestDataset(
            u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(
            u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        self.graph = self.get_graph_InBornDec(u_b_graph_train, u_i_graph, b_i_graph)

        self.train_loader = DataLoader(
            self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=20,worker_init_fn = seed_worker)
            # self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=20)
        self.val_loader = DataLoader(
            self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
        self.test_loader = DataLoader(
            self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)


    def get_graph_InBornDec(self, B, I, A):
        try:
            pre_adj_mat = sp.load_npz(os.path.join(
                self.path, self.name, 'UB2UI_mat.npz'))
            norm_adj = pre_adj_mat

        except:
            adj_mat = sp.dok_matrix(
                (self.num_users + self.num_bundles, self.num_users + self.num_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()

            B = B.tolil()  # U B
            I = I.tolil()  # U I
            A = A.tolil()  # B I

            adj_mat[:self.num_users, self.num_users:] = I  # U I
            adj_mat[self.num_users:, :self.num_users] = B.T  # B U
            adj_mat[self.num_users:, self.num_users:] = A
            adj_mat = adj_mat.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            row_inv = np.power(rowsum, -0.5).flatten()
            row_inv[np.isinf(row_inv)] = 0.
            row_mat = sp.diags(row_inv)

            colsum = np.array(adj_mat.sum(axis=0))
            col_inv = np.power(colsum, -0.5).flatten()
            col_inv[np.isinf(col_inv)] = 0.
            col_mat = sp.diags(col_inv)

            norm_adj = row_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(col_mat)
            norm_adj = norm_adj.tocsr()
            sp.save_npz(os.path.join(self.path, self.name,
                                     'UB2UI_mat.npz'), norm_adj)
        return self._convert_sp_mat_to_sp_tensor(norm_adj)

    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]

    def get_bi(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i)
                             for i in s[:-1].split('\t')), f.readlines()))
        b_i_graph = self.get_csr_graph(np.array(b_i_pairs, dtype=np.int32),self.num_bundles, self.num_items)
        print_statistics(b_i_graph, 'B-I statistics')
        return b_i_pairs, b_i_graph

    def get_ui(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i)
                             for i in s[:-1].split('\t')), f.readlines()))
        u_i_graph = self.get_csr_graph(np.array(u_i_pairs, dtype=np.int32),self.num_users,self.num_items)
        print_statistics(u_i_graph, 'U-I statistics')
        return u_i_pairs, u_i_graph

    def get_ub(self, task):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i)
                             for i in s[:-1].split('\t')), f.readlines()))
        u_b_graph = self.get_csr_graph(np.array(u_b_pairs, dtype=np.int32),self.num_users,self.num_bundles)
        print_statistics(u_b_graph, "U-B statistics in %s" % (task))
        return u_b_pairs, u_b_graph

    def get_csr_graph(self, pairs, n, m):
        values = np.ones(len(pairs), dtype=np.float32)
        return sp.coo_matrix((values, (pairs[:, 0], pairs[:, 1])), shape=(n, m)).tocsr()
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def print_statistics(X, string):
    print('>'*10 + string + '>'*10)
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero number',X.nnz)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', X.nnz/(X.shape[0]*X.shape[1]))
