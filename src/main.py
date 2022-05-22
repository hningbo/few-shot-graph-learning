"""
@Filename       : main.py
@Create Time    : 2022/5/15 22:16
@Author         : Rylynn
@Description    : 

"""

import argparse
import time

import torch as th
import torch.functional.F as F
import random
import numpy as np
# Training settings
from utils.data import task_generator, load_data
from utils.metric import euclidean_dist, accuracy, f1

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--episodes', type=int, default=1000,
                    help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


parser.add_argument('--way', type=int, default=5, help='way.')
parser.add_argument('--shot', type=int, default=5, help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=20)
parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset:Amazon_clothing/Amazon_eletronics/dblp')

args = parser.parse_args()
args.cuda = args.use_cuda and th.cuda.is_available()

random.seed(args.seed)
th.manual_seed(args.seed)
if args.cuda:
    th.cuda.manual_seed(args.seed)

dataset = args.dataset
adj, features, labels, degrees, idx_list_train, idx_list_valid, idx_list_test, id_by_class = load_data('../data', dataset)


def train(class_selected, id_support, id_query, n_way, k_shot):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = th.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = th.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / th.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)

    output = F.log_softmax(-dists, dim=1)

    labels_new = th.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)

    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train


def test(class_selected, id_support, id_query, n_way, k_shot):
    encoder.eval()
    scorer.eval()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = th.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = th.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / th.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = th.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = F.nll_loss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test


if __name__ == '__main__':
    n_way = args.way
    k_shot = args.shot
    n_query = args.qry
    meta_test_num = 50
    meta_valid_num = 50

    # Sampling a pool of tasks for validation/testing
    valid_pool = [task_generator(id_by_class, idx_list_valid, n_way, k_shot, n_query) for i in range(meta_valid_num)]
    test_pool = [task_generator(id_by_class, idx_list_test, n_way, k_shot, n_query) for i in range(meta_test_num)]

    # Train model
    t_total = time.time()
    meta_train_acc = []

    for episode in range(args.episodes):
        id_support, id_query, class_selected = \
            task_generator(id_by_class, idx_list_train, n_way, k_shot, n_query)
        acc_train, f1_train = train(class_selected, id_support, id_query, n_way, k_shot)
        meta_train_acc.append(acc_train)
        if episode > 0 and episode % 10 == 0:
            print("-------Episode {}-------".format(episode))
            print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))

            # validation
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_valid_num):
                id_support, id_query, class_selected = valid_pool[idx]
                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            print("Meta-valid_Accuracy: {}, Meta-valid_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                                      np.array(meta_test_f1).mean(axis=0)))
            # testing
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_test_num):
                id_support, id_query, class_selected = test_pool[idx]
                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                                    np.array(meta_test_f1).mean(axis=0)))

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))