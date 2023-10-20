#using early stopping models to filter noise and then train on it 

from world import cprint,bprint
from model import GTN,LightGCN,NGCF
import Procedure
import dataloader
import torch
import random
import numpy as np
import world
import utils
from tqdm import tqdm
import copy
from scipy.sparse import csr_matrix

seed = 2023
config = world.config
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

utils.set_seed(world.seed)
print(">>SEED:", world.seed)
model_name = 'Robust-' + config['model']
file_path = '../data/gowalla/'
save_path = '/root/autodl-tmp/models/'
dataset = dataloader.Loader(path=file_path)
Recmodel = LightGCN(config,dataset)
Recmodel = Recmodel.to(world.device)
Recmodel.load_state_dict(torch.load('../Robust-LGCN-gowalla-0.3.pth.tar',map_location=torch.device('cpu')))
users = torch.tensor(dataset.trainUniqueUsers)



def getConfidence(Recmodel):
    users = dataset.testUniqueUsers
    t_users = torch.Tensor(users).long()
    ratings = Recmodel.getUsersRating(t_users)
    ratings = ratings.detach().cpu().numpy()
    test_item = dataset.m_test_items
    mcs = 0
    for user in users:
        mc = 0
        for item in test_item[user]:
            mc += ratings[user][item]
        mc /= len(test_item[user])
        mcs += mc
    mcs /= len(users)
    return mcs

mean_cf = getConfidence(Recmodel)
fixed_ts = 0.95
g = dataset.UserItemNet.toarray()
g = torch.tensor(g)
# g = g.to(world.device)
args = world.args
noise_items = dataset.noise_items
print(len(noise_items))
for i in range(1):
    w = None
    Neg_k = 1
    fixed_ts += 0.05
    ts = fixed_ts * mean_cf
    # cprint(mean_cf)
    # bprint(ts)
    ratings = Recmodel.getUsersRating(users.long())
    # Recmodel = Recmodel.detach().cpu()
    ratings = ratings.detach().cpu()
    # print(ratings)
    adj = torch.mul(g,ratings)
    rowind = 0
    hit = 0 
    bprint("[DENOSING]")
    for nitems in tqdm(noise_items):
        for item in nitems:
            if adj[rowind][item] <= ts:
                adj[rowind][item] = 0
                hit += 1
        rowind += 1
    adj[adj > ts] = 1
    adj = adj.detach().cpu().numpy()
    # hit = 0
    # i = 0
    # for user in adj:
    #     filtered_items = np.where(user == 1)
    #     filtered_items = set(filtered_items[0].tolist())
    #     noisy_items = set(noise_items[i])
    #     i += 1
    #     hit += len(filtered_items.intersection(noisy_items))
    # bprint(hit)
    # print(adj)
    dataset_tmp = dataloader.Loader(path=file_path,flag=1,g=adj,hit=hit)
    Recmodel = LightGCN(config,dataset_tmp)
    Recmodel = Recmodel.to(world.device)
    best_m1 = 0
    best_m2 = 0
    if config['model'] == 'GTN':
        bpr = utils.BPRLoss(Recmodel,config)
    elif config['model'] == 'LGCN':
        bpr = utils.BPRLoss1(Recmodel,config)
    elif config['model'] == 'NGCF':
        bpr = utils.BPRLoss1(Recmodel,config)
    for epoch in range(world.TRAIN_epochs):
        if config['model']=='GTN':
            output_information = Procedure.BPR_train_original(dataset_tmp, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        else:
            output_information = Procedure.BPR_train_original_1(dataset_tmp, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        if epoch % 10 == 0:
            bprint("[TEST]")
            results = Procedure.Test(dataset_tmp, Recmodel, epoch, w, world.config['multicore'], val=False)

            pre = round(results['precision'][0], 5)
            recall = round(results['recall'][0], 5)
            ndcg = round(results['ndcg'][0], 5)
        # if recall >= best_m1 and ndcg >= best_m2:
        #     bprint('es model selected , best epoch saved')
        #     torch.save(Recmodel.state_dict(),'model_tmp/test_model2.tar.pth')
        #     best_m1 = recall
        #     best_m2 = ndcg
        topk_txt = f'Testing EPOCH[{epoch + 1}/{world.TRAIN_epochs}]  {output_information} | Results Top-k (pre, recall, ndcg): {pre}, {recall}, {ndcg}'
        print(topk_txt)
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information} | Results val Top-k (recall, ndcg):  {recall}, {ndcg}')
     