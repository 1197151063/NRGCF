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

config = world.config
seed = config['seed']
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

utils.set_seed(world.seed)
print(">>SEED:", world.seed)
model_name = 'Denoised-' + config['model'] + '-' + config['dataset'] + '-' + str(config['seed']) +  '-' + str(config['noise_rate']) + '.pth.tar'
file_path = '../data/gowalla/'
save_path = '/root/autodl-tmp/models/'
dataset = dataloader.Loader(path=file_path)
# Recmodel = LightGCN(config,dataset)
Recmodel = GTN(config,dataset,args = world.args)
Recmodel = Recmodel.to(world.device)
Recmodel.load_state_dict(torch.load('/root/autodl-tmp/models/Robust-GTN-gowalla-2023-0.5.pth.tar',map_location=torch.device('cpu')))
users = torch.tensor(dataset.testUniqueUsers)


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
fixed_ts = 0
g = dataset.UserItemNet.toarray()
g = torch.tensor(g)
# g = g.to(world.device)
args = world.args
noise_items = dataset.noise_items
ratings = Recmodel.getUsersRating(users.long())
# Recmodel = Recmodel.detach().cpu()
ratings = ratings.detach().cpu()
# print(ratings)

# print(len(noise_items))
best_ts = 0
max_gap = 0
ts = 0
for i in range(100):
    adj = torch.mul(g,ratings)
    ts += 0.01
    # ts = fixed_ts * mean_cf
    # cprint(f"fixed:{fixed_ts} , mean_cf:{mean_cf} , ts : {ts}")
    # cprint(mean_cf)
    # bprint(ts)

    rowind = 0
    hit = 0 
    noise_total = 0
    # bprint("[DENOSING]")
    for nitems in noise_items:
        noise_total += len(nitems)
        for item in nitems:
            if adj[rowind][item] <= ts:
                hit += 1
        rowind += 1
    interaction_total = dataset.trainDataSize
    # bprint(f"hitting {hit} noise , hit success rate {hit / noise_total}")
    adj[adj <= ts] = 0
    adj[adj > ts] = 1
    adj = adj.detach().numpy()
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
    # del dataset
    # dataset_tmp = dataloader.Loader(path=file_path,flag=1,g=adj,hit=hit)
    # interaction_filtered = dataset_tmp.trainDataSize
    removed = interaction_total - adj.sum()
    if hit/noise_total - (removed - hit) * 1 / (interaction_total - noise_total) >= max_gap:
        max_gap = hit/noise_total - (removed - hit) * 1 / (interaction_total - noise_total)
        best_ts = ts
    # bprint(f"removing {removed} edges , mistake rate {(removed - hit) / (interaction_total - noise_total)}")
    # print('\n')

cprint(best_ts)