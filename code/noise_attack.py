##this is the script to train a poisoned model 

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


seed = world.seed

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

utils.set_seed(world.seed)
print(">>SEED:", world.seed)

dataset = dataloader.Loader(path="../data/" + world.dataset)
args = world.args
config = world.config
if config['model'] == 'GTN':
    Recmodel = GTN(config,dataset,args)
    bpr = utils.BPRLoss(Recmodel,config)
elif config['model'] == 'LGCN':
    Recmodel = LightGCN(config,dataset)
    bpr = utils.BPRLoss1(Recmodel,config)
elif config['model'] == 'NGCF':
    Recmodel = NGCF(config,dataset)
    bpr = utils.BPRLoss1(Recmodel,config)
Recmodel = Recmodel.to(world.device)
Neg_k = 1
w = None
for epoch in range(world.TRAIN_epochs):
    if config['model']=='GTN':
        output_information = Procedure.BPR_train_original_1(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
    else:
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
    if epoch % 50 == 0:
        cprint("[TEST]")
        results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], val=False)

        pre = round(results['precision'][0], 5)
        recall = round(results['recall'][0], 5)
        ndcg = round(results['ndcg'][0], 5)

        topk_txt = f'Testing EPOCH[{epoch + 1}/{world.TRAIN_epochs}]  {output_information} | Results Top-k (pre, recall, ndcg): {pre}, {recall}, {ndcg}'
        print(topk_txt)
    print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information} | Results val Top-k (recall, ndcg):  {recall}, {ndcg}')

torch.save(Recmodel.state_dict(),'model_tmp/' + '-noised-' + config['model'] + '-' + config['dataset'] + '-' + config['seed'] + '-' + config['noise_rate'] + '.pth.tar')
