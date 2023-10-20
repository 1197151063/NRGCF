#using earlystopping to train models 

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
config = world.config
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

utils.set_seed(world.seed)
print(">>SEED:", world.seed)
model_name = 'Robust-' + config['model']
save_path = '/root/autodl-tmp/models/'
log_path = '/root/autodl-tmp/log/'

noise_ratio = round(config['noise_rate'],1)
file_path = '../data/' + config['dataset']

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


log_file_name = config['model'] + '-' + config['dataset'] + '-' + str(noise_ratio) + '-log.txt'
save_file_name = model_name + '-' + config['dataset'] + '-' + str(noise_ratio) + '.pth.tar'
best_m1 = 0
best_m2 = 0
cprint(f"{model_name} is ready to go with noise ratio: {noise_ratio}")
print(f"model saved to {save_path}")

with open(log_path+log_file_name,'w') as f:
    for epoch in range(world.TRAIN_epochs):
        if config['model']=='GTN':
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        else:
            output_information = Procedure.BPR_train_original_1(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        bprint("[TEST]")
        results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], val=False)

        pre = round(results['precision'][0], 5)
        recall = round(results['recall'][0], 5)
        ndcg = round(results['ndcg'][0], 5)
        if recall >= best_m1 and ndcg >= best_m2:
            bprint('es model selected , best epoch saved')
            torch.save(Recmodel.state_dict(),save_path+save_file_name)
            best_m1 = recall
            best_m2 = ndcg
        topk_txt = f'Testing EPOCH[{epoch + 1}/{world.TRAIN_epochs}]  {output_information} | Results Top-k (pre, recall, ndcg): {pre}, {recall}, {ndcg}'
        print(topk_txt)
        f.write(topk_txt+'\n')
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information} | Results val Top-k (recall, ndcg):  {recall}, {ndcg}')