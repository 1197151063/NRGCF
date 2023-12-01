import torch 
import numpy as np
from world import cprint
from model import LightGCN,NGCF,GTN 
import world
import random
import utils
import dataloader
import Procedure
from ts_test import getConfidence

seed = world.seed

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
utils.set_seed(seed)
cprint(f"SEED:>>{seed}")

dataset = dataloader.Loader(path = "../data/" +  world.dataset)
config = world.config
args = world.args
device = world.device

if config['model']=='GTN':
    Recmodel = GTN(config,dataset,args)
    bpr = utils.BPRLoss(Recmodel,config)
elif config['model']=='LGCN':
    Recmodel = LightGCN(config,dataset)
    bpr = utils.BPRLoss1(Recmodel,config)
elif config['model']=='NGCF':
    Recmodel = NGCF(config,dataset)
    bpr = utils.BPRLoss1(Recmodel,config)
    
Recmodel = Recmodel.to(device)
Neg_k = 1 
w = None
model_name = config['model']

with open(model_name + ' mean confidence','w') as f: 
    for epoch in range(world.TRAIN_epochs):
        if config['model']=='GTN':
            output_information = Procedure.BPR_train_original_1(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        else:
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        mcs = getConfidence(Recmodel)
        mcs = round(mcs,2)
        cprint(mcs)
        f.write(mcs+'\n')
