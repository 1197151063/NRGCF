#this is a script to test the performance of a model

from model import GTN,LightGCN,NGCF
import Procedure
import world 
from world import cprint, bprint
import numpy as np
import torch 
import utils 
import dataloader
import random

seed = world.seed

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
utils.set_seed(world.seed)
print(">>SEED:", world.seed)


model_path = 'model_tmp/Robust-LGCN-gowalla-2023-0.5.pth.tar'

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

Recmodel.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
Recmodel = Recmodel.to(world.device)
Neg_k = 1
w = None
epoch = 0
results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], val=False)
cprint(results)