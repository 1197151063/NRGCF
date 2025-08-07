from model import NRGCL
import torch
from dataloader import Loader
import world
from procedure import train,test
import utils
import torch.nn.functional as F
import time
from utils import init_logger, print_log, write_final_log
if world.config['dataset'] == 'yelp2018':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':0.01,#INIT WEIGHT
        'K':3,#GCN_LAYER
        'lambda': world.lambda_,
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'ssl_tmp':0.2,#TEMPERATURE
        'ssl_decay':0.2,#SSL_STRENGTH
        'drop_ratio':0.1,#AUG_RATIO
        'type':'ED',#GRAPH_AUG_TYPE
    }
if world.config['dataset'] == 'amazon-book':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':0.01,#INIT WEIGHT
        'K':3,#GCN_LAYER
        'lambda': world.lambda_,
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'ssl_tmp':0.2,#TEMPERATURE
        'ssl_decay':0.5,#SSL_STRENGTH
        'drop_ratio':0.1,#AUG_RATIO
        'type':'ED',#GRAPH_AUG_TYPE
    }

    
device = world.device
dataset = Loader()
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = NRGCL(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.
log_path = init_logger(model_name='NR-GCF_InfoNCE', dataset_name=world.config['dataset'])
for epoch in range(1, 1001):
    start_time = time.time()
    model.generate_graph()
    loss = train(dataset=dataset,model=model,opt=opt)
    end_time = time.time()
    recall,ndcg = test([20],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if patience == 0:
        best_epoch = epoch
        best_recall = recall[20]
        best_ndcg = ndcg[20]
    if flag == 1:
        break
    print_log(f'Epoch: {epoch:03d}, aver_loss : {loss:.5f}, R@20: '
            f'{recall[20]:.4f}, N@20: {ndcg[20]:.4f}, '
            f'time:{end_time-start_time:.2f} seconds')
write_final_log(best_epoch=best_epoch, recall=best_recall, ndcg=best_ndcg, config=config)
print_log(f"Log saved to: {log_path}")