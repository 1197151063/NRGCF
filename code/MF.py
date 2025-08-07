import torch
from dataloader import Loader
import world
from procedure import train,test
import utils
import time
from model import MF
from utils import init_logger, print_log, write_final_log


if world.config['dataset'] == 'yelp2018':
    config = {
        'init':'uniform',#NORMAL DISTRIBUTION
        'init_weight':world.init_weight,#INIT WEIGHT
        'dim':64,#EMBEDDING_SIZE
        'decay':world.decay,#L2_NORM
        'K':1,
        'tau':world.tau,
        'lr':world.lr,#LEARNING_RATE
        'num_neg':world.num_neg,
        'dropout':world.dropout_rate,#DROPOUT_RATE
    }

if world.config['dataset'] == 'amazon-book':
    config = {
        'init':'uniform',#NORMAL DISTRIBUTION
        'init_weight':world.init_weight,#INIT WEIGHT
        'dim':64,#EMBEDDING_SIZE
        'decay':world.decay,#L2_NORM
        'K':1,
        'tau':world.tau,
        'lr':world.lr,#LEARNING_RATE
        'num_neg':world.num_neg,
        'dropout':world.dropout_rate,#DROPOUT_RATE
    }
    

device = world.device
dataset = Loader()
log_path = init_logger(model_name='MF', dataset_name=world.config['dataset'])


train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = MF(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.
best_recall = 0.
best_epoch = 0
best_ndcg = 0.
# print(model.generate_weight(train_edge_index))
for epoch in range(1, 2001):
    start_time = time.time()
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
