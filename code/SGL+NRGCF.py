from model import RecModel
from torch import LongTensor,Tensor
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
from dataloader import Loader
import world
from procedure import train_bpr_sgl,test
import utils
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge,dropout_path,bipartite_subgraph
import time
from utils import init_logger, print_log, write_final_log
if world.config['dataset'] == 'yelp2018':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':0.01,#INIT WEIGHT
        'K':3,#GCN_LAYER
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
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'ssl_tmp':0.2,#TEMPERATURE
        'ssl_decay':0.5,#SSL_STRENGTH
        'drop_ratio':0.1,#AUG_RATIO
        'type':'ED',#GRAPH_AUG_TYPE
    }


def dropout_node_bipartite(
    edge_index: Tensor,
    p: float = 0.5,
    num_users = None,
    num_items = None,
    relabel_nodes: bool = False,
) -> Tensor:
    r"""Randomly drops nodes from bipartite graph
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        size: The number of two type of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index[0]` and `edge_index[1]`. 
        relabel_nodes (bool, optional): If set to `True`, the resulting
            `edge_index` will be relabeled to hold consecutive indices
            starting from zero.

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')
    if num_users is None:
        num_users = edge_index[0].max() + 1
    if num_items is None:
        num_items = edge_index[1].max() + 1

    prob_users = torch.rand(num_users, device=edge_index.device)
    prob_items = torch.rand(num_items, device=edge_index.device)
    user_index = torch.arange(0,num_users,device=edge_index.device)
    item_index = torch.arange(0,num_items,device=edge_index.device)
    user_mask = prob_users > p
    item_mask = prob_items > p
    node_mask = (user_index[user_mask],item_index[item_mask])

    edge_index, _ = bipartite_subgraph(
        node_mask,
        edge_index,
        relabel_nodes=relabel_nodes,
        size=(num_users,num_items),
        return_edge_mask=False,
    )
    return edge_index


class SGL(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 config):
        super().__init__(num_users=num_users,
                         num_items=num_items,
                         edge_index=edge_index,
                         config=config)
        self.K = config['K']
        self.num_interactions = edge_index.size(1)
        self.edge_index = self.get_sparse_graph(edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(self.edge_index)        
        self.alpha= 1./ (1 + self.K)
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['dim'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['dim'])
        self.init_weight()
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K + 1
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K + 1))
        self.ssl_tmp = config['ssl_tmp']
        self.ssl_decay = config['ssl_decay']    
        print('Go SGL')
        print(f"params settings: \n emb_size:{config['dim']}\n L2 reg:{config['decay']}\n layer:{self.K}")
        print(f" ssl_tmp:{config['ssl_tmp']}\n ssl_decay:{config['ssl_decay']}\n graph aug type:{config['type']}")

    def init_weight(self):
        if config['init'] == 'normal':
            nn.init.normal_(self.user_emb.weight,std=config['init_weight'])
            nn.init.normal_(self.item_emb.weight,std=config['init_weight'])
        else:
            nn.init.xavier_uniform_(self.user_emb.weight,gain=config['init_weight'])
            nn.init.xavier_uniform_(self.item_emb.weight,gain=config['init_weight'])

    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            x = self.cross_norm(x)
            out = out + x * self.alpha[i + 1]
        return out
    def cross_norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_norm = (1e-6 + users.pow(2).sum(dim=1).mean()).sqrt()
        items_norm = (1e-6 + items.pow(2).sum(dim=1).mean()).sqrt()
        users = users / (items_norm)
        items = items / (users_norm)
        x = torch.cat([users,items])
        return x
    def forward(self,
                edge_label_index:Tensor):
        out = self.get_embedding()
        out_u,out_i = torch.split(out,[self.num_users,self.num_items])
        out_src = out_u[edge_label_index[0]]
        out_dst = out_i[edge_label_index[1]]
        out_dst_neg = out_i[edge_label_index[2]]
        return (out_src * out_dst).sum(dim=-1) ,(out_src * out_dst_neg).sum(dim=-1) 

    def get_ssl_embedding(self,edge_index):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=edge_index,x=x)
            x = self.cross_norm(x)
            out = out + x * self.alpha[i + 1]
        return out

    def ssl_loss(self,
                    edge_index1,
                    edge_index2,
                    edge_label_index):
        info_out1 = self.get_ssl_embedding(edge_index1)
        info_out2 = self.get_ssl_embedding(edge_index2)
        info_out_u_1,info_out_i_1 = torch.split(info_out1,[self.num_users,self.num_items])
        info_out_u_2,info_out_i_2 = torch.split(info_out2,[self.num_users,self.num_items])
        u_idx = torch.unique(edge_label_index[0])
        i_idx = torch.unique(edge_label_index[1])
        info_out_u1 = info_out_u_1[u_idx]
        info_out_u2 = info_out_u_2[u_idx]
        info_out_i1 = info_out_i_1[i_idx]
        info_out_i2 = info_out_i_2[i_idx]
        info_out_u1 = F.normalize(info_out_u1,dim=1)
        info_out_u2 = F.normalize(info_out_u2,dim=1)
        info_out_u_2 = F.normalize(info_out_u_2,dim=1)
        info_out_i_2 = F.normalize(info_out_i_2,dim=1)
        info_pos_user = (info_out_u1 * info_out_u2).sum(dim=1)/ self.ssl_tmp
        info_pos_user = torch.exp(info_pos_user)
        info_neg_user = (info_out_u1 @ info_out_u_2.t())/ self.ssl_tmp
        info_neg_user = torch.exp(info_neg_user)
        info_neg_user = torch.sum(info_neg_user,dim=1,keepdim=True)
        info_neg_user = info_neg_user.T
        ssl_logits_user = -torch.log(info_pos_user / info_neg_user).mean()
        info_out_i1 = F.normalize(info_out_i1,dim=1)
        info_out_i2 = F.normalize(info_out_i2,dim=1)
        info_pos_item = (info_out_i1 * info_out_i2).sum(dim=1)/ self.ssl_tmp
        info_neg_item = (info_out_i1 @ info_out_i_2.t())/ self.ssl_tmp
        info_pos_item = torch.exp(info_pos_item)
        info_neg_item = torch.exp(info_neg_item)
        info_neg_item = torch.sum(info_neg_item,dim=1,keepdim=True)
        info_neg_item = info_neg_item.T
        ssl_logits_item = -torch.log(info_pos_item / info_neg_item).mean()
        return self.ssl_decay * (ssl_logits_user + ssl_logits_item)
    
    def bpr_loss(self,pos_rank,neg_rank):
        return F.softplus(neg_rank - pos_rank).mean()
    
    def L2_reg(self,edge_label_index):
        u_idx,i_idx_pos,i_idx_neg = edge_label_index
        userEmb0 = self.user_emb.weight[u_idx]
        posEmb0 = self.item_emb.weight[i_idx_pos]
        negEmb0 = self.item_emb.weight[i_idx_neg]
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2)) / edge_label_index.size(1)
        regularization = self.config['decay'] * reg_loss
        return regularization
    
device = world.device
dataset = Loader()
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = SGL(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.
log_path = init_logger(model_name='NR-GCF', dataset_name=world.config['dataset'])
for epoch in range(1, 1001):
    edge_index = train_edge_index
    start_time = time.time()
    if config['type'] == 'ED':
        edge_index1,_ = dropout_edge(edge_index=edge_index,p=config['drop_ratio'])
        edge_index2,_ = dropout_edge(edge_index=edge_index,p=config['drop_ratio'])
    if config['type'] == 'ND':
        edge_index1 = dropout_node_bipartite(edge_index=edge_index,num_users=num_users,num_items=num_items,p=config['drop_ratio']/2)
        edge_index2 = dropout_node_bipartite(edge_index=edge_index,num_users=num_users,num_items=num_items,p=config['drop_ratio']/2)
    if config['type'] == 'RW':
        edge_index1,_ = dropout_path(edge_index=edge_index,p=config['drop_ratio'])
        edge_index2,_ = dropout_path(edge_index=edge_index,p=config['drop_ratio'])
        
    edge_index1 = model.get_sparse_graph(edge_index1)
    edge_index2 = model.get_sparse_graph(edge_index2)
    edge_index1 = gcn_norm(edge_index1)
    edge_index2 = gcn_norm(edge_index2)
    loss = train_bpr_sgl(dataset=dataset,
                         model=model,
                         opt=opt,
                         edge_index1=edge_index1,
                         edge_index2=edge_index2)
    end_time = time.time()
    recall,ndcg = test([20,50],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss}, R@20: '
          f'{recall[20]:.4f}, N@20: {ndcg[20]:.4f}, '
          f'time:{end_time-start_time:.2f} seconds')