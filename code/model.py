from torch import nn,Tensor,LongTensor
from torch_geometric.utils import degree
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import SparseTensor
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
import torch
import world
device = world.device


class RecModel(MessagePassing):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.config = config
        self.edge_index = edge_index
        self.embedding_dim = config['dim']
        self.user_embedding = nn.Embedding(num_users,self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items,self.embedding_dim)
        self.dropout = nn.Dropout(p=world.dropout_rate)
        self.reset_parameters(config)
    
    def reset_parameters(self,config):
        if(config['init'] == 'normal'):
            nn.init.normal_(self.user_embedding.weight.data,std=config['init_weight'])
            nn.init.normal_(self.item_embedding.weight.data,std=config['init_weight'])
        else:
            nn.init.xavier_uniform_(self.user_embedding.weight.data,gain=config['init_weight'])
            nn.init.xavier_uniform_(self.item_embedding.weight.data,gain=config['init_weight'])
        self.f = nn.Sigmoid()

    def get_sparse_bipartite_graph(self,
                                    edge_index,
                                    use_value=False,
                                    value=None):
        num_users = self.num_users
        num_items = self.num_items
        r,c = edge_index
        if use_value:
            return SparseTensor(row=r,col=c,value=value,sparse_sizes=(num_users,num_items))
        else:
            return SparseTensor(row=r,col=c,sparse_sizes=(num_users,num_items))
        
    def get_sparse_graph(self,
                         edge_index,
                         use_value=False,
                         value=None):
        num_users = self.num_users
        num_nodes = self.num_nodes
        r,c = edge_index
        row = torch.cat([r , c + num_users])
        col = torch.cat([c + num_users , r])
        if use_value:
            value = torch.cat([value,value])
            return SparseTensor(row=row,col=col,value=value,sparse_sizes=(num_nodes,num_nodes))
        else:
            return SparseTensor(row=row,col=col,sparse_sizes=(num_nodes,num_nodes))
    
    def link_prediction(self,
                        src_index:Tensor=None,
                        dst_index:Tensor=None):
        out_u,out_i = self.forward(edge_index=self.edge_index)
        # out_u = F.normalize(out_u, dim=-1)
        # out_i = F.normalize(out_i, dim=-1)
        if src_index is None:
            src_index = torch.arange(self.num_users).long()
        if dst_index is None:
            dst_index = torch.arange(self.num_items).long()
        out_src = out_u[src_index]
        out_dst = out_i[dst_index]
        pred = out_src @ out_dst.t()
        return pred
    
    def get_sparse_bipartite_graph_transpose(self,
                                             edge_index:LongTensor,
                                             use_value=False,
                                             value=None):
        num_users = self.num_users
        num_items = self.num_items
        r,c = edge_index
        if use_value:
            return SparseTensor(row=c,col=r,value=value,sparse_sizes=(num_items,num_users))
        else:
            return SparseTensor(row=c,col=r,sparse_sizes=(num_items,num_users))
    
    def forward(self,edge_index:LongTensor=None):
        pass

    def bpr_loss(self,edge_label_index):
        user_emb,item_emb = self.forward(edge_index=self.edge_index)
        user_emb = user_emb[edge_label_index[0]]
        pos_item_emb = item_emb[edge_label_index[1]]
        neg_item_emb = item_emb[edge_label_index[2]]
        pos_rank = (user_emb * pos_item_emb).sum(dim=-1)
        neg_rank = (user_emb * neg_item_emb).sum(dim=-1)
        return F.softplus(neg_rank - pos_rank).mean()
    
    def get_loss(self,edge_label_index):
        pass 
    
    def l2_reg(self,edge_label_index):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        embedding = torch.cat([user_emb[edge_label_index[0]],
                               item_emb[edge_label_index[1]],
                               item_emb[edge_label_index[2]]])
        regularization =  (1/2) * embedding.norm(p=2).pow(2)/ edge_label_index.size(1)
        return self.config['decay'] * regularization

    def ssm_loss(self,edge_label_index:LongTensor):
        user_emb,item_emb = self.forward(edge_index=None)
        neg_edge_index = torch.randint(0, self.num_items,(edge_label_index[1].numel(),world.num_neg), device=device)
        embedding = torch.cat([user_emb[edge_label_index[0]],
                               item_emb[edge_label_index[1]],
                               item_emb[neg_edge_index].view(-1, item_emb.size(-1))])
        regularization = self.config['decay'] * (1/2) * embedding.norm(p=2).pow(2)/ edge_label_index.size(1)
        user_emb = user_emb[edge_label_index[0]]
        pos_item_emb = item_emb[edge_label_index[1]]
        neg_item_emb = item_emb[neg_edge_index]
        user_emb = F.normalize(user_emb, dim=-1)
        item_emb = torch.cat([pos_item_emb.unsqueeze(1), neg_item_emb], dim=1)
        item_emb = F.normalize(item_emb, dim=-1)
        # user_emb = self.dropout(user_emb)
        y_pred = torch.bmm(item_emb, user_emb.unsqueeze(-1)).squeeze(-1)
        pos_logits = torch.exp(y_pred[:, 0] / self.config['tau']) 
        neg_logits = torch.exp(y_pred[:, 1:]/ self.config['tau']) 
        Ng = neg_logits.sum(dim=-1)
        loss = (- torch.log(pos_logits / Ng))
        return loss.mean() + regularization
    
    def alignment_loss(self,edge_label_index:LongTensor):
        user_emb,item_emb = self.forward(edge_index=None)
        user_emb = user_emb[edge_label_index[0]]
        item_emb = item_emb[edge_label_index[1]]
        user_emb = F.normalize(user_emb, dim=-1)
        item_emb = F.normalize(item_emb, dim=-1)
        return (user_emb - item_emb).norm(dim=1).pow(2).mean()
    
    def uniformity(self,x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    def uniformity_loss(self,edge_label_index:LongTensor):
        user_emb,item_emb = self.forward(edge_index=None)
        user_emb = user_emb[edge_label_index[0]]
        item_emb = item_emb[edge_label_index[1]]
        return   (self.uniformity(user_emb) + self.uniformity(item_emb))
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t,x)
    
class MF(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__(num_users,num_items,config,edge_index)
        self.user_degree = degree(self.edge_index[0], self.num_users)
        self.item_degree = degree(self.edge_index[1], self.num_items)
        
    
    def forward(self,edge_index=None):
        return self.user_embedding.weight, self.item_embedding.weight
    
    def get_loss(self,edge_label_index):
        rank_loss = self.bpr_loss(edge_label_index) + self.l2_reg(edge_label_index)
        return rank_loss 

class LightGCN(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__(num_users,num_items,config,edge_index)
        self.edge_index = self.get_sparse_graph(edge_index, use_value=False)
        self.edge_index = gcn_norm(self.edge_index)

    
    def forward(self,edge_index):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        x = torch.cat([user_emb, item_emb], dim=0)
        out = [x]
        for i in range(self.config['K']):
            x = self.propagate(edge_index, x=x)
            out.append(x)
        out = torch.stack(out, dim=1)
        out = out.mean(dim=1)
        user_emb = out[:self.num_users]
        item_emb = out[self.num_users:]
        return user_emb, item_emb

    def get_loss(self,edge_label_index):
        rank_loss = self.bpr_loss(edge_label_index) + self.l2_reg(edge_label_index)
        return rank_loss 
        

class NRGCF(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__(num_users,num_items,config,edge_index)
        self.edge_index = self.get_sparse_graph(edge_index, use_value=False)
        self.edge_index = gcn_norm(self.edge_index)
    
    def cross_norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_norm = (1e-6 + users.pow(2).sum(dim=1).mean())
        items_norm = (1e-6 + items.pow(2).sum(dim=1).mean())
        users = users / (items_norm ** 0.8)
        items = items / (users_norm ** 0.8)
        x = torch.cat([users,items])
        return x
        
    def forward(self,edge_index):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        x = torch.cat([user_emb, item_emb], dim=0)
        out = [x]
        for i in range(self.config['K']):
            x = self.propagate(edge_index, x=x)
            x = self.cross_norm(x)
            out.append(x)
        out = torch.stack(out, dim=1)
        out = out.mean(dim=1)
        user_emb = out[:self.num_users]
        item_emb = out[self.num_users:]
        return user_emb, item_emb
    
    def get_loss(self,edge_label_index):
        rank_loss = self.bpr_loss(edge_label_index) + self.l2_reg(edge_label_index)
        return rank_loss 