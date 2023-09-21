import torch.nn as nn
import torch
import torch.nn.functional as F
class AGFN(nn.Module):
    def __init__(self,dataset,embedding_size, num_layers,drop,device,leaky):
        super(AGFN,self).__init__()
        self.dataset = dataset
        self.num_layers = num_layers
        self.latent_dim = embedding_size
        self.drop = 0
        self.leaky = leaky

        self.device = device
        self.set_weight()

    def set_weight(self):
        self.num_users  = self.dataset.num_users
        self.num_bundles  = self.dataset.num_bundles
        self.num_items = self.dataset.num_items
        self.embedding_u = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_b= nn.Embedding(
            num_embeddings=self.num_bundles, embedding_dim=self.latent_dim)
        self.filter = nn.Embedding(num_embeddings = self.num_layers,embedding_dim=self.num_items+self.num_users)
        nn.init.normal_(self.embedding_u.weight, std=0.1)
        nn.init.normal_(self.embedding_b.weight, std=0.1)
        nn.init.ones_(self.filter.weight)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.graph.to(self.device) ### Graph here
        print(self.Graph.shape)
        if self.drop:
            self.dropout_u = nn.Dropout(self.drop)
            self.dropout_b = nn.Dropout(self.drop)
    
    def _compute(self):
        users = self.embedding_u.weight
        bundles = self.embedding_b.weight
        filter = nn.functional.leaky_relu(self.filter.weight,self.leaky)

        if self.drop:
            users = self.dropout_u(users)
            bundles = self.dropout_b(bundles)
        all_embedding = torch.cat([users,bundles])
        embedding_buffer = [all_embedding]
        for l in range(self.num_layers):
            all_embedding = torch.sparse.mm(self.Graph.t(),all_embedding)
            all_embedding = torch.mul(all_embedding,filter[l].unsqueeze(-1))
            all_embedding = torch.sparse.mm(self.Graph,all_embedding)
            embedding_buffer.append(all_embedding)
        E = torch.stack(embedding_buffer,dim = 1)
        final_embedding = torch.mean(E,dim=1)
        computed_users, computed_bundles = torch.split(final_embedding, [self.num_users, self.num_bundles])
        return computed_users,computed_bundles
    
    def evaluate(self,result, u_idx):
        users, bundles = result
        users = users[u_idx.long()]
        rating = self.f(torch.matmul(users, bundles.t()))
        return rating
        
    def propagate(self,test=True):
        return self._compute()

    def compute_loss_filter(self,rating,user,bundle):
        L2_loss = (1/2)*(torch.linalg.norm(user).pow(2) + torch.linalg.norm(bundle).pow(2))/user.shape[0]
        filter_loss = torch.max(torch.sum(torch.abs(self.filter.weight-1), dim=1))/(self.num_items+self.num_users) # norm 1
        L2_loss = L2_loss+filter_loss
        pos = rating[:, 0]          
        neg = rating[:, 1]
        loss = torch.mean(torch.nn.functional.softplus(neg-pos))
        return loss,L2_loss
    
    def compute_loss(self,rating,user,bundle):
        L2_loss = (1/2)*(torch.linalg.norm(user).pow(2) + torch.linalg.norm(bundle).pow(2))/user.shape[0]
        pos = rating[:, 0]          
        neg = rating[:, 1]
        loss = torch.mean(torch.nn.functional.softplus(neg-pos))
        return loss,L2_loss

    def forward(self,batch):
        users,bundles = self._compute()
        u,b =  batch
        u = users[u].squeeze()
        b = bundles[b]

        rating = torch.bmm(b, u.unsqueeze(-1)).squeeze(-1)
        loss,L2_loss = self.compute_loss(rating,self.embedding_u.weight,self.embedding_b.weight)
        return loss,L2_loss