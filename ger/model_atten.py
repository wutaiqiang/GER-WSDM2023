import os
from tabnanny import verbose
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import torch.distributed as dist
from dgl.nn import GATConv
import dgl

class GAT(nn.Module):
    def __init__(self,in_dim,out_dim,num_heads=8,n_layers=2,activation=F.relu,dropout=0.1,feat_drop=0.0,attn_drop=0, node_drop=0,):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.node_drop_rate = nn.Dropout2d(p=node_drop)
        self.layers.append(
            GATConv(in_dim, out_dim, 
                num_heads=num_heads,feat_drop=feat_drop,
                attn_drop=attn_drop,allow_zero_in_degree=True
                )
            ) #in [56,768]-->[56,heads,768]
        for i in range(n_layers-1):
            # https://github.com/dmlc/dgl/blob/5798ee8d989e804be9b8dd6fcdd0c76f67339180/python/dgl/nn/pytorch/conv/gatconv.py#L13
            self.layers.append(
                GATConv(in_dim*num_heads, out_dim, 
                    num_heads=num_heads,feat_drop=feat_drop,
                    attn_drop=attn_drop,allow_zero_in_degree=True
                    )
            )
        
    def forward(self, graph, h):
        h = self.dropout(h)
        h = self.node_drop_rate(h.unsqueeze(0)).squeeze(0) #drop node
        for l in range(self.n_layers-1):
            h1 = self.layers[l](graph, h).flatten(1) #self.layers[l](graph, h) --> [56,heads,768], flatten--> 56,heads*768
            h = self.activation(h1)
            h = self.dropout(h)
        h, atten = self.layers[-1](graph,h,  get_attention=True) #layer以后[56,heads,768]-->[56,768]
        return h.mean(1), atten

def gen_pos_mask(token_idx, typea='mention'):
    #return None
    mask = torch.zeros_like(token_idx).float()
    if typea == 'mention':
        for index1,tokens in enumerate(token_idx):
            flag = False
            for index,token in enumerate(tokens):
                if token.item() == 2: # unused 1
                    flag = False
                    break
                if flag:
                    mask[index1][index] = 1 # # unused 0
                if token.item() == 1:
                    flag = True
    else:
        for index1,tokens in enumerate(token_idx):
            flag = False
            for index,token in enumerate(tokens):
                if token.item() == 3: # unused 2
                    flag = False
                    break
                if flag:
                    mask[index1][index] = 1
                if token.item() == 101: #[CLS]
                    flag = True
    #mask = torch.div(mask,torch.sum(mask,dim=-1,keepdim=True)) 为了避免除0,这里不平均
    return mask


class BiEncoder(nn.Module):
    def __init__(self, args):
        super(BiEncoder, self).__init__()
        self.args = args 
        self.ctx_encoder = BertModel.from_pretrained(args.pretrained_model,output_attentions = True, return_dict = True, cache_dir=args.cache_dir)#, output_attentions = True)
        self.ent_encoder = BertModel.from_pretrained(args.pretrained_model,output_attentions = True, return_dict = True, cache_dir=args.cache_dir)#, output_attentions = True)

        bert_output_dim = self.ctx_encoder.embeddings.word_embeddings.weight.size(1)
        #! Graph
        if args.graph:
            self.GNN_m = GAT(in_dim=bert_output_dim, out_dim=bert_output_dim, \
                n_layers=args.gnn_layers, feat_drop=args.feat_drop, node_drop=args.node_drop)
            self.mu_m = nn.Parameter(torch.FloatTensor([args.mu]),requires_grad=True)

            self.GNN_e = GAT(in_dim=bert_output_dim, out_dim=bert_output_dim, \
                n_layers=args.gnn_layers, feat_drop=args.feat_drop, node_drop=args.node_drop)
            self.mu_e = nn.Parameter(torch.FloatTensor([args.mu]),requires_grad=True)

            self.tri_m = nn.Linear(3*bert_output_dim, bert_output_dim) # for triplet node
            self.gra_m = nn.Linear(bert_output_dim, bert_output_dim) # for graph output
        
            self.tri_e = nn.Linear(3*bert_output_dim, bert_output_dim) # for triplet node
            self.gra_e = nn.Linear(bert_output_dim, bert_output_dim)


    def to_bert_input(self, input_ids):
        attention_mask = 1 - (input_ids == 0).long()
        token_type_ids = torch.zeros_like(input_ids).long()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
    
    def encode(self, ids, node_mask, adjs, linear1, linear2, GNN, mu, type="mention"):
        # Bert output
        # https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
        bert_output = self.ctx_encoder(**self.to_bert_input(ids))
        key_mask = gen_pos_mask(ids, typea=type)

        vector_all = bert_output.last_hidden_state # batch_size X 128 X 768
        #! return [CLS] 
        if self.args.return_type == "bert_only":
            return vector_all[:,0,:]
        
        #! return mean of all words
        if self.args.return_type == "bert_only_mean":
            return vector_all.mean(dim=1,keepdim=False)
        
        #! mean of node
        if self.args.return_type == "node_mean_only":
            mask = torch.div(key_mask,torch.sum(key_mask,dim=-1,keepdim=True)+1e-5)
            return torch.bmm(mask.unsqueeze(1),vector_all).squeeze(1)

        #! mean of node + CLS
        if self.args.return_type == "node_mean_add":
            mask = torch.div(key_mask,torch.sum(key_mask,dim=-1,keepdim=True)+1e-5)
            return vector_all[:,0,:] + mu*torch.bmm(mask.unsqueeze(1),vector_all).squeeze(1)
        
        #! max of node
        if self.args.return_type == "node_max_only":
            vec = []
            for i in range(vector_all.size(0)):
                indice = torch.nonzero(key_mask[i])
                if indice.sum().item() == 0:
                    vec.append(vector_all[i][0].squeeze())
                else:
                    xx = torch.max(vector_all[i].index_select(dim=0,index=indice.squeeze()),dim=0)[0]
                    vec.append(xx.squeeze())
            return torch.stack(vec,dim=0)
        
        #! max of node + CLS
        if self.args.return_type == "node_max_add":
            vec = []
            for i in range(vector_all.size(0)):
                indice = torch.nonzero(key_mask[i])
                if indice.sum().item() == 0:
                    vec.append(vector_all[i][0].squeeze())
                else:
                    xx = torch.max(vector_all[i].index_select(dim=0,index=indice.squeeze()),dim=0)[0]
                    vec.append(xx.squeeze())
            return vector_all[:,0,:] + mu*torch.stack(vec,dim=0)
        
        dim = ids.size(-1)

        # get node vector
        node_mask_flag = torch.zeros([node_mask.size(0),node_mask.size(1),dim]).to(vector_all.device) # batch_size X node_lens X 128
        for i in range(node_mask.size(0)):
            for j in range(node_mask.size(1)):
                if node_mask[i,j,0].item() != -1:
                    start = min(node_mask[i,j][0].item(),dim)
                    end   = min(node_mask[i,j][1].item()+1,dim)
                    if end > start:
                        node_mask_flag[i,j,start:end] = 1/(end-start)
        node_vecs = node_mask_flag.bmm(vector_all) # vector for all node, batch X node_lens X 768

        #! 将对应节点的向量取出，直接mean
        if self.args.return_type == "linear_attention":
            node_lens = (node_vecs.sum(dim=-1)!=0).float().sum(dim=-1, keepdim=True)
            vec_m = node_vecs.sum(dim=1)/node_lens
            return vector_all[:, 0, :] + mu*vec_m
        
        
        # bulid graph
        g_nodes = []
        for index,adj in enumerate(adjs):
            if adj.size(0) == 0: #! 有些句子没有抽到图
                g_n = dgl.graph((torch.tensor([0]).to(adj.device),torch.tensor([0]).to(adj.device)))
                g_n.ndata["ft"] = node_vecs[index,0,:]
                g_nodes.append(g_n)
            else:
                tri_num = 0
                for pair in adj:
                    if pair[0].item()==-1: # padded edges
                        break
                    tri_num += 1
                
                adj = adj[:tri_num].t()
                # Triple_Node
                if tri_num == 0:
                    cur_node = 0
                else:
                    cur_node =max([max(i) for i in adj])

                if self.args.return_type == "gat":
                    tri = torch.zeros_like(adj[0]).to(adj.device) #triplet_node
                    s = torch.cat(
                            (torch.tensor([0]).to(adj.device),adj[0],adj[1],adj[2],adj[1],adj[2],adj[0],   tri,   tri,   tri,adj[0],adj[1],adj[2],),
                            dim=-1)
                    t = torch.cat(
                            (torch.tensor([0]).to(adj.device),adj[1],adj[2],adj[0],adj[0],adj[1],adj[2],adj[0],adj[1],adj[2],   tri,   tri,   tri,),
                            dim=-1)
                    g_n = dgl.graph((s.to(torch.int32),t.to(torch.int32)))
                    g_n = dgl.add_self_loop(g_n) # 自环
                    g_n = g_n.cpu().to_simple().to(s.device) #remove abudunt edge
                    # Node vec
                    node_emb = node_vecs[index,:cur_node+1,:]
                    g_n.ndata['ft'] = node_emb #赋予节点的embbedding
                elif self.args.return_type == "hgat":
                    tri = torch.tensor(list(range(cur_node+1,cur_node+1+tri_num))).to(adj.device)
                    #s = torch.cat(
                    #    (torch.tensor([0]).to(adj.device),torch.zeros_like(tri),tri,                  adj[0],adj[1],adj[2],adj[0],adj[1],adj[2],adj[1],adj[2],adj[0],tri,   tri,   tri,   ),
                    #    dim=-1)
                    #t = torch.cat(
                    #    (torch.tensor([0]).to(adj.device),tri,                  torch.zeros_like(tri),adj[1],adj[2],adj[0],tri,   tri,   tri,   adj[0],adj[1],adj[2],adj[0],adj[1],adj[2],),
                    #    dim=-1)
                    s = torch.cat(
                        (torch.tensor([0]).to(adj.device),torch.zeros_like(tri),tri,                 adj[0],adj[1],adj[2],tri,   tri,   tri,   ),
                        dim=-1)
                    t = torch.cat(
                        (torch.tensor([0]).to(adj.device),tri,                  torch.zeros_like(tri),tri,   tri,   tri,   adj[0],adj[1],adj[2], ),
                       dim=-1)
                    g_n = dgl.graph((s.to(torch.int32),t.to(torch.int32)))
                    g_n = dgl.add_self_loop(g_n) # 自环
                    g_n = g_n.cpu().to_simple().to(s.device) #remove abudunt edge
                    # Node vec
                    node_emb = node_vecs[index,:cur_node+1,:]
                    # Triple Vec
                    S_vec = node_emb.index_select(0,adj[0])
                    P_vec = node_emb.index_select(0,adj[1])
                    O_vec = node_emb.index_select(0,adj[2])
                    tri_vec = linear1(torch.cat((S_vec,P_vec,O_vec),dim=-1))

                    g_n.ndata['ft'] = torch.cat((node_emb,tri_vec),dim=0) #赋予节点的embbedding
                else:
                    print("wrong return type!")
                # batch
                g_nodes.append(g_n)
        
        # batch graph
        bgn = dgl.batch(g_nodes)
        bgn.ndata['ft'], atten = GNN(bgn, bgn.ndata['ft']) # GNN Forward
        # atten = atten.mean(1).squeeze()

        # src, dst = bgn.edges()[0], bgn.edges()[1]
        # l = 0 # num for zero node
        # for idx in src:
        #     if idx.item() == 0:
        #         l += 1
        #     else:
        #         break
        # central_score = atten[:l]
        # max_score_triplet = dst[central_score.argmax()]
        # picked_node = dst[src==max_score_triplet][1:-1] # drop node 0, drop node self
        # res = []
        # for node in picked_node:
        #     pos = node_mask[0,node,:]
        #     res.append(ids[0,pos[0]:pos[1]+1])

        node_embs = []
        for index,gs in enumerate(dgl.unbatch(bgn)): # unbatch
            node_embs.append(gs.ndata['ft'][0,:]) #取Global节点

        node_embs = torch.stack(node_embs,dim=0)

        #return node_embs #! graph only
        mask = torch.div(key_mask,torch.sum(key_mask,dim=-1,keepdim=True)+1e-5)

        return vector_all[:,0,:], mu*linear2(node_embs), torch.bmm(mask.unsqueeze(1),vector_all).squeeze(1)

    def encode_mention(self, mention):
        ids, node_mask, adjs = mention["ids"], mention["node_mask"], mention["rels"]
        return self.encode(ids, node_mask, adjs, self.tri_m, self.gra_m, self.GNN_m, self.mu_m, type="mention")
    
    def encode_entity(self, entity):
        ids, node_mask, adjs = entity["ids"], entity["node_mask"], entity["rels"]
        return self.encode(ids, node_mask, adjs, self.tri_e, self.gra_e, self.GNN_e, self.mu_e, type="entity")
    
    def score_candidates(self, mention, ctx_world, candidate_pool = None):
        ctx_output = self.encode_mention(mention).cpu().detach()
        res = []
        for world, ctx_repr in zip(ctx_world, ctx_output):
            ctx_repr = ctx_repr.to(candidate_pool[world].device)
            res.append(ctx_repr.unsqueeze(0).mm(candidate_pool[world].T).squeeze(0))
        return res

    def forward(self, mention, entity):
        ctx_output = self.encode_mention(mention)
        ent_output = self.encode_entity(entity)
        return ctx_output.contiguous(), ent_output.contiguous()
    

class NCE_Random(nn.Module):
    def __init__(self, num_gpus, dual_loss=False):
        super(NCE_Random, self).__init__()
        self.num_gpus = num_gpus
        self.dual_loss = dual_loss

    def forward(self, ctx_output, ent_output):
        
        if self.num_gpus > 1:
            ctx_output = torch.cat(GatherLayer.apply(ctx_output), dim=0)
            ent_output = torch.cat(GatherLayer.apply(ent_output), dim=0)

        score = torch.matmul(ent_output, ctx_output.T) 
        
        target = torch.arange(score.size(0)).to(ctx_output.device)
        loss = F.cross_entropy(score, target, reduction="mean")
        if self.dual_loss: # sysmetric loss
            loss += F.cross_entropy(score.T, target, reduction="mean")

        predict = torch.max(score, -1).indices
        acc = sum(predict == target) * 1.0 / score.size(0)

        return loss, acc, score



class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
