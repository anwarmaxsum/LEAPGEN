import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

import copy

from attention import Gen_Attention2

# from kan import *

def get_out_batch(batch_size, task_mean, task_std):
    out = []
    for i in range(batch_size):
        out.append(task_mean + task_std * torch.randn_like(task_mean))
    return torch.stack(out).to(task_mean.device)

class LPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, num_tasks=10, num_classes=100, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=1, top_k_l=3, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,
                 prompts_per_task=5):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.top_k = top_k
        self.top_k_l = top_k_l
        self.n_tasks = num_tasks
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.use_prompt_embed_matcher = True 
        self.prompts_per_task = prompts_per_task
        self.old_num_k = 0
        self.new_num_k = 0
        self.sigmoid = nn.Sigmoid()

        # self.ker_size = kernel_size
        # self.stride = 1
        # self.dilation = 1
        # self.conv_channels = 1 
        self.embed_dim = embed_dim
        self.num_classes =  num_classes
        self.numclasses_per_task = int(self.num_classes/self.n_tasks) 
        self.old_num_c = 0
        self.new_num_c = 0

        self.kmax_list = []
        self.lmax_list = []

        # This is the max number of kernels (self.pool_size) that can occur if we use max number of prompts per task
        self.pool_size = self.n_tasks * prompts_per_task
        print("Num Tasks: ", self.n_tasks, "pool_size: ", self.pool_size, "num_classes: ", num_classes, "top_k: ", top_k)
        # if self.use_prompt_embed_matcher:
        #     self.prompt_embed_matcher = nn.Sequential(OrderedDict([
        #         ('linear1', nn.Linear(embed_dim, embed_dim // 2)),
        #         ('relu1', nn.ReLU()),
        #         ('linear2', nn.Linear(embed_dim // 2, embed_dim // 4))
        #     ]))
        self.k_comp_gen = nn.ModuleDict()
        for i in range(self.num_layers):
            self.k_comp_gen[str(i)] = nn.ModuleDict()
            for j in range(self.num_heads):
                self.k_comp_gen[str(i)][str(j)] = nn.ModuleList()
                for k in range(self.top_k_l):
                    # k_comp = Gen_Attention2(int(embed_dim/self.num_heads), 1, False, 0., 0.)
                    k_comp = nn.Conv1d(1, 1, 3, stride=1, padding=1)
                    self.k_comp_gen[str(i)][str(j)].append(k_comp)
    
        

        self.v_comp_gen = nn.ModuleDict()
        for i in range(self.num_layers):
            self.v_comp_gen[str(i)] = nn.ModuleDict()
            for j in range(self.num_heads):
                self.v_comp_gen[str(i)][str(j)] = nn.ModuleList()
                for k in range(self.top_k_l):
                    # v_comp = Gen_Attention2(int(embed_dim/self.num_heads), 1, False, 0., 0.)
                    v_comp = nn.Conv1d(1, 1, 3, stride=1, padding=1)
                    self.v_comp_gen[str(i)][str(j)].append(v_comp)

        
        key_shape = (self.num_classes, embed_dim)
        key2_shape = (self.n_tasks, embed_dim)
        if prompt_key_init == 'zero':
            self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            self.prompt_key2 = nn.Parameter(torch.zeros(key2_shape))
        elif prompt_key_init == 'uniform':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            self.prompt_key2 = nn.Parameter(torch.randn(key2_shape))
            nn.init.uniform_(self.prompt_key, -1, 1)
            nn.init.uniform_(self.prompt_key2, -1, 1)
        elif prompt_key_init == 'ortho':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            self.prompt_key2 = nn.Parameter(torch.randn(key2_shape))
            nn.init.orthogonal_(self.prompt_key)
            nn.init.orthogonal_(self.prompt_key2)

        print(self.k_comp_gen[str(0)][str(1)])
        print("Init prompt finished")
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def process_new_task(self, old_num_k, new_num_k, new_desc_embed):
        self.old_num_k = old_num_k
        self.new_num_k = new_num_k

        print("Old Num K: ", self.old_num_k, "New Num K: ", self.new_num_k)
        self.kmax_list.append(new_num_k)


        self.desc_embed = new_desc_embed
        self.old_num_c = self.new_num_c
        self.new_num_c = self.new_num_c + int(self.num_classes / self.n_tasks)
        self.lmax_list.append(self.new_num_c)
        print(self.kmax_list)
        print(self.lmax_list)
        # self.prompt_key = self.gram_schmidt(self.prompt_key)

        # if old_num_k > 0:
        #     for i in range(old_num_k):
        #         for n, p in self.prompt_key[i].named_parameters():
        #             p.requires_grad = False

        if self.old_num_c > 0:
            for i in range(self.num_layers):
                for j in range(self.num_heads):
                    # for k in range(int(self.num_classes / self.n_tasks)):
                    # for k in range(self.top_k_l):
                    for k in range(self.old_num_k-self.top_k_l,self.old_num_k):
                        # k_comp = Gen_Attention2(int(self.embed_dim/self.num_heads), 1, False, 0., 0.).cuda()
                        k_comp = nn.Conv1d(1, 1, 3, stride=1, padding=1).cuda()
                        # k_comp = copy.deepcopy(self.k_comp_gen[str(i)][str(j)][k])
                        self.k_comp_gen[str(i)][str(j)].append(k_comp)
                        # v_comp = Gen_Attention2(int(self.embed_dim/self.num_heads), 1, False, 0., 0.).cuda()
                        v_comp = nn.Conv1d(1, 1, 3, stride=1, padding=1).cuda()
                        # v_comp = copy.deepcopy(self.v_comp_gen[str(i)][str(j)][k])
                        self.v_comp_gen[str(i)][str(j)].append(v_comp)



            for i in range(self.num_layers):
                for j in range(self.num_heads):
                    # for k in range(self.old_num_c):
                    for k in range(self.old_num_k):
                        gen = self.k_comp_gen[str(i)][str(j)][k]
                        for n, p in gen.named_parameters():
                            p.requires_grad = False
                        gen = self.v_comp_gen[str(i)][str(j)][k]
                        for n, p in gen.named_parameters():
                            p.requires_grad = False

    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point

        s = self.old_num_k
        f = self.new_num_k
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_embed, y, task_id=-1, prompt_mask=None, layer_num= -1, cls_features=None):
        # if y is None:
        #     print("check 0")
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            s = self.old_num_k
            f = self.new_num_k

            prompt_key2_norm = self.l2_normalize(self.prompt_key2[0:len(self.kmax_list)], dim=-1)
            prompt_key_norm = self.l2_normalize(self.prompt_key[0:self.lmax_list[-1]], dim=-1)
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) # B, C
            
            prompt_key2_norm_allclass = torch.flatten((prompt_key2_norm.unsqueeze(1).repeat(1,self.numclasses_per_task,1)), start_dim=0, end_dim=1)
            # print("prompt_key2_norm_allclass shape: ",prompt_key2_norm_allclass.shape)

            sim = torch.matmul(prompt_key_norm, x_embed_norm.t())
            sim = sim.t()
            sim2 = torch.matmul(prompt_key2_norm_allclass, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
            sim2 = sim2.t() # B, pool_size
            sim2 = (1000*sim) * (1000*sim2)
            
            # sim2 = torch.matmul(prompt_key2_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
            # sim2 = sim2.t()

            # print(sim.shape)
            # print(sim2.shape)
            (sim2_top_k, idx2) = torch.topk(sim2, k=1, dim=1) # B, top_k
            idx2 = torch.floor(idx2/self.numclasses_per_task).long()
            # print(idx2.shape, idx2.flatten())

            if self.training:
                idx2[0:] = task_id 
                pred_task_id = task_id
            else:
                pred_task_id = torch.mode(idx2.detach().clone().flatten().cpu()).values.item()
                print("pred task_id: " + str(pred_task_id))
                # print(pred_task_id)

            batched_key2_norm = prompt_key2_norm[idx2]
            sim2 = batched_key2_norm * x_embed_norm.unsqueeze(1) # B, top_k, C
            reduce_sim = torch.sum(sim2) / x_embed.shape[0] # Scalar
            out['reduce_sim2'] = reduce_sim
           
            

            f = self.kmax_list[pred_task_id]
            fl = self.lmax_list[pred_task_id]
            if task_id == 0 or pred_task_id==0:
                sl = 0
                s = 0
                f = self.top_k_l
            else:
                sl = self.lmax_list[pred_task_id-1]
                s = self.kmax_list[pred_task_id-1]
                f = self.kmax_list[pred_task_id]
            out['max_t'] = pred_task_id+1


            if self.training:
                prompt_key = self.prompt_key[sl:fl]
                # prompt_key = self.desc_embed[sl:fl]
            else:
                prompt_key = self.prompt_key[0:fl]
                # prompt_key = self.desc_embed[0:fl]

            prompt_key_norm = self.l2_normalize(prompt_key, dim=-1) # Pool_size, C

            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
            similarity = similarity.t() # B, pool_size
            # similarity
            out['similarity'] = similarity
            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k_l, dim=1)

            batched_prompt_raw = torch.matmul(similarity_top_k.unsqueeze(1), self.desc_embed[idx])
            out['desc_embed'] = batched_prompt_raw.squeeze(1).detach().clone()

            if self.training:
                pkey_norm = self.l2_normalize(self.prompt_key, dim=-1) 
                batched_key_norm = pkey_norm[y]
                sim = batched_key_norm * x_embed_norm # B, top_k, C
                reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
                out['reduce_sim'] = reduce_sim

            if self.use_prefix_tune_for_e_prompt:
                batched_prompt_raw = F.interpolate(batched_prompt_raw, [self.embed_dim // self.num_heads])
                batched_prompt_raw = batched_prompt_raw.repeat(1,self.num_heads,1)
                batched_prompt_raw = batched_prompt_raw.view(x_embed.shape[0],self.num_heads, self.embed_dim // self.num_heads).unsqueeze(0)
                batched_prompt_raw = torch.cat((batched_prompt_raw,batched_prompt_raw),dim=0) # 2 x B x n_head x dim/n_head

            else:
                batched_prompt_raw = F.interpolate(batched_prompt_raw, [self.embed_dim // self.num_heads])
                batched_prompt_raw = batched_prompt_raw.repeat(1,self.num_heads,1)
                batched_prompt_raw = batched_prompt_raw.view(x_embed.shape[0],self.num_heads, self.embed_dim // self.num_heads)
                batched_prompt_raw = torch.cat((batched_prompt_raw,batched_prompt_raw),dim=0)

            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm


        similarity_new  = torch.zeros_like(similarity,device=similarity.device)
        for c in range(idx.shape[1]):
            similarity_new[:,idx[:,c]] = similarity_top_k[:,c]
        
       
        batched_prompt = self.compute_att_over_prompt(batched_prompt_raw, s,f, layer_num, similarity_new)
        out['batched_prompt'] = batched_prompt

        return out


    def ortho_penalty(self, t):
        return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean() * 1e-6


    def compute_att_over_prompt(self, batched_prompt, s, f, layer_num, similarity):
        #  batch_size, dual, length // dual, self.num_heads, embed_dim // self.num_heads
        # batched_prompt = batched_prompt.permute(1, 3, 0, 2, 4) #  dual, num_heads, B, length, head_dim
        k_prompt_list = []
        v_prompt_list = []
        
        k_prompt_layer = batched_prompt[0] # B, num_heads, head_dim
        v_prompt_layer = batched_prompt[1] # B, num_heads, head_dim
        # k_prompt_layer = k_prompt_layer.unsqueeze(0).repeat(self.length,1,1,1)  # length, B, num_heads, head_dim
        # v_prompt_layer = v_prompt_layer.unsqueeze(0).repeat(self.length,1,1,1)  # length, B, num_heads, head_dim
        # k_prompt_layer = k_prompt_layer.permute(2, 1, 0, 3) # num_heads, B,length, head_dim
        # v_prompt_layer = v_prompt_layer.permute(2, 1, 0, 3) # num_heads, B,length, head_dim
        k_prompt_layer = k_prompt_layer.permute(1, 0, 2) # num_heads, B,head_dim
        v_prompt_layer = v_prompt_layer.permute(1, 0, 2) # num_heads, B,head_dim

        # n_heads, batch_size, length,  head_dim = k_prompt_layer.shape
        n_heads, batch_size, head_dim = k_prompt_layer.shape

        new_k_prompt_layer = torch.zeros((n_heads, batch_size, head_dim), device=k_prompt_layer.device)
        new_v_prompt_layer = torch.zeros((n_heads, batch_size, head_dim), device=v_prompt_layer.device)
        
        for h in range(self.num_heads):
            k_comp_gen = self.k_comp_gen[str(layer_num)][str(h)]
            v_comp_gen = self.v_comp_gen[str(layer_num)][str(h)]
            
            # k_prompt_head = k_prompt_layer[h].unsqueeze(1) #  B, 1, length, head_dim
            # v_prompt_head = v_prompt_layer[h].unsqueeze(1) # B, 1, length, head_dim
            k_prompt_head = k_prompt_layer[h].unsqueeze(1) #  B, 1, head_dim
            v_prompt_head = v_prompt_layer[h].unsqueeze(1) # B, 1, head_dim

            for p in range(s,f):
                k_comp_val = k_comp_gen[p]
                v_comp_val = v_comp_gen[p]

                # new_k_prompt_layer[h] += k_comp_val(k_prompt_head).squeeze(1) * similarity[:, p-s].unsqueeze(1).unsqueeze(2)
                # new_v_prompt_layer[h] += v_comp_val(v_prompt_head).squeeze(1)  * similarity[:, p-s].unsqueeze(1).unsqueeze(2)
                # print("check shape")
                # print(new_k_prompt_layer[h].shape)
                # print(k_prompt_head.shape)
                # print(k_comp_val(k_prompt_head).shape)
                # print(similarity[:, p-s].shape)

                new_k_prompt_layer[h] += k_comp_val(k_prompt_head).squeeze(1) * similarity[:, p-s].unsqueeze(1)
                new_v_prompt_layer[h] += v_comp_val(v_prompt_head).squeeze(1) * similarity[:, p-s].unsqueeze(1)
         
        new_batched_prompt = torch.stack([new_k_prompt_layer, new_v_prompt_layer], dim=0) # dual, num_heads, B, head_dim
        new_batched_prompt = new_batched_prompt.unsqueeze(3).repeat(1,1,1,self.length,1)        # dual, num_heads, B, length, head_dim
        new_batched_prompt = new_batched_prompt.permute(2, 0, 3, 1, 4) # B, dual, length, num_heads, head_dim

        return new_batched_prompt


    
    def conv_ortho(self, weights):
        w = weights
        in_channels, out_channels, kernel_size, kernel_size = w.shape
        w =w.permute(1, 0, 2, 3). view(out_channels, -1)
        W1 = w.t()
        Ident = torch.eye(w.shape[1]).to(w.device)
        # print("W1 shape: ", W1.shape, w.shape)
        W_new = torch.matmul(W1, w)
        Norm = W_new - Ident
        b_k = torch.rand(Norm.shape[1]).to(Norm.device)
        b_k = b_k.unsqueeze(1)
        v1 = torch.matmul(Norm, b_k)
        norm1 = torch.sum(torch.square(v1))**0.5
        v2 = v1 / norm1
        v3 = torch.matmul(Norm, v2)

        return 0.01*(torch.sum(torch.square(v3))**0.5) + (1e-4)*(torch.sum(torch.square(w))**0.5)

    def conv_orthogonality(self, conv_vals):
        ortho_norm = 0
        for i in range(self.num_layers):

            for h in range(self.num_heads):

                conv_val = conv_vals[str(i)][str(h)]
                for j in range(len(conv_val)):
                    ortho_norm += self.conv_ortho(conv_val[j].weight)
        return ortho_norm
