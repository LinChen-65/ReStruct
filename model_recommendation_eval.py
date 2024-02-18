import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb

class Op(nn.Module):

    def __init__(self):
        super(Op, self).__init__()
    
    def forward(self, x, adjs, idx):
        return torch.spmm(adjs[idx], x)

class Cell(nn.Module):

    def __init__(self, n_step, n_hid_prev, n_hid, use_norm = True, use_nl = True):
        super(Cell, self).__init__()
        
        self.affine = nn.Linear(n_hid_prev, n_hid) 
        self.n_step = n_step # n_step = 4 or 6
        self.norm = nn.LayerNorm(n_hid) if use_norm is True else lambda x : x
        self.use_nl = use_nl          
        self.ops_seq = nn.ModuleList()
        self.ops_res = nn.ModuleList()
        for i in range(self.n_step):
            self.ops_seq.append(Op())
        for i in range(1, self.n_step): 
            for j in range(i):
                self.ops_res.append(Op())
    
    def forward(self, x, adjs, idxes_seq, idxes_res, connection_dict):
        
        x = self.affine(x)
        states = [x]
        temp_states = []
        offset = 0
        for i in range(self.n_step): # 4 or 6
            #seqi = self.ops_seq[i](states[i], adjs[:-1], idxes_seq[i]) #! exclude zero Op
            seqi = self.ops_seq[i](states[i], adjs, idxes_seq[i])
            try:
                resi = sum(self.ops_res[offset + j](h, adjs, idxes_res[offset + j]) for j, h in enumerate(states[:i]))
            except:
                pdb.set_trace()
            #for j, h in enumerate(states[:i]):
            #    resi = sum(self.ops_res[offset + j](h, adjs, idxes_res[offset + j]))
            offset += i
            states.append(seqi + resi) # 融合同一个state的信息
            temp_states.append(seqi)
            
            # Fusion and update
            #for j in range(self.n_step):
            #    connected_states = connection_dict[idxes_seq[j]]
            #    for k in range(connected_states):
     
        #assert(offset == len(self.ops_res))

        output = self.norm(states[-1])
        if self.use_nl:
            output = F.gelu(output)
        return output


class Model(nn.Module):

    def __init__(self, in_dims, n_hid, n_steps, dropout = None, single=False, attn_dim = 64, num_heads=8, use_norm = True, out_nl = True):
        # n_steps = steps_s = [4] 或 steps_t = [6]
        super(Model, self).__init__()
        self.n_hid = n_hid
        self.single = single
        self.ws = nn.ModuleList()
        assert(isinstance(in_dims, list))
        for i in range(len(in_dims)):
            self.ws.append(nn.Linear(in_dims[i], n_hid))
        assert(isinstance(n_steps, list))
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)): # DiffMG里len(n_steps)=1
            self.metas.append(Cell(n_steps[i], n_hid, n_hid, use_norm = use_norm, use_nl = out_nl)) # n_steps[0]=4 or 6 
        
        #* [Optional] Combine more than one meta graph?
        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)
        self.fc1 = nn.Linear(n_hid, attn_dim)
        self.fc2 = nn.Linear(n_hid, attn_dim)
        self.fc3 = nn.Linear(n_hid, attn_dim)

        self.feats_drop = nn.Dropout(dropout) if dropout is not None else lambda x : x
        self.multihead_attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res, connection_dict): 
        # idxes_seq = archs[args.dataset]["source"][0], example: [[9,1,4,3]]
        # idxes_res = archs[args.dataset]["source"][1], example: [[10,10,10,10,10,10]] (both "source" can be replaced by target)
        hid = torch.zeros((node_types.size(0), self.n_hid)).cuda()
        for i in range(len(node_feats)):
            hid[node_types == i] = self.ws[i](node_feats[i])
        hid = self.feats_drop(hid)
        temps = []
        attns = []
        for i, meta in enumerate(self.metas): # DiffMG中i只取0
            hidi = meta(hid, adjs, idxes_seq[i], idxes_res[i], connection_dict)  # idxes_seq[0] = [9,1,4,3], idxes_res[0] = [10,10,10,10,10,10]
            temps.append(hidi)
            if(self.single):
                attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1])))
                attns.append(attni)
        
        if(self.single):
            hids = torch.stack(temps, dim=0).transpose(0, 1)
            attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)
            output = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)
        else:
            hids = torch.stack(temps, dim=0).transpose(0, 1)
            hids_query = self.fc1(hids)
            hids_key = self.fc2(hids)
            hids_value = self.fc3(hids)
            #attn_output, _ = self.multihead_attn(query=hids, key=hids, value=hids)
            attn_output, _ = self.multihead_attn(query=hids_query, key=hids_key, value=hids_value)
            output = attn_output.sum(dim=1)
        return output # torch.Size([31092, 64])