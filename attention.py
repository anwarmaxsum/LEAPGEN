# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt):
        # print("cek prompt shape before Att")
        # print(prompt.shape)
        # print(self.qkv.weight.shape)
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads
            
            # print("prompt and key for prefix")
            # print(key_prefix.shape)
            # print(value_prefix.shape)

            expected_shape = (B, self.num_heads, C // self.num_heads)
            
            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)
            # print(key_prefix.shape)
        # print(q.shape, k.shape, v.shape)
        # print(k.transpose(-2, -1).shape)

        # exit()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print("attn", attn.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print("x", x.shape)
        # exit()
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class Gen_Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.squeeze(1)
    
        # print("cek x shape before Att")
        # print(x.shape)

        # print(self.qkv.weight.shape)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # exit()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print("attn", attn.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print("x", x.shape)
        # exit()
        x = self.proj(x)
        x = self.proj_drop(x)

        # x = x.unsqueeze(1)
        # print("cek x shape after Att")
        # print(x.shape)

        return x



class Gen_Attention2(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0.0, proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = head_dim ** -1.0
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x = x.squeeze(1)
    
        # print("cek x shape before Att")
        # print(x.shape)

        # print(self.qkv.weight.shape)
        # print(x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # exit()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print("attn", attn.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print("x", x.shape)
        # exit()
        # x = self.proj(x)
        # x = self.proj_drop(x)

        # x = x.unsqueeze(1)
        # print("cek x shape after Att")
        # print(x.shape)

        return x

    # def forward(self, x):
    #     x = x.squeeze(1)

    #     return x

#Simple MLP generator 
# class Gen_Attention2(nn.Module):
#     def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0.0, proj_drop=0.):
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # self.scale = head_dim ** -1.0
#         self.scale = head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         # self.proj = nn.Linear(dim, dim)
#         # self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         x = x.squeeze(1)

#         x = self.qkv(x)
#         x = self.attn_drop(x)

#         return x



# Simple 3 layered MLP generator 
# class Gen_Attention2(nn.Module):
#     def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0.0, proj_drop=0.):
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # self.scale = head_dim ** -1.0
#         self.scale = head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
#         self.qkv2 = nn.Linear(dim, dim, bias=qkv_bias)
#         self.qkv3 = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         # self.proj = nn.Linear(dim, dim)
#         # self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         x = x.squeeze(1)
#         x = self.qkv(x)
#         x = self.qkv2(x)
#         x = self.qkv3(x)
#         x = self.attn_drop(x)

#         return x

# Singe layered LSTM generator    
# class Gen_Attention2(nn.Module):
#     def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0.0, proj_drop=0.):
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # self.scale = head_dim ** -1.0
#         self.scale = head_dim ** -0.5

#         self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        
#         # self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
#         # self.qkv2 = nn.Linear(dim, dim, bias=qkv_bias)
#         # self.qkv3 = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         # self.proj = nn.Linear(dim, dim)
#         # self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         x = x.squeeze(1)
#         # print(x.shape)
#         x, _ = self.lstm(x)
#         # x = self.attn_drop(a)
#         # print(x.shape)
#         return x

# Singe layered  GRU generator    
# class Gen_Attention2(nn.Module):
#     def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0.0, proj_drop=0.):
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # self.scale = head_dim ** -1.0
#         self.scale = head_dim ** -0.5

#         self.gru = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        
#         # self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
#         # self.qkv2 = nn.Linear(dim, dim, bias=qkv_bias)
#         # self.qkv3 = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         # self.proj = nn.Linear(dim, dim)
#         # self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         x = x.squeeze(1)
#         # print(x.shape)
#         x, _ = self.gru(x)
#         # x = self.attn_drop(a)
#         # print(x.shape)
#         return x
   
# class Gen_Attention2(nn.Module):
#     def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0.0, proj_drop=0.):
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # self.scale = head_dim ** -1.0
#         self.scale = head_dim ** -0.5

#         self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
#         self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
#         # self.qkv2 = nn.Linear(dim, dim, bias=qkv_bias)
#         # self.qkv3 = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         # self.proj = nn.Linear(dim, dim)
#         # self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         x = x.squeeze(1)
#         # print(x.shape)
#         x, _ = self.lstm(x)
#         x = self.qkv(x)
#         # x = self.attn_drop(a)
#         # print(x.shape)
#         return x

#    
