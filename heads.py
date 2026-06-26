# For Temporal: Reshape input to (Batch * Patches, Frames, Channels).
# For Spatial: Reshape input to (Batch * Frames, Patches, Channels).

import torch
from torch import nn

class TemporalAttention(nn.Module):
    def __init__(self, dim,qkv_bias=False,num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Standard QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input x shape: (B * N, T, C) 
        Where B=Batch, N=Number of Patches, T=Frames, C=Channels
        """
        B_N, T, C = x.shape
        
        # 1. Generate Q, K, V
        # Shape: (B*N, T, 3, heads, head_dim)
        qkv = self.qkv(x).reshape(B_N, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # Individual shapes: (B*N, heads, T, head_dim)

        # 2. Scaled Dot-Product Attention
        # Result: (B*N, heads, T, T)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3. Context Vector
        # Shape: (B*N, T, C)
        x = (attn @ v).transpose(1, 2).reshape(B_N, T, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self,dim,qkv_bias=False,num_heads=8,dropout=0.1):
        super().__init__()
        self.num_heads=num_heads
        self.scale=(dim//num_heads)**-0.5
        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop=nn.Dropout(dropout)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(dropout)
    def forward(self,x):
        """
        Input shape: (B*F,P,C)
        """
        B_N,P,C=x.shape
        qkv=self.qkv(x).reshape(B_N,P,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]        
        attn=(q@k.transpose(-2,-1))*self.scale
        attn=attn.softmax(dim=-1)
        attn=self.attn_drop(attn)
        x=(attn@v).transpose(1,2).reshape(B_N,P,C)
        x=self.proj(x)
        x=self.proj_drop(x)
        return x
                