from torch import nn
from heads import SpatialAttention,TemporalAttention
import torch
class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_frames,num_patches,num_heads=6,dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.num_frames=num_frames
        self.num_patches=num_patches+1
        self.layerNorm1=nn.LayerNorm(d_model)
        self.layerNorm2=nn.LayerNorm(d_model)
        self.layerNorm3=nn.LayerNorm(d_model)
        self.spatialHead=SpatialAttention(d_model,qkv_bias=False,num_heads=num_heads,dropout=dropout)
        self.temporalHead=TemporalAttention(d_model,qkv_bias=False,num_heads=num_heads,dropout=dropout)
        
        self.ffn=nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model,d_model),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        BF,P,C=x.shape
        B=BF//self.num_frames
        F=self.num_frames
        res=x
        x=x.view(B,F,P,C).permute(0,2,1,3).reshape(B*P,F,C)
        x=self.temporalHead(x)
        x=x.view(B,P,F,C).permute(0,2,1,3).reshape(BF,P,C)
        x=x+res
        res=x
        x=self.layerNorm2(x)
        x=self.spatialHead(x)
        x=x+res
        res=x
        x=self.layerNorm3(x)
        x=self.ffn(x)
        x=x+res
        return x
class fTSA(nn.Module):
    def __init__(self, kernel, n_channels, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.scale=out_dim**-0.5
        self.proj_0 = nn.Conv2d(n_channels, out_dim, kernel_size=kernel, stride=kernel)
        self.proj_1 = nn.Conv2d(n_channels, out_dim, kernel_size=kernel, stride=kernel)
        self.proj_2 = nn.Conv2d(n_channels, out_dim, kernel_size=kernel, stride=kernel)
        self.flatten = nn.Flatten(start_dim=2)
        self.BN = nn.BatchNorm1d(num_features=out_dim)
        
    def forward(self, Xl_2, Xl_1, Xl_0):
        """
        Input shape expected: (Batch, H, W, C)
        """
        Xl_0 = Xl_0.permute(0, 3, 1, 2)
        Xl_1 = Xl_1.permute(0, 3, 1, 2)
        Xl_2 = Xl_2.permute(0, 3, 1, 2)
        
        y_0 = self.proj_0(Xl_0)
        y_1 = self.proj_1(Xl_1)
        y_2 = self.proj_2(Xl_2)
        
        y_0 = self.flatten(y_0)  
        y_1 = self.flatten(y_1)
        y_2 = self.flatten(y_2)
        
        y_0 = y_0.transpose(1, 2)
        y_1 = y_1.transpose(1, 2)
        y_2 = y_2.transpose(1, 2)
        
        attn_scores = torch.matmul(y_1, y_2.transpose(-1, -2))*self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        res = torch.matmul(attn_weights, y_0)
        
        res = res.transpose(1, 2)
        
        return self.BN(res)

class fTCA(nn.Module):
    def __init__(self,n_channels,out_dim):
        super().__init__()
        self.g=nn.Sequential(
            nn.Linear(n_channels,2*n_channels),
            nn.GELU(),
            nn.Linear(2*n_channels,out_dim),
            nn.Sigmoid()
        )
    def forward(self,Xl_2,Xl_1,Xl_0):
        diff1=Xl_0-Xl_1
        diff2=Xl_0-Xl_2
        GAP1=torch.mean(diff1,dim=(1,2))
        GAP2=torch.mean(diff2,dim=(1,2))
        term1=self.g(GAP1)*0.5
        term2=self.g(GAP2)*0.5
        return term1+term2

class temporalAttentionPooling(nn.Module):
    def __init__(self,channels,d_model,kernel,iterations=3):
        super().__init__()
        self.kernel=kernel
        self.iter=iterations
        self.fTCA=fTCA(channels,out_dim=d_model)
        self.fTSA=fTSA(kernel,channels,d_model)
        self.TC=nn.Conv1d(d_model,d_model,kernel,padding=kernel//2)
    def calculateXlHat(self,x):
        x0=x[:,2:]
        x1=x[:,1:-1]
        x2=x[:,:-2]
        B,L,H,W,C=x0.shape
        x0_flat=x0.reshape(-1,H,W,C)
        x1_flat=x1.reshape(-1,H,W,C)
        x2_flat=x2.reshape(-1,H,W,C)
        spatial_attn=self.fTSA(x2_flat,x1_flat,x0_flat)
        channel_attn=self.fTCA(x2_flat,x1_flat,x0_flat)
        xl_hat=spatial_attn*channel_attn.unsqueeze(-1)
        _,d_model,seq_length=xl_hat.shape
        xl_hat=xl_hat.view(B,L,d_model,seq_length)
        return xl_hat
    def calcPTCP(self,x):
        B,L,H,W,C=x.shape
        xl_hat=self.calculateXlHat(x)
        B,L,d_model,seq_length=xl_hat.shape
        xl_hat_cv=xl_hat.permute(0,2,1,3).reshape(B,d_model,-1)
        tc_out=self.TC(xl_hat_cv)
        ptcp=torch.matmul(tc_out,tc_out.transpose(-1,-2))
        return (1.0/L)*ptcp
    def forward(self,x):
        """
        Input X:(batch,Time,Height,Width,Channels)
        """
        Q0=self.calcPTCP(x)
        B,D,_=Q0.shape
        trace=torch.diagonal(Q0,dim1=-2,dim2=-1).sum(-1,keepdim=True).unsqueeze(-1)
        Q0=Q0/(trace+1e-6)
        I=torch.eye(D,device=Q0.device).unsqueeze(0).expand(B,-1,-1)
        R0=torch.eye(D,device=Q0.device).unsqueeze(0).expand(B,-1,-1)
        for i in range(self.iter):
            term=3*I-torch.matmul(R0,Q0) 
            Q0=0.5*torch.matmul(Q0,term)
            R0=0.5*torch.matmul(term,R0)
        return Q0

class ClassifierHead(nn.Module):
    def __init__(self,dim_in,dim_out,dropout=0.3):
        super().__init__()
        self.out_dim=dim_out
        self.nn=nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Dropout(dropout),
            nn.Linear(dim_in,dim_out),
            nn.GELU()
        )
    def forward(self,x):
        return self.nn(x)

class BinaryClassifier(nn.Module):
    def __init__(self, dim_in, dim_out=1):
        super().__init__()
        self.classfier = nn.Linear(dim_in, dim_out)
        self.sigmoid = nn.Sigmoid()  
    def forward(self, x):
        return self.sigmoid(self.classfier(x))

class BoundingBoxPredictor(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.ffn=nn.Sequential(
            nn.Linear(d_model*d_model,2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model,d_model),
            nn.GELU(),
            nn.Linear(d_model,d_model//4),
            nn.GELU(),
            nn.Linear(d_model//4,4),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.ffn(x)

class SegmentationHead(nn.Module):
    def __init__(self,d_model=256,patch_grid=16,output_size=80) -> None:
        super().__init__()
        self.patch_grid=patch_grid
        self.output_size=output_size
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(d_model,128,2,stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128,64,2,stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64,32,kernel_size=3,padding=1),
            nn.GELU(),
            nn.Upsample(size=(output_size,output_size),mode='bilinear',align_corners=False),
            nn.Conv2d(32,1,kernel_size=1)
        )
    def forward(self,patch_tokens,B,F):
        BF=B*F
        x=patch_tokens.permute(0,2,1)
        x=x.view(BF,-1,self.patch_grid,self.patch_grid)
        x=self.decoder(x)
        return x.view(B,F,80,80)