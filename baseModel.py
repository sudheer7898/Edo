import torch
import timm 
from torch import nn

class ptm(nn.Module):
    def __init__(self,model_name = 'resnet50',file:str=r'ptm.pth',out_indices=(1,2,3,4)):
        super().__init__()
        self.model=timm.create_model(model_name,features_only=True,out_indices=out_indices)
        state_dict=torch.load(file,weights_only=True)
        self.model.load_state_dict(state_dict,strict=False)
        self.model.eval()
        print("pre trained model craetion successful")
    def forward(self,x):
        return self.model(x)

class PatchEmbedding(nn.Module):
    def __init__(self,img_Size,patch_Size,n_channels,out_dim):
        super().__init__()
        self.out_dim=out_dim
        self.patch_Size=patch_Size
        self.num_patches=(img_Size//patch_Size)**2
        self.proj=nn.Conv2d(n_channels,out_dim,kernel_size=self.patch_Size,stride=self.patch_Size)
    def forward(self,x):
        y=self.proj(x)
        y=y.flatten(2)
        y=y.transpose(1,2)
        return y 

class Embedding(nn.Module):
    def __init__(self,img_Size,patch_size,n_channels,out_dim):
        super().__init__()
        self.patch_embeddings=PatchEmbedding(img_Size,patch_size,n_channels,out_dim)
        self.cls_Token=nn.Parameter(torch.randn(1,1,out_dim)*0.02)
        num_patches=(img_Size//patch_size)**2
        self.positional_embeddings=nn.Parameter(torch.randn(1,num_patches+1,out_dim)*0.02)
    def forward(self,x):
        y=self.patch_embeddings(x)
        batch_size,_,_ = y.size()
        cls_tokens=self.cls_Token.expand(batch_size,-1,-1)
        y=torch.cat((cls_tokens,y),dim=1)
        y=y+self.positional_embeddings
        return y

