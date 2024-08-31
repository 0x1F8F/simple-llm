import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class ModelArg:
    vocab_size: int = -1
    emb_size: int = 32
    head_size: int = 64
    window_size: int = 20
    lr: float  = 1e-3
    batch_size: int = 32


class Head(nn.Module):
    def __init__(self, arg: ModelArg , inp : int):
        super().__init__()
        self.arg = arg
        self.head_size = arg.head_size
        self.bsz = arg.batch_size
        emb = inp
        self.wk = nn.Linear(emb , arg.head_size ,bias=False)
        self.wq = nn.Linear(emb , arg.head_size ,bias=False)
        self.wv = nn.Linear(emb, arg.head_size ,bias=False)
        wz = arg.window_size
        self.register_buffer("mask", torch.ones(wz,wz))
        
    def forward(self,x):
        k = self.wk(x)
        q = self.wq(x)
        v = self.wv(x) # w = ( (k @ q)/root(head_dim) ) @ v
        w = (k @ q.transpose(1,2)).mul(self.head_size ** -0.5)
        w = torch.masked_fill(w , self.mask == 0 , float('-inf'))
        w = nn.functional.softmax(w ,dim=-1) @ v
        return w


class MultiHead(nn.Module):
    def __init__(self , arg: ModelArg):
        super().__init__()
        self.arg = arg
    
    def forward(self , x):
        x = self.head(x)
        return(x)

class Transformer(nn.Module):
    def __init__(self , arg: ModelArg):
        super().__init__()
        self.arg = arg
        self.embedding = nn.Embedding(arg.vocab_size , arg.emb_size) # -> (batch , token , emb)
        self.head = Head(arg,arg.emb_size)
    
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        b , t = x.shape
        logits = self.embedding(x)
        logits = self.head(logits)
        # logits = nn.functional.softmax(logits, dim=-1)
        return logits # -> ( batch , token , channels )
    
    def trainer(self, x , y):
        logits = self(x)
        b , t , c = logits.shape
        loss = nn.functional.cross_entropy(logits.view(b*t,c) , y.view(b*t))
        return logits , loss

