import math
import torch
import torch.nn as nn

class CFG:
    num_nodes=207
    input_dim=1
    history=12
    horizon=12
    embed_dim=64
    num_heads=8
    num_hyperedges=64
    ffn_ratio=4
    temporal_layers=2
    conv_kernel=3
    dropout=0.1

cfg=CFG()

class InputProjection(nn.Module):
    def __init__(self,input_dim,embed_dim):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(input_dim,embed_dim),nn.GELU(),nn.Linear(embed_dim,embed_dim))
    def forward(self,x): return self.proj(x)

class TemporalPositionalEncoding(nn.Module):
    def __init__(self,embed_dim,max_len=500):
        super().__init__()
        pe=torch.zeros(max_len,embed_dim)
        position=torch.arange(0,max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,embed_dim,2)*(-math.log(10000.0)/embed_dim))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x):
        return x+self.pe[:,:x.shape[1]].unsqueeze(2)

class FrequencyAttention(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.real_gate=nn.Sequential(nn.Linear(embed_dim,embed_dim),nn.GELU(),nn.Linear(embed_dim,embed_dim),nn.Sigmoid())
        self.imag_gate=nn.Sequential(nn.Linear(embed_dim,embed_dim),nn.GELU(),nn.Linear(embed_dim,embed_dim),nn.Sigmoid())
    def forward(self,real,imag):
        return real*self.real_gate(real),imag*self.imag_gate(imag)

class FrequencyEncoder(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.attention=FrequencyAttention(embed_dim)
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self,x):
        freq=torch.fft.fft(x,dim=1)
        real,imag=self.attention(freq.real,freq.imag)
        signal=torch.fft.ifft(torch.complex(real,imag),dim=1).real
        return self.norm(signal+x)

class DynamicHyperedgeGenerator(nn.Module):
    def __init__(self,embed_dim,num_hyperedges):
        super().__init__()
        self.project=nn.Linear(embed_dim,num_hyperedges)
    def forward(self,x):
        return torch.softmax(self.project(x),dim=-1)

class HypergraphConv(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.linear=nn.Linear(embed_dim,embed_dim)
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self,x,incidence):
        hyper=torch.einsum("btnd,btnh->bthd",x,incidence)
        node=torch.einsum("bthd,btnh->btnd",hyper,incidence)
        return self.norm(self.linear(node)+x)

class HypergraphAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.attention=nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self,x):
        B,T,N,D=x.shape
        z=x.reshape(B*T,N,D)
        out,_=self.attention(z,z,z)
        return self.norm(out+z).reshape(B,T,N,D)

class HypergraphEncoder(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.frequency=FrequencyEncoder(embed_dim)
        self.generator=DynamicHyperedgeGenerator(embed_dim,cfg.num_hyperedges)
        self.hyperconv=HypergraphConv(embed_dim)
        self.attention=HypergraphAttention(embed_dim,cfg.num_heads)
        self.ffn=nn.Sequential(nn.Linear(embed_dim,embed_dim*cfg.ffn_ratio),nn.GELU(),nn.Linear(embed_dim*cfg.ffn_ratio,embed_dim))
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self,x):
        x=self.frequency(x)
        incidence=self.generator(x)
        x=self.hyperconv(x,incidence)
        x=self.attention(x)
        return self.norm(x+self.ffn(x))

class GraphAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0.1):
        super().__init__()
        self.attn=nn.MultiheadAttention(embed_dim,num_heads,dropout=dropout,batch_first=True)
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self,x):
        B,T,N,D=x.shape
        z=x.reshape(B*T,N,D)
        out,_=self.attn(z,z,z)
        return self.norm(out+z).reshape(B,T,N,D)

class GraphTransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.attention=GraphAttention(embed_dim,num_heads,cfg.dropout)
        self.ffn=nn.Sequential(nn.Linear(embed_dim,embed_dim*cfg.ffn_ratio),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(embed_dim*cfg.ffn_ratio,embed_dim))
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self,x):
        x=self.attention(x)
        return self.norm(x+self.ffn(x))

class CrossScaleFusion(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.query=nn.Linear(embed_dim,embed_dim)
        self.key=nn.Linear(embed_dim,embed_dim)
        self.value=nn.Linear(embed_dim,embed_dim)
        self.gate=nn.Sequential(nn.Linear(embed_dim*2,embed_dim),nn.Sigmoid())
    def forward(self,graph,hyper):
        q,k,v=self.query(graph),self.key(hyper),self.value(hyper)
        score=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(q.shape[-1])
        score=torch.softmax(score,dim=-1)
        fused=torch.matmul(score,v)
        gate=self.gate(torch.cat([graph,fused],dim=-1))
        return gate*fused+(1-gate)*graph

class TemporalStateSpace(nn.Module):
    # This reproduces the notebook's gated depthwise Conv1D block.
    def __init__(self,embed_dim):
        super().__init__()
        self.in_proj=nn.Linear(embed_dim,embed_dim)
        self.conv=nn.Conv1d(embed_dim,embed_dim,kernel_size=cfg.conv_kernel,padding=cfg.conv_kernel//2,groups=embed_dim)
        self.gate=nn.Sequential(nn.Linear(embed_dim,embed_dim),nn.Sigmoid())
        self.out_proj=nn.Linear(embed_dim,embed_dim)
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self,x):
        B,T,N,D=x.shape
        residual=x
        z=self.in_proj(x).permute(0,2,3,1).reshape(B*N,D,T)
        z=self.conv(z).transpose(1,2).reshape(B,N,T,D).permute(0,2,1,3)
        z=self.out_proj(self.gate(z)*z)
        return self.norm(residual+z)

class TemporalEncoder(nn.Module):
    def __init__(self,embed_dim,layers=2):
        super().__init__()
        self.layers=nn.ModuleList([TemporalStateSpace(embed_dim) for _ in range(layers)])
    def forward(self,x):
        for layer in self.layers: x=layer(x)
        return x

class SpatialTemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hyper=HypergraphEncoder(cfg.embed_dim)
        self.graph=GraphTransformerBlock(cfg.embed_dim,cfg.num_heads)
        self.fusion=CrossScaleFusion(cfg.embed_dim)
        self.temporal=TemporalEncoder(cfg.embed_dim,cfg.temporal_layers)
    def forward(self,x):
        hyper=self.hyper(x)
        graph=self.graph(hyper)
        x=self.fusion(graph,hyper)
        return self.temporal(x)

class PredictionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder=nn.Sequential(nn.Linear(cfg.embed_dim,cfg.embed_dim),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(cfg.embed_dim,cfg.horizon))
    def forward(self,x):
        x=x[:,-1]
        return self.decoder(x).permute(0,2,1).unsqueeze(-1)

class HGFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=InputProjection(cfg.input_dim,cfg.embed_dim)
        self.position=TemporalPositionalEncoding(cfg.embed_dim)
        self.encoder=SpatialTemporalEncoder()
        self.decoder=PredictionHead()
    def forward(self,x):
        return self.decoder(self.encoder(self.position(self.input(x))))
