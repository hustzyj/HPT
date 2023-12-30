from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
import sys
sys.path.append("../")
from clip.model import LayerNorm, QuickGELU, DropPath
from torch import einsum
from einops import rearrange, reduce, repeat

class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath = 0., T=0, temporal_layers=2):
        super().__init__()
        self.T = T
        self.message_ln=nn.ModuleList()
        self.message_attn=nn.ModuleList()
        self.temporal_layers = temporal_layers
        for i in range(temporal_layers):
            self.message_ln.append(LayerNorm(d_model))
            self.message_attn.append(nn.MultiheadAttention(d_model, n_head,))

        self.message_fc = nn.Linear(d_model, d_model)

        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.token_inter = tokens_interact(width=d_model)
        self.alpha1 = nn.Parameter(torch.ones(d_model) * 0.5)
        

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def split_x(self, x, num_temporal_token):
        return x[:, :-num_temporal_token], x[:, -num_temporal_token:]
    def concat_x(self, x, temporal_token=None):
        return torch.cat([x, temporal_token], dim=1)
    def forward(self, x, temporal_tokens):
        _, num_temporal_token, _ = temporal_tokens.shape
        l, bt, d = x.size()
        b = bt // self.T


        ###tokens_temporal
        # frame_tokens = self.message_fc(x[0, :, :])
        # frame_tokens = frame_tokens.view(b, self.T, d)
        # concat_x = self.concat_x(frame_tokens, temporal_tokens.expand(b, -1, -1)).permute(1, 0, 2)
        # for j in range(self.temporal_layers):
        #     concat_x = concat_x + self.drop_path(self.message_attn[j](self.message_ln[j](concat_x),self.message_ln[j](concat_x),self.message_ln[j](concat_x),need_weights=False)[0])
        # frame_res, split_tokens = self.split_x(concat_x.permute(1, 0, 2), num_temporal_token)
        # frame_res = self.token_inter(split_tokens, frame_res) + frame_res####b,t,d
        # frame_res = frame_res.view(-1, d).unsqueeze(0)
        
        # x = torch.cat([x, frame_res], dim=0)

        # x = x + self.drop_path(self.attention(self.ln_1(x)))###50x128x768
        # x = x[:l, :, :]
        # x = x + self.drop_path(self.mlp(self.ln_2(x)))#######l,bt,d


        x = x + self.drop_path(self.attention(self.ln_1(x)))###50x128x768
        x = x + self.drop_path(self.mlp(self.ln_2(x)))#######l,bt,d

        ####tokens_temporal
        frame_tokens = self.message_fc(x[0, :, :])
        frame_tokens = frame_tokens.view(b, self.T, d)
        concat_x = self.concat_x(frame_tokens, temporal_tokens.expand(b, -1, -1)).permute(1, 0, 2)
        for j in range(self.temporal_layers):
            concat_x = concat_x + self.drop_path(self.message_attn[j](self.message_ln[j](concat_x),self.message_ln[j](concat_x),self.message_ln[j](concat_x),need_weights=False)[0])
        frame_res, split_tokens = self.split_x(concat_x.permute(1, 0, 2), num_temporal_token)
        frame_res = self.token_inter(split_tokens, frame_res) + frame_res####b,t,d
        frame_res = frame_res.view(-1, d).unsqueeze(0)
        x[0, :, :] =  x[0, :, :] + self.alpha1*frame_res
        

        return x, split_tokens





        # x = x.view(l, b, self.T, d) 


        # msg_token = self.message_fc(x[0,:,:,:]) 
        # msg_token = msg_token.view(b, self.T, 1, d) 
        
        # msg_token = msg_token.permute(1,2,0,3).view(self.T, b, d) 
        # msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token),self.message_ln(msg_token),self.message_ln(msg_token),need_weights=False)[0])
        # msg_token = msg_token.view(self.T, 1, b, d).permute(1,2,0,3)
        
        # x = torch.cat([x, msg_token], dim=0)
        
        # x = x.view(l+1, -1, d)
        # x = x + self.drop_path(self.attention(self.ln_1(x)))
        # x = x[:l,:,:]
        # x = x + self.drop_path(self.mlp(self.ln_2(x)))
        # return x
class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
class tokens_interact(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.cross_attn = MulitHeadAttention(width, width//64)
            self.norm1 = nn.LayerNorm(width)
            self.norm3 = nn.LayerNorm(width)
            self.mlp = nn.Sequential(
                nn.Linear(width, width * 4),
                QuickGELU(),
                nn.Linear(width * 4, width)
            )
        def forward(self, split_tokens, tokens):
            q = self.norm1(tokens)
            tokens = self.cross_attn(q, split_tokens, split_tokens) + tokens
            return tokens
class temporal_interact(nn.Module):
    def __init__(self, layers, width,):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.token_inter = nn.ModuleList([tokens_interact(width) for _ in range(layers)])
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x, tokens):
        x = self.norm(x)
        for layers in self.token_inter:
            tokens = layers(x, tokens)
        return tokens


class temporal_layer(nn.Module):
    def __init__(self, width, num_temporal_token, num_prev_tokens=None):
        super().__init__()
        self.width = width
        self.num_temporal_token = num_temporal_token
        self.ln_pre = LayerNorm(width)
        if num_prev_tokens:
            self.project = nn.Sequential(OrderedDict([
                ('down_sample', nn.Linear(num_prev_tokens, num_temporal_token)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(num_temporal_token, num_temporal_token)),
                ("gelu", QuickGELU())
            ]))
        self.temporal_token = nn.Parameter(torch.randn(1, num_temporal_token, width))
        self.temporal_interact = temporal_interact(layers=2, width=width)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    
    def split_x(self, x):
        return x[:, :-self.num_temporal_token], x[:, -self.num_temporal_token:]
    def concat_x(self, x, temporal_token=None):
        return torch.cat([x, temporal_token], dim=1)
    def forward(self, x, prev_temporal_token=None, return_attn=False, blk=None, is_first = False):
        b, tl ,d = x.shape#####16x392x768
        tokens_present = self.temporal_token.expand(b, -1, -1)####16x256x768
        if is_first:
            cat_x = self.concat_x(x, tokens_present)
        else:
            ##temporal_tokens_fusion
            downsample_tokens = self.project(prev_temporal_token.transpose(1, 2)).transpose(1, 2)
            tokens_ = tokens_present + downsample_tokens
            cat_x = self.concat_x(x, tokens_)
        for block_layer in blk:
            cat_x = block_layer(cat_x)
        x, tokens_split = self.split_x(cat_x)#####16x392x768  16x256x768
        x = self.temporal_interact(x, tokens_split)
        return x, tokens_split
        

class Atten_blcok(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath = 0., T=0, temporal_tokens=0):
        super().__init__()
        self.T = T
        self.attn1 = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn1(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x.transpose(0, 1)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None, use_checkpoint=False, T=8, temporal_tokens=[]):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)] 
        self.width = width
        self.layers = layers
        self.temporal_tokens = temporal_tokens
        self.split = self.layers//len(self.temporal_tokens)
        # self.temporal_tokens = nn.Parameter(torch.randn(1, self.temporal_tokens[0], width))
        
        self.temporal_tokens_parameter = nn.ParameterList()
        self.temporal_tokens_MLP = nn.ModuleList()
        self.temporal_tokens_weight =  nn.ParameterList()
        for tokens in self.temporal_tokens:
            self.temporal_tokens_parameter.append(nn.Parameter(torch.randn(1, tokens, width), requires_grad=True))  
        
        
        for i in range(len(self.temporal_tokens)-1):
            self.temporal_tokens_MLP.append(nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(self.temporal_tokens[i], self.temporal_tokens[i+1])),
                ("gelu", QuickGELU()),
                # ("norm", nn.LayerNorm(self.temporal_tokens[i+1])),
                ("c_proj", nn.Linear(self.temporal_tokens[i+1] , self.temporal_tokens[i+1]))])))
            self.temporal_tokens_weight.append(nn.Parameter(torch.ones(width) * 1e-4))
        
        self.temporal_tokens_MLP_1_3 = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(self.temporal_tokens[0], self.temporal_tokens[2])),
                ("gelu", QuickGELU()),
                # ("norm", nn.LayerNorm(self.temporal_tokens[2])),
                ("c_proj", nn.Linear(self.temporal_tokens[2] , self.temporal_tokens[2]))]))
        self.temporal_tokens_weight_1_3 = nn.Parameter(torch.ones(width) * 1e-4)


        self.resblocks = nn.Sequential(*[CrossFramelAttentionBlock(width, heads, attn_mask, droppath[i], T, ) for i in range(layers)])
        
        # self.temporal_model = nn.ModuleList()
        # for i in range(len(self.temporal_tokens)):
        #     if i ==0:
        #         self.temporal_model.append(temporal_layer(width, self.temporal_tokens[i]))
        #     else:
        #         self.temporal_model.append(temporal_layer(width, self.temporal_tokens[i], self.temporal_tokens[i-1]))


    def forward(self, x: torch.Tensor):
        split_tokens_list = []
        for i in range(len(self.resblocks)):
            # if i==0:
            #     x, split_tokens = self.resblocks[i](x, self.temporal_tokens)
            # else:
            #     x, split_tokens = self.resblocks[i](x, split_tokens)

            # if i==0:
            #     pre_tokens = self.temporal_tokens_parameter[i//self.split]
            # if i!=0 and i//(self.split) == 1 and i%(self.split) ==0:
            #     split_tokens_0 = split_tokens
            #     new_tokens = self.temporal_tokens_MLP[(i//self.split) -1](split_tokens.transpose(1, 2)).transpose(1, 2)
            #     pre_tokens =self.temporal_tokens_parameter[i//self.split] + ((self.temporal_tokens_weight[(i//self.split) -1])*new_tokens).mean(dim=0, keepdim = True)
            #     # x, split_tokens = self.resblocks[i](x, pre_tokens)
            # elif i!=0 and i//(self.split) == 2 and i%(self.split) ==0:
            #     new_tokens_2 = self.temporal_tokens_MLP[(i//self.split) -1](split_tokens.transpose(1, 2)).transpose(1, 2)
            #     new_tokens_1 = self.temporal_tokens_MLP_1_3(split_tokens_0.transpose(1, 2)).transpose(1, 2)
            #     new_temporal_tokens_2 = ((self.temporal_tokens_weight[(i//self.split) -1])*new_tokens_2).mean(dim=0, keepdim = True) 
            #     new_temporal_tokens_1 = (self.temporal_tokens_weight_1_3*new_tokens_1).mean(dim=0, keepdim = True)  
            #     # pre_tokens =    new_temporal_tokens_2 + new_temporal_tokens_1 + self.temporal_tokens_parameter[i//self.split]
            #     pre_tokens =    new_temporal_tokens_2 + self.temporal_tokens_parameter[i//self.split]
            # elif i!=0 and i//(self.split) == 3 and i%(self.split) ==0:
            #     new_tokens_2_ = self.temporal_tokens_MLP[(i//self.split) -1](split_tokens.transpose(1, 2)).transpose(1, 2)
            #     new_temporal_tokens_2_ = ((self.temporal_tokens_weight[(i//self.split) -1])*new_tokens_2_).mean(dim=0, keepdim = True)
            #     pre_tokens =    new_temporal_tokens_2_ + self.temporal_tokens_parameter[i//self.split]
            # x, split_tokens = self.resblocks[i](x, pre_tokens)####
            if i%(self.split) == 0:
                x, split_tokens = self.resblocks[i](x, self.temporal_tokens_parameter[i//self.split])
            else:
                x, split_tokens = self.resblocks[i](x, split_tokens)
                if (i+1)%(self.split) == 0:
                    split_tokens_list.append(split_tokens)
        return x, split_tokens_list

class TokenTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 droppath = None, T = 8, use_checkpoint = False,temporal_tokens = []):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 +1, width))
        # self.pos_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2, width))
        self.ln_pre = LayerNorm(width)
        # self.time_embedding = nn.Parameter(scale * torch.randn(1, T, width))
        # self.ln_pre_time = LayerNorm(width)
        ## Attention Blocks
        self.transformer = Transformer(width, layers, heads, droppath=droppath, use_checkpoint=use_checkpoint, T=T,temporal_tokens=temporal_tokens)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.temporal_tokens_proj = nn.ParameterList()
        for i in range(len(temporal_tokens)):
            self.temporal_tokens_proj.append(nn.Parameter(scale * torch.randn(width, output_dim)))
        self.num_frame = T
        # self.temporal_tokens_cls = nn.Parameter(scale * torch.randn(width))
        self.apply(self._init_weights)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):

       B = x.shape[0]//self.num_frame
       x = self.conv1(x)#####128x768x7x7
       x = x.reshape(x.shape[0], x.shape[1], -1)####128x768x49
       x = x.permute(0, 2, 1)####128x49x768
       x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
       x = x + self.positional_embedding.to(x.dtype)
    #    x = torch.cat([x, self.temporal_tokens_cls.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)], dim=1)
       x = self.ln_pre(x)
       x = x.permute(1, 0, 2)#####50x128x768
       x, split_tokens_list = self.transformer(x)
       x = x.permute(1, 0, 2)
       cls_x = self.ln_post(x[:, 0, :])
       cls_x = cls_x @ self.proj

       for i in range(len(split_tokens_list)):
           split_tokens_list[i] = split_tokens_list[i] @ self.temporal_tokens_proj[i]

       return cls_x, x[:, 1:, :], split_tokens_list

