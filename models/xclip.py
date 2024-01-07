from typing import Tuple, Union
import torch
from torch import nn
import numpy as np
from .mit import MultiframeIntegrationTransformer
from .prompt import VideoSpecificPrompt
from .cct import TokenTransformer
import sys
import warnings
sys.path.append("../")
from clip.model import CLIP,LayerNorm,Transformer
import clip

class XCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int, 
                 # video
                 T=8, 
                 droppath=0.,
                 mit_layers=1,
                 # prompt 
                #  prompts_alpha=1e-4,
                #  prompts_layers=1,
                 # other
                 #temporal_tokens 
                 temporal_tokens = [],
                 use_cache=True,
                 use_checkpoint=False,
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        
        self.prompts_generator = VideoSpecificPrompt(layers=1, embed_dim=embed_dim, alpha=1e-4,)
        self.use_cache=use_cache
        self.mit = MultiframeIntegrationTransformer(T=T, embed_dim=embed_dim, layers=mit_layers,)

        dpr = [x.item() for x in torch.linspace(0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64
        self.visual = TokenTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            droppath=dpr,
            T=T,
            use_checkpoint=use_checkpoint,
            temporal_tokens=temporal_tokens
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cache_text_features = None
        self.prompts_visual_ln = LayerNorm(vision_width)
        self.prompts_visual_proj = nn.Parameter(torch.randn(vision_width, embed_dim))
        self.prompts_text_video = LayerNorm(embed_dim)
        self.prompts_text_video_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.prompts_weight = nn.Parameter(torch.ones(embed_dim) * 1e-4)
        self.initialize_parameters()
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)
        eos_indx = text.argmax(dim=-1)
        K, N1, C = x.shape

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(K, -1)
        return x

    def encode_video(self, image, text_features):  ####text_features: 8x400x512
        b,t,c,h,w = image.size()
        image = image.reshape(-1,c,h,w)####64x3x224x224

        temporal_features, image_features, split_tokens_list = self.encode_image(image)#####64x512 64x49x768
        image_features = self.prompts_visual_ln(image_features)
        image_features = image_features @ self.prompts_visual_proj
        temporal_features = temporal_features.view(b, t, -1)
        image_features = image_features.view(b,t,-1,temporal_features.shape[-1])
        
        # attn_weight = temporal_features @ text_features.transpose(-1, -2)
        # values_1, _ = attn_weight.topk(5, dim=-1)
        # values_1 = values_1.mean(dim=-1, keepdim = False).softmax(dim = -1)
        # prompts_temporal_features = torch.einsum("btd, bt->btd", temporal_features, values_1)
        # prompts_temporal_features = self.prompts_text_video(prompts_temporal_features) @ self.prompts_text_video_proj
        # temporal_features = temporal_features + self.prompts_weight*prompts_temporal_features

        video_features = self.mit(temporal_features)
  
        return video_features, image_features, split_tokens_list

    def cache_text(self, text):
        self.eval()
        with torch.no_grad():
            if self.cache_text_features is None:
                self.cache_text_features = self.encode_text(text)
        self.train()
        return self.cache_text_features

    def forward(self, image, text):
        b = image.shape[0]####image: 8x8x3x224x224  text: 400x77
        if self.use_cache:
            text_features = self.cache_text(text)
        else:
            text_features = self.encode_text(text)
        
        text_features = text_features.unsqueeze(0).expand(b, -1, -1)


        video_features, image_features, split_tokens_list= self.encode_video(image, text_features)
        image_features = image_features.mean(dim=1, keepdim=False)
        temporal_fea = torch.cat(split_tokens_list,dim=1)
        # text_features = text_features + self.prompts_generator(text_features, image_features)
        text_features = text_features + self.prompts_generator(text_features, temporal_fea)

        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = torch.einsum("bd,bkd->bk", video_features, logit_scale * text_features)
        logits_list = []
        for tokens in split_tokens_list:
            tokens = tokens.mean(dim = 1, keepdim = False)
            tokens = tokens / tokens.norm(dim=-1, keepdim=True)
            logits_tokens = torch.einsum("bd,bkd->bk", tokens, logit_scale * text_features)
            logits_list.append(logits_tokens)
        return logits, logits_list


def build_model(state_dict: dict, T=8, droppath=0., use_checkpoint=False, logger=None,  use_cache=True, mit_layers=4, temporal_tokens = []):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = XCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  
        T=T, droppath=droppath, mit_layers=mit_layers,
        # prompts_alpha=prompts_alpha, prompts_layers=prompts_layers,
        use_checkpoint=use_checkpoint, use_cache=use_cache,
        temporal_tokens = temporal_tokens
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    msg = model.load_state_dict(state_dict,strict=False)
    logger.info(f"load pretrained CLIP: {msg}")
    
    return model.eval()


def load(model_path, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", 
         jit=True, T=8, droppath=0., use_checkpoint=False, logger=None, use_cache=True, prompts_alpha=1e-1, prompts_layers=2, mit_layers=1, temporal_tokens = []
):
    if model_path is None:
        model_path = clip._download(clip._MODELS[name])
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), T=T, droppath=droppath, 
                        use_checkpoint=use_checkpoint, logger=logger,
                        # prompts_alpha=prompts_alpha, 
                        # prompts_layers=prompts_layers,
                        use_cache=use_cache,
                        mit_layers=mit_layers,
                        temporal_tokens = temporal_tokens
                        )
    if str(device) == "cpu":
        model.float()
    return model, model.state_dict()
