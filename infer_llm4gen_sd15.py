import pandas as pd
import torch
import torch.nn as nn

from transformers import T5Tokenizer, T5EncoderModel, AutoModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import AutoPipelineForText2Image

import os
import pandas as pd
from tqdm import tqdm
import random
from PIL import Image

# model_id = "runwayml/stable-diffusion-v1-5"
model_id = "sd-legacy/stable-diffusion-v1-5"

generator = torch.Generator("cuda").manual_seed(1)

class RankGenEncoder():
    def __init__(self, model_path, max_batch_size=32, model_size=None, cache_dir=None):
        assert model_path in ["kalpeshk2011/rankgen-t5-xl-all", "kalpeshk2011/rankgen-t5-xl-pg19", "kalpeshk2011/rankgen-t5-base-all", "kalpeshk2011/rankgen-t5-large-all"]
        self.max_batch_size = max_batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_size is None:
            if "t5-large" in model_path or "t5_large" in model_path:
                self.model_size = "large"
            elif "t5-xl" in model_path or "t5_xl" in model_path:
                self.model_size = "xl"
            else:
                self.model_size = "base"
        else:
            self.model_size = model_size

        self.tokenizer = T5Tokenizer.from_pretrained(f"google/t5-v1_1-{self.model_size}", cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()



    def encode(self, inputs, vectors_type="prefix", verbose=False, return_input_ids=False, max_length=256):
        tokenizer = self.tokenizer
        max_batch_size = self.max_batch_size
        if isinstance(inputs, str):
            inputs = [inputs]

        if vectors_type == 'prefix':
            inputs = ['pre ' + input for input in inputs]
        else:
            inputs = ['suffi ' + input for input in inputs]
        

        tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, max_length=max_length)
        length = tokenized_inputs['input_ids'].shape[1]
        if length > max_length:
            tokenized_inputs['input_ids'] = tokenized_inputs['input_ids'][:, :max_length]
            tokenized_inputs['attention_mask'] = tokenized_inputs['attention_mask'][:, :max_length]
        else:
            padding_length = max_length - length
            padding_tokens = torch.zeros(tokenized_inputs['input_ids'].shape[0], padding_length, dtype=tokenized_inputs['input_ids'].dtype)                
            tokenized_inputs['input_ids'] = torch.cat([tokenized_inputs['input_ids'], padding_tokens], dim=1)
            tokenized_inputs['attention_mask'] = torch.cat([tokenized_inputs['attention_mask'], padding_tokens], dim=1)
        
        tokenized_inputs = tokenized_inputs.to(self.device)
        with torch.no_grad():
            batch_embeddings = self.model.t5_encoder(**tokenized_inputs).last_hidden_state
        return batch_embeddings

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class CrossFusion(nn.Module):
    def __init__(self, llama_dim, dim, heads):
        super(CrossFusion, self).__init__()
        self.num_heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        assert self.head_dim * heads == dim, "dim must be divisible"
        
        self.scale = self.head_dim ** -0.5
        self.llm_proj = nn.Linear(llama_dim, dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)     
        self.out_proj = nn.Linear(dim, dim)

        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.norm = nn.LayerNorm(dim)

        self.FFN = nn.Sequential(
            nn.Linear(dim, dim * 4),
            QuickGELU(),
            nn.Linear(dim * 4, dim)
        )
        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.trunc_normal_(self.q_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.k_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.v_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        
    def forward(self, clip_embed, llm_embed):
        B, _, _ = llm_embed.shape
        llm_embed = self.llm_proj(llm_embed)
        clip_embed_norm = self.q_norm(clip_embed)
        llm_embed_norm = self.kv_norm(llm_embed)        
        query = self.q_proj(llm_embed_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)
        key = self.k_proj(clip_embed_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)
        value = self.v_proj(clip_embed_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)
        attention_weights = (query @ key.transpose(-2, -1)) * self.scale
        attention_weights = attention_weights.softmax(dim=-1)
        out = (attention_weights @ value).transpose(1,2).reshape(B, -1, self.dim)
        llm_embed = self.out_proj(out)+ llm_embed
        llm_embed = self.FFN(self.norm(llm_embed)) + llm_embed
        concat_embed = torch.cat((clip_embed, llm_embed), dim=1)
        return concat_embed

class LLMFusionModule(nn.Module):
    def __init__(self, clip_dim, llm_dim, num_heads):
        super(LLMFusionModule, self).__init__()
        self.CrossFusionModule = nn.ModuleList(
            [CrossFusion(llm_dim, clip_dim, num_heads) for _ in range(1)]
        )

    def forward(self, clip_text, llm_text):
        for module in self.CrossFusionModule:
            clip_text = module(clip_text, llm_text)
        return clip_text


model_path = "unet"
llm_projector_path = f"projector.pth"
llm_projector = LLMFusionModule(768, 2048, 8)
msg = llm_projector.load_state_dict(torch.load(llm_projector_path))
llm_projector.to("cuda").eval()
unet = UNet2DConditionModel.from_pretrained(
    model_path, 
).cuda()

t5_model = RankGenEncoder("kalpeshk2011/rankgen-t5-xl-all")

pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                               unet=unet,
                                               safety_checker=None,
                                               torch_dtype=torch.float32).to("cuda")

prompt = '''
Ground view of the Great Pyramids and Sphinx on the moon's surface. The back of an astronaut is in the foreground. The planet Earth looms in the sky.
'''

file_name = "example.jpg"
llm_embed = t5_model.encode(prompt)
input_ids = pipe.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=77).input_ids
input_ids = input_ids.to("cuda")
clip_embed = pipe.text_encoder(input_ids, return_dict=False)[0]
encoder_hidden_states = llm_projector(clip_embed, llm_embed)
neg_llm_embed = t5_model.encode("", max_length=llm_embed.shape[1])
negative_ids = pipe.tokenizer("", truncation=True, return_tensors="pt", padding="max_length", max_length=77).input_ids
negative_ids = negative_ids.cuda()
neg_clip_embeds = pipe.text_encoder(negative_ids, return_dict=False)[0]
neg_embeds = llm_projector(neg_clip_embeds, neg_llm_embed)

with torch.no_grad():                                     
    image = pipe(prompt_embeds=encoder_hidden_states, negative_prompt_embeds=neg_embeds).images[0]
    image.save(file_name)
