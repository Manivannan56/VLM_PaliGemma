import torch
import torch.nn as nn
from typing import Optional,Tuple,List
from torch.nn import CrossEntropyLoss
import math
from Siglip_model import SiglipVisionConfig, SiglipVisionModel

class GemmaConfig():
     
 def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,):
          
    super().__init__()
    self.vocab_size=vocab_size
    self.max_position_embeddings=max_position_embeddings
    self.hidden_size=hidden_size
    self.intermediate_size=intermediate_size
    self.num_hidden_layers=num_hidden_layers
    self.num_attention_heads=num_attention_heads
    self.head_dim=head_dim
    self.num_key_value_heads=num_key_value_heads
    self.rms_norm_eps=rms_norm_eps
    self.rope_theta=rope_theta
    self.attention_bias=attention_bias
    self.attention_dropout=attention_dropout
    self.pad_token_id=pad_token_id



class PaliGemmaConfig():
    
    def __init__(
            self,
            vision_config=None,
            text_config=None,
            ignore_index=-100,
            image_token_index=256000,
            vocab_size=257152,
            projection_dim=2048,
            hidden_size=2048,
            pad_token_id=None,
            **kwargs,
    ):
        
         super().__init__()
         self.ignore_index=ignore_index
         self.image_token_index=image_token_index
         self.vocab_size=vocab_size
         self.projection_dim=projection_dim
         self.hidden_size=hidden_size
         self.vision_config=vision_config
         self.is_encoder_decoder=False
         self.pad_token_id=pad_token_id

         self.vison_config=SiglipVisionConfig(**vision_config)
         self.text_config=text_config

         self.text_config=GemmaConfig(**text_config,pad_token_id=pad_token_id)
         self.vocab_size=self.text_config.vocab_size
         self.vision_config.projection_dim=projection_dim


class GemmaRMSNorm(nn.Module):
   def __init__(self,dim:int, eps:float=1e-6):
      super().__init__()
      self.eps=eps
      self.weight=nn.Parameter(torch.zeros(dim))

   def _norm(self,x):
      return x* torch.sqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
   
   def forward(self,x):
      output=self._norm(x.float())

      #Llama does x.to(float16) * w whilst Gemma is(x*w).(float16)
      output=output*(1.0+self.weight.float())
      return output.type_as(x)
   
class GemmaMLp(nn.Module):
   def __init__(self,config):
      super().__init__()
      self.config=config
      self.hidden_size=config.hidden_Size
      self.intermediate_size=config.intermediate_size
      self.gate_proj= nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
      self.up_proj=nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
      self.down_proj=nn.Linear(self.intermediate_size,self.hidden_size,bias=False)

   
   def foward(self,x):
       #[Batch_size,Seq_len,hidden_size] -> [Batch_Size,Seq_len,Intermediate_size]
       y=self.gate_proj(x)
       #[Batch_Size,Seq_len,Intermediate_size]
       y=nn.functional.gelu(y,approximate="tanh")
       #[Batch_size,Seq_len,hidden_size] -> [Batch_Size,Seq_len,Intermediate_size] 
       j=self.up_proj(x)
       #[Batch_size,Seq_len,Intermediate_size] 
       z=y*j
       #[Batch_size,Seq_len,Intermediate_size] -> [Batch_Size,Seq_len,hidden_size] 
       z=self.down_proj(z)

       return z 

       
  

   
class GemmaDecoderLayer(nn.Module):
     def __init__(self, config:GemmaConfig, layer_idx:int):
        super().__init__()
        self.hidden_size=config.hidden_size
        self.ml-=GemmaMLP(config)

        self.self_attn=GemmaAttention(config=config,layer_idx=layer_idx)
        self.input_layernorm=GemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm=GemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)

     def forward(self,
                 hidden_states:torch.Tensor,
                 attention_mask:Optional[torch.Tensor]=None,
                 position_ids:Optional[torch.Tensor]=None,
                 kv_cache:Optional[KVCache]=None,
                 )-> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,torch.FloatTensor]]]:
        
        residual=hidden_states

        #[Batch_size,Seq_len,Hidden_size]
        hidden_states=self.input_layernorm(hidden_states)

        #[Batch_size,Seq_len,Hidden_Size]

        hidden_states, _,=self.self_attn(
           hidden_states=hidden_states,
           attention_mask=attention_mask,
           position_ids=position_ids,
           kv_cache=kv_cache,
        )

        #[Batch_Size,Seq_len,Hidden_size]

        hidden_states=residual+hidden_states

        #[Batch_Size,Seq_len,Hidden_size]
        resiual=hidden_states

        #[Batch_Size,Seq_len,Hidden_size]
        hidden_states=self.post_attention_layernorm(hidden_states)

        #[Batch_Size,Seq_len,Hidden_size]
        hidden_states=self.mlp(hidden_states)
        #[Batch_Size,Seq_len,Hidden_size]
        hidden_states=residual+hidden_states

        return hidden_states




class GemmaModel(nn.Module):
   
   def __init__(self,config:GemmaConfig):
      super().__init__()
      self.config=config
      self.padding_idx=config.pad_token_id
      self.vocab_size=config.vocab_size

      self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size,self.padding_idx)
      self.layers=nn.ModuleList(
         [GemmaDecoderLayer(config,layer_idx)  for layer_idx in range(config.num_hidden_layers)]
      )

      self.norm=GemmaRMSNorm(config.hidden_size,eps=config.rms_norm_rps)


   def get_input_embeddings(self):
      return self.embed_tokens
   
   def forward(self,
               attention_mask: Optional[torch.Tensor]=None,
               position_ids:Optional[torch.LongTensor]=None,
               input_embeds: Optional[torch.FloatTensor]=None,
               kv_cache:Optional[KVCache]=None,)->torch.FloatTensor:
      
      #[Batch_size,seq_len,Hidden_size]
      hidden_states=input_embeds

      normalizer=torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
      hidden_states=hidden_states*normalizer


      for decoder_layer in self.layers:
         #[Batch_size,seq_len,hidden_size]

         hidden_states=decoder_layer(hidden_states,attention_mask=attention_mask,
                                     position_ids=position_ids,
                                     kv_cache=kv_cache,)
         
      #[Batch_size,Seq_len,hidden_size]
      
      hidden_states=self.norm(hidden_states)

         #[Batch_size,Seq_len,hidden_size]
      return hidden_states




   


class GemmaForCausalLM(nn.Module):
   def __init__(self,config):
      super().__init__()
      self.config=config
      self.model=GemmaModel(config)
      self.vocab_size=config.vocab_size
      self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
   
   def get_input_embeddings(self):
      return self.model.embed_tokens
   
   def tie_weights(self):
      self.lm_head.weight=self.model.embed_tokens.weight
   
   def forward(
         self,
         attention_mask:Optional[torch.Tensor]=None,
         position_ids:Optional[torch.Tensor]=None,
         input_embeds:Optional[torch.FloatTensor]=None,
         kv_cache:Optional[KVCache]=None,
   )->Tuple:
      

      #input_embeds:[Batch_size,Seq_len,Hidden_size]
      #outputs:[Batch_size,Seq_len,Hidden_Size]

      outputs=self.model(
         attention_mask=attention_mask,
         position_ids=position_ids,
         input_embeds=input_embeds,
         kv_cache=kv_cache,
      )

      hidden_states=outputs
      logits=self.lm_head(hidden_states)
      logits=logits.float()

      return_data={
         "logits":logits,
      }


      if kv_cache is not None:
         # Return the updated cache
         return_data["kv_cache"]=kv_cache
      
      return return_data
   

   


class PaliGemmaMultiModalProjector(nn.Module):
   def __init__(self,config:PaliGemmaConfig):
      super().__init__()
      self.linear=nn.Linear(config.vision_config.hidden_size,config.vision_config.projection_dim,bias=True)
   
   def forward(self,image_features):
      #[Batch_size,Num_Patches,Embed_dim]->[Batch_size,Num_patches,Projection_dim]
      hidden_states=self.linear(image_features)
      return hidden_states




class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self,config:PaliGemmaConfig):
      
      super().__init__()
      self.config=config
      self.vision_tower=SiglipVisionModel(config=config.vision_config)
      self.multi_modal_projector=PaliGemmaMultiModalProjector(config) 
      self.vocab_size=config.vocab_size

      language_model=GemmaForCausalLM(config.text_config)
      self.language_model=language_model


    def tie_weights(self):
       return self.language_model.tie_weights()


    def merge_input_ids_with_image_features(self,image_features:torch.Tensor,
                                            input_embeds:torch.Tensor,
                                          input_ids:torch.Tensor,
                                          attention_mask:torch.Tensor,
                                          kv_cache:Optional[KVCache]=None
         
    ):
       _,_,embed_dim=image_features.shape
       batch_size,sequence_length=input_ids.shape

       #Shape: [Batch_size,Seq_len,Hidden_Size]
       dtype,device=input_embeds.dtype,input_embeds.device

       scaled_image_features=image_features/(self.config.hidden_size**0.5)
       
       #Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens
       final_embedding=torch.zeros(batch_size,sequence_length,embed_dim,dtype=input_embeds.dtype,device=input_embeds.device)
       
       #Shape:[Batch_Size,Seq_len]. True for text tokens
       text_mask= (input_ids != self.config.image_token_index) & (input_ids!= self.pad_token_id)

       #Shape: [Batch_Size, Seq_len]. True for image tokens

       image_mask= input_ids==self.config.image_token_index

       #Shape: [Batch_Size, Seq_len]. True for padding tokens

       pad_mask= input_ids==self.pad_token_id


       # We need to expand the masks to the embedding dimension otherwise we can't use them 
       text_mask_expanded=text_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
       pad_mask_expanded=pad_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
       image_mask_expanded=image_mask.unsqueeze(-1).expand(-1,-1,embed_dim)

       #Add the text embeddings
       final_embedding=torch.where(text_mask_expanded,input_embeds,final_embedding)

       #Insert image embeddings.We can't use torch.where because the sequence length of scaled_image_features in not equal to the sequence length of the final embeddings
       final_embedding=final_embedding.masked_scatter(image_mask_expanded,scaled_image_features)
       
       #Zero Out the Padding tokens
       final_embedding=torch.where(pad_mask_expanded,torch.zeros_like(final_embedding),final_embedding)

       ## Create the Attention Mask  ##
       dtype,device=input_embeds.dtype,input_embeds.device
       min_dtype=torch.finfo(dtype).min
       q_len=input_embeds.shape[1]

       if kv_cache is None or kv_cache.num_items()==0:
          
          #Do not mask any token because we're in the prefill phase where the attention matrix if of dimension seq_len x seq_len  , so the mask shape is also like that

          causal_mask=torch.full((batch_size,q_len,q_len),fill_value=0,dtype=dtype,device=device)

       else:
          assert q_len==1
          kv_len=kv_cache.num_items()+q_len
          # Also in this case we don't need to mask anything, since each query should be able to attend all the previous tokens, since we are adding a new generated tokne everytime to the kv_cache the length is seq_len x kv_len

          causal_mask=torch.full((batch_size,q_len,kv_len),fill_value=0,dtype=dtype,device=device)

       #Adding the head dimension
       #[Batch_size,q_len,kv_len] ->[Batch_size,Num_heads,q_len,kv_len]
       causal_mask=causal_mask.unsqueeze(1)

       if kv_cache is not None and kv_cache.num_items()>0:
          position_ids=attention_mask.cumsum(-1)[:,-1]
          if position_ids.dim()==1:
             position_ids=position_ids.unsqueeze(0)
          
          else:
             # Create a position_ids based on the size of the attention_mask
             # For masked tokens, use the number 1 as position
             position_ids=(attention_mask.cumsum(-1)).masked_fill((attention_mask==0),1).to(device)
       
       return final_embedding,causal_mask,position_ids
    
             
          









       


       











    
          