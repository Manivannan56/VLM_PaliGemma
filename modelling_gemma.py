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



class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self,config:PaliGemmaConfig):
      
      super().__init__()
      self.config=config
      self.vision_tower=SiglipVisionModel(config=config.vision_config)

    



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
       








       


       











    
          