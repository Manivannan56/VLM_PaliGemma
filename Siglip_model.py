import torch
import torch.nn as nn
from typing import Optional, Tuple

class SiglipVisionConfig:
    
    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int=None,
            **kwargs
    ):
        super().__init__()
        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        self.num_hidden_layers=num_hidden_layers
        self.num_channels=num_channels
        self.num_attention_heads=num_attention_heads
        self.image_size=image_size
        self.patch_size=patch_size
        self.layer_norm_eps=layer_norm_eps
        self.attention_dropout=attention_dropout
        self.num_image_tokens=num_image_tokens


'''
Here we are converting the image into patches and then into embeddings

'''

class SiglipVisionEmbeddings(nn.Module):
    def __init__(
            self,
            config: SiglipVisionConfig):
        
        super().__init__()
        self.embed_dim=config.hidden_size
        self.image_size=config.image_size
        self.patch_size=config.patch_size

        self.patch_embeddings=nn.Conv2d(in_channels=config.num_channels,
                                        out_channels=self.embed_dim,
                                        kernel_size=self.patch_size,
                                        stride=self.patch_size,
                                        padding="valid")                            ## No Padding is added   
        
        
        self.num_patches=(self.image_size//self.patch_size)**2
        self.num_positions=self.num_patches
        self.position_embeddings=nn.Embedding(self.num_positions,self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False
        )

    def forward(self,pixel_values:torch.FloatTensor)->torch.Tensor:
       _,_,height,width= pixel_values.shape
       patch_embeds=self.patch_embeddings(pixel_values)
      ## print(patch_embeds)    Checking the tensor shape which is [Batch_size,embedding_dim,num_patches_H, num_patches_W] ==>([2, 768, 14, 14])
       patch_embeds=patch_embeds.flatten(2)
       patch_embeds=patch_embeds.transpose(1,2)

       ## Concatenating the flattened patch and the position embeddings
       total_embeddings=patch_embeds+self.position_embeddings(self.position_ids)

       ## total_embeddings_dimension= [Batch_size,num_patches,embed_dim]

       return total_embeddings

class SiglipMLP(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.fc1=nn.Linear(in_features=config.hidden_size,out_features=config.intermediate_size)
        self.fc2=nn.Linear(in_features=config.intermediate_size,out_features=config.hidden_size)
    
    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        ##Tensor Dimension=[Batch_size,num_patches,embed_dim]
        hidden_states=self.fc1(hidden_states)

        ##Tensor Dimension=[Batch_size,num_patches,intermediate_size]
        hidden_states=nn.functional.gelu(hidden_states,approximate="tanh")
        
        hidden_states=self.fc2(hidden_states)
        ##Tensor Dimension=[Batch_size,num_patches,embed_dim]

        return hidden_states


class SiglipAttention(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.num_heads=config.num_attention_heads
        self.embed_dim=config.hidden_size
        self.head_dim=self.embed_dim//self.num_heads
        self.scale=self.head_dim**-0.5
        self.dropout=config.attention_dropout

        self.k_proj=nn.Linear(self.embed_dim,self.embed_dim)
        self.v_proj=nn.Linear(self.embed_dim,self.embed_dim)
        self.q_proj=nn.Linear(self.embed_dim,self.embed_dim)
        self.out_proj=nn.Linear(self.embed_dim,self.embed_dim)
    
    def forward(self,hidden_states:torch.Tensor)->Tuple[torch.Tensor,Optional[torch.Tensor]]:
         # Tensor Shape=[Batch_size,num_patches,embed_dim]
         batch_size,seq_len,_=hidden_states.size()
         query_states=self.q_proj(hidden_states)
         key_states=self.k_proj(hidden_states)
         value_states=self.v_proj(hidden_states)

         query_states=query_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
         #Tensor Shape=[Batch_size,num_heads,num_patches,head_dim]
        
         key_states=key_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
         #Tensor Shape=[Batch_size,num_heads,num_patches,head_dim]

         value_states=value_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
         #Tensor Shape=[Batch_size,num_heads,num_patches,head_dim]

         attention_weights= (torch.matmul(query_states,key_states.transpose(2,3))*self.scale)

         if attention_weights.size()!=(batch_size,self.num_heads,seq_len,seq_len):
              raise ValueError(
                  f"'attn_output' should be of size{(batch_size,self.num_heads,seq_len,seq_len)},but is"
                  f"{attention_weights.size()}" 
              )
         
         ## Applying softmax row wise (dim=-1) ,tensor shape=[Batch_size,num_heads,num_patches,num_patches]
         
         attention_weights=nn.functional.softmax(attention_weights,dim=-1,dtype=torch.float32).to(query_states.dtype)
         
         ## Only apply dropout during training
         attention_weights=nn.functional.dropout(attention_weights,p=self.dropout)

         attention_output=torch.matmul(attention_weights,value_states)
         #Output tensor shape= [Batch_size,num_heads,num_patches,head_dim]

         if attention_output.size()!=(batch_size,self.num_heads,seq_len,self.head_dim):
              raise ValueError(
                  f"'attn_output' should be of size{(batch_size,self.num_heads,seq_len,self.head_dim)},but is"
                  f"{attention_output.size()}" 
              )
         
         ## Now we need to concatenate every head's predictions back into the same tensor dimension=[Batch_Size,num_heads,num_patches,head_dim]-> [Batch_size,num_patches,embed_dim]

         attention_output=attention_output.transpose(1,2).contiguous()

         attention_output=attention_output.reshape(batch_size,seq_len,self.embed_dim)
         attention_output=self.out_proj(attention_output)

         

         return attention_output,attention_weights




class SiglipEncoderlayer(nn.Module):
    def __init__(self,conif:SiglipVisionConfig):
        super().__init__()
        self.embed_dim=config.hidden_size
        self.self_attn=SiglipAttention(config)
        self.layer_norm1=nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
        self.mlp=SiglipMLP(config)
        self.layer_norm2=nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
    
    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        residual=hidden_states
        ## Tensor Shape- [Batch_size,Num_patches,embed_dim]
        hidden_states=self.layer_norm1(hidden_states)
        #Tensor shape- [Batch_size,Num_patches,embed_dim]

        hidden_states,_=self.self_attn(hidden_states=hidden_states)

        hidden_states=residual+hidden_states

        residual=hidden_states

        hidden_states=self.layer_norm2(hidden_states)
        hidden_states=self.mlp(hidden_states)
        hidden_states=residual+hidden_states
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.layers=nn.ModuleList(
            [SiglipEncoderlayer(config) for _ in range(config.num_hidden_layers) ]
        )

    def forward(self,input_embeds:torch.Tensor)-> torch.Tensor:
        hidden_states= input_embeds

        for encoder_layer in self.layers:
            hidden_states=encoder_layer(hidden_states)
        
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        embed_dim=config.hidden_size

        self.embeddings=SiglipVisionEmbeddings(config)
        self.encoder=SiglipEncoder(config)
        self.post_layernorm=nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)
    
    def forward(self,pixel_values:torch.Tensor)->torch.Tensor:
        
        #Tensor shape= [Batch_size,num_channels,height,width]
        hidden_states=self.embeddings(pixel_values)

        last_hidden_state=self.encoder(hidden_states)
        last_hidden_state=self.post_layernorm(last_hidden_state)

        return last_hidden_state

class SiglipVisionModel(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.vision_model=SiglipVisionTransformer(config)
    
    def forward(self,pixel_values)->Tuple:

        #Tensor Shape Change:  [Batch_size,num_channels,height,width]--> [Batch_size,num_patches,embed_dim]
        return self.vision_model(pixel_values=pixel_values)




config=SiglipVisionConfig()
model=SiglipVisionModel(config=config)

dummy=torch.randn(2,3,224,224)
out=model(dummy)

model_1=SiglipAttention(config=config)
dummy_1=torch.randn(2,196,768)
out_1=model_1(dummy_1)






