"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import (
    CLIPEncoder,
    CLIPPreTrainedModel,
    # _expand_mask,
    _prepare_4d_attention_mask
)


class CtxCLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CtxCLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def forward(
        self,
        ctx_embeddings: torch.Tensor = None,
        ctx_begin_pos: list = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return self.text_model(
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=ctx_begin_pos,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

### BraVO
class CtxCLIPTextModel_stage_1(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CtxCLIPTextTransformer_stage_1(config)
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        ctx_embeddings: torch.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, causal_attention_mask = self.text_model(
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=ctx_begin_pos,
            input_ids=input_ids,
        )
        return hidden_states, causal_attention_mask

class CtxCLIPTextModel_stage_2(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CtxCLIPTextTransformer_stage_2(config)
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_attention_mask: torch.Tensor 
    ) -> torch.Tensor:
        return self.text_model(
            hidden_states=hidden_states,
            causal_attention_mask=causal_attention_mask
        )
### BraVO

class CtxCLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CtxCLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ctx_embeddings: torch.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # hidden_states.shape = [bs, 77, 768]
        hidden_states = self.embeddings(
            input_ids=input_ids, # torch.Tensor
            position_ids=position_ids, # None
            ctx_embeddings=ctx_embeddings, # torch.Tensor
            ctx_begin_pos=ctx_begin_pos, # list
        )
        
        bsz, seq_len = input_shape
        if ctx_embeddings is not None:
            seq_len += ctx_embeddings.size(1)
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states.dtype
        ).to(hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder( # transformers.models.clip.modeling_clip.CLIPEncoder
            inputs_embeds=hidden_states,  # torch.Tensor
            attention_mask=attention_mask,  # None
            causal_attention_mask=causal_attention_mask,  # torch.Tensor
            output_attentions=output_attentions,  # bool
            output_hidden_states=output_hidden_states,  # bool
            return_dict=return_dict,  # bool
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=input_ids.device),
            input_ids.to(torch.int).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state, # torch.Tensor [bs, 77, 768]
            pooler_output=pooled_output, # torch.Tensor [bs, 768]
            hidden_states=encoder_outputs.hidden_states, # None
            attentions=encoder_outputs.attentions, # None
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

### BraVO
class CtxCLIPTextTransformer_stage_1(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.embeddings = CtxCLIPTextEmbeddings(config)

    def forward(
        self,
        ctx_embeddings: torch.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def __build_causal_attention_mask__(bsz, seq_len, dtype) -> torch.Tensor:
            # lazily create causal attention mask, with full attention between the vision tokens
            # pytorch uses additive attention mask; fill with -inf
            mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
            mask.fill_(torch.tensor(torch.finfo(dtype).min))
            mask.triu_(1)  # zero out the lower diagonal
            mask = mask.unsqueeze(1)  # expand mask
            return mask

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # hidden_states.shape = [bs, 77, 768]
        hidden_states = self.embeddings(
            input_ids=input_ids, # torch.Tensor
            ctx_embeddings=ctx_embeddings, # torch.Tensor
            ctx_begin_pos=ctx_begin_pos, # list
        )

        bsz, seq_len = input_shape
        seq_len += ctx_embeddings.size(1)
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = __build_causal_attention_mask__(bsz, seq_len, hidden_states.dtype).to(hidden_states.device)
        return hidden_states, causal_attention_mask

class CtxCLIPTextTransformer_stage_2(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states : torch.Tensor,
        causal_attention_mask : torch.Tensor
    ) -> torch.Tensor:

        encoder_outputs = self.encoder( # transformers.models.clip.modeling_clip.CLIPEncoder
            inputs_embeds=hidden_states,  # torch.Tensor
            causal_attention_mask=causal_attention_mask,  # torch.Tensor
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        # pooled_output = last_hidden_state[
        #     torch.arange(last_hidden_state.shape[0], device=input_ids.device),
        #     input_ids.to(torch.int).argmax(dim=-1),
        # ]
        return last_hidden_state

        # return BaseModelOutputWithPooling(
        #     last_hidden_state=last_hidden_state, # torch.Tensor [bs, 77, 768]
        #     # pooler_output=pooled_output, # torch.Tensor [bs, 768]
        #     # hidden_states=encoder_outputs.hidden_states, # None
        #     # attentions=encoder_outputs.attentions, # None
        # )
    
### BraVO


class CtxCLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(
        self,
        ctx_embeddings: torch.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if ctx_embeddings is None:
            ctx_len = 0
        else:
            ctx_len = ctx_embeddings.shape[1]

        seq_length = (
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        ) + ctx_len

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            # inputs_embeds' shape = [bs, 61, 768]
            inputs_embeds = self.token_embedding(input_ids)
            
            # for each input embeddings, add the ctx embeddings at the correct position
            input_embeds_ctx = []
            bsz = inputs_embeds.shape[0]

            if ctx_embeddings is not None:
                for i in range(bsz):
                    cbp = ctx_begin_pos[i]

                    prefix = inputs_embeds[i, :cbp]
                    # remove the special token embedding
                    suffix = inputs_embeds[i, cbp:]

                    input_embeds_ctx.append(
                        torch.cat([prefix, ctx_embeddings[i], suffix], dim=0)
                    )
                inputs_embeds = torch.stack(input_embeds_ctx, dim=0)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings # [bs, 77, 768]
