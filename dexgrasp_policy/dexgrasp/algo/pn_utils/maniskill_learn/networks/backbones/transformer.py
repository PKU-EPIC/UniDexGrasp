import torch
import torch.nn as nn

from ..builder import BACKBONES, build_backbone
from ..modules import build_attention_layer


class TransformerBlock(nn.Module):
    def __init__(self, attention_cfg, mlp_cfg, dropout=None):
        super().__init__()
        self.attn = build_attention_layer(attention_cfg)
        self.mlp = build_backbone(mlp_cfg)
        assert mlp_cfg['mlp_spec'][0] == mlp_cfg['mlp_spec'][-1] == attention_cfg['embed_dim']
        self.ln = nn.LayerNorm(attention_cfg['embed_dim'])
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x, mask):
        """
        :param x: [B, N, C] [batch size, length, embed_dim]  the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, length, length] a mask for disallowing attention to padding tokens
        :return: [B, N, C] [batch size, length, length] a single tensor containing the output from the Transformer block
        """
        o = self.attn(x, mask)
        x = x + o
        x = self.ln(x)
        o = self.mlp(x)
        o = self.dropout(o)
        x = x + o
        x = self.ln(x)
        return x

@BACKBONES.register_module()
class TransformerEncoder(nn.Module):
    def __init__(self, block_cfg, pooling_cfg, mlp_cfg=None, num_blocks=6):
        super().__init__()
        embed_dim = block_cfg["attention_cfg"]["embed_dim"]
        self.task_embedding = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.xavier_normal_(self.task_embedding)
        self.attn_blocks = nn.ModuleList([TransformerBlock(**block_cfg) for i in range(num_blocks)])
        self.pooling = build_attention_layer(pooling_cfg, default_args=dict(type='AttentionPooling'))
        self.global_mlp = build_backbone(mlp_cfg) if mlp_cfg is not None else None

    def forward(self, x, mask):
        """
        :param x: [B, N, C] [batch size, length, embed_dim] the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, len, len] a mask for disallowing attention to padding tokens.
        :return: [B, F] A single tensor containing the output from the Transformer
        """
        # print('1', x.shape, torch.isnan(x).any())
        one = torch.ones_like(mask[:,:,0])
        mask = torch.cat([one.unsqueeze(1), mask], dim=1) # (B, N+1, N)
        one = torch.ones_like(mask[:,:,0])
        mask = torch.cat([one.unsqueeze(2), mask], dim=2) # (B, N+1, N+1)
        x = torch.cat([torch.repeat_interleave(self.task_embedding, x.size(0), dim=0), x], dim=1)
        for attn in self.attn_blocks:
            x = attn(x, mask)
        x = self.pooling(x, mask[:, -1:])
        if self.global_mlp is not None:
            x = self.global_mlp(x)
        return x


@BACKBONES.register_module()
class TransformerDex(nn.Module):
    def __init__(self, block_cfg, pooling_cfg, mlp_cfg=None, num_blocks=6):
        super().__init__()
        embed_dim = block_cfg["attention_cfg"]["embed_dim"]
        self.task_embedding = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.xavier_normal_(self.task_embedding)
        self.attn_blocks = nn.ModuleList([TransformerBlock(**block_cfg) for i in range(num_blocks)])
        self.pooling = build_attention_layer(pooling_cfg, default_args=dict(type='AttentionPooling'))
        self.global_mlp = build_backbone(mlp_cfg) if mlp_cfg is not None else None

    def forward(self, x, mask):
        """
        :param x: [B, N, C] [batch size, length, embed_dim] the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, len, len] a mask for disallowing attention to padding tokens.
        :return: [B, F] A single tensor containing the output from the Transformer
        """
        # print('1', x.shape, torch.isnan(x).any())
        one = torch.ones_like(mask[:,:,0])
        mask = torch.cat([one.unsqueeze(1), mask], dim=1) # (B, N+1, N)
        one = torch.ones_like(mask[:,:,0])
        mask = torch.cat([one.unsqueeze(2), mask], dim=2) # (B, N+1, N+1)
        x = torch.cat([torch.repeat_interleave(self.task_embedding, x.size(0), dim=0), x], dim=1)
        for attn in self.attn_blocks:
            x = attn(x, mask)
        x = self.pooling(x, mask[:, -1:])
        if self.global_mlp is not None:
            x = self.global_mlp(x)
        return x
