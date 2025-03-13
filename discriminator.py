import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

class Discriminator(nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None, direct_num=1):
        super(Discriminator, self).__init__()
        assert in_planes >= 1, "Input dimension must be at least 1"
        _hidden = in_planes if hidden is None else hidden
        self.flatten = nn.Flatten()
        self.attention_path = nn.ModuleDict()
        self.MLP_path = nn.ModuleDict()

        # 注意力路径
        if n_layers > 1:
            self.attention_path.add_module('block1', nn.Sequential(
                nn.Linear(in_planes, _hidden),
                nn.BatchNorm1d(_hidden),
                nn.LeakyReLU(0.2)
            ))
            
            self.attention = nn.MultiheadAttention(
                embed_dim=_hidden,
                num_heads=2,
                batch_first=True
            )
            self.norm = nn.LayerNorm(_hidden)
            self.attention_dropout = nn.Dropout(0.1)

            _attn_hidden = _hidden
            for i in range(1, n_layers-1):
                _attn_hidden = max(1, int(_attn_hidden//1.5)) if hidden is None else hidden
                self.attention_path.add_module(f'block{i+1}', nn.Sequential(
                    nn.Linear(_attn_hidden, _attn_hidden),
                    nn.BatchNorm1d(_attn_hidden),
                    nn.LeakyReLU(0.2)
                ))

        # MLP路径
        self.MLP_path.add_module('block1', nn.Sequential(
            nn.Linear(in_planes, _hidden),
            nn.BatchNorm1d(_hidden),
            nn.LeakyReLU(0.2)
        ))
        
        _MLP_hidden = _hidden
        for i in range(1, n_layers-1):
            _MLP_hidden = max(1, int(_MLP_hidden//1.5)) if hidden is None else hidden
            self.MLP_path.add_module(f'block{i+1}', nn.Sequential(
                nn.Linear(_MLP_hidden, _MLP_hidden),
                nn.BatchNorm1d(_MLP_hidden),
                nn.LeakyReLU(0.2)
            ))

        # 确保尾部层维度有效性
        attn_tail_in = _attn_hidden if n_layers > 1 else _hidden
        mlp_tail_in = _MLP_hidden if n_layers > 1 else _hidden
        
        self.attn_tail = nn.Linear(attn_tail_in, max(1, attn_tail_in // 2))
        self.MLP_tail = nn.Linear(mlp_tail_in, max(1, mlp_tail_in // 2))
        
        fusion_in = max(1, attn_tail_in//2) + max(1, mlp_tail_in//2)
        self.fusion = nn.Linear(fusion_in, direct_num)

        self.apply(init_weight)

    def forward(self, x):
        # 注意力路径
        x = self.flatten(x)
        x_attn = x
        for layer in self.attention_path.values():
            x_attn = layer(x_attn)
            
        if hasattr(self, 'attention'):
            x_reshaped = x_attn.unsqueeze(1)
            attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
            attn_output = attn_output.squeeze(1)
            x_attn = x_attn + self.attention_dropout(attn_output)
            x_attn = self.norm(x_attn)
        
        x_attn = self.attn_tail(x_attn)

        # MLP路径
        x_MLP = x
        for layer in self.MLP_path.values():
            x_MLP = layer(x_MLP)
        x_MLP = self.MLP_tail(x_MLP)

        # 融合输出
        combined = torch.cat([x_attn, x_MLP], dim=1)
        return self.fusion(combined)