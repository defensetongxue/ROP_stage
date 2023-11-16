import torch
from .layers import *

class SentenceModel(nn.Module):
    def __init__(self, patch_size=16, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., num_patches=5, norm_layer=nn.LayerNorm):
        
        self.patch_embed = CustomPatchEmbed(patch_size=patch_size,embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.head=nn.Linear(embed_dim,num_classes)
        
    def load_pretrained(self, pretrained_path):
        # Load the state dict from the pretrained model
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint['model']
    
        # Custom loading for patch_embed
        patch_embed_state_dict = {k.replace('patch_embed.', ''): v for k, v in state_dict.items() if 'patch_embed' in k}
        self.patch_embed.load_state_dict(patch_embed_state_dict)
    
        # Initialize cls_token and pos_embed with trunc_normal
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
    
        # Load state dict for the rest of the model, excluding cls_token and pos_embed
        model_state_dict = self.state_dict()
        excluded_params = {'cls_token', 'pos_embed'}
        for name, param in state_dict.items():
            if name in excluded_params:
                continue  # Skip loading pretrained weights for excluded params
            if name in model_state_dict:
                if param.shape != model_state_dict[name].shape:
                    print(f"Shape mismatch at: {name}, model: {model_state_dict[name].shape}, loaded: {param.shape}")
                else:
                    model_state_dict[name].copy_(param)
            else:
                print(f"Skipping loading parameter: {name}")
    
        # Initialize the head with trunc_normal
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x