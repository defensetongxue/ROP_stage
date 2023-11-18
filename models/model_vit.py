import torch
from .layers import *
from functools import partial

class SentenceModel(nn.Module):
    def __init__(self, patch_size=16, num_classes=1000,embed_dim=1024, depth=24,
                 num_heads=16, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., num_patches=5, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
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
        self.head=nn.Linear(embed_dim+num_patches,num_classes)
        self.norm = norm_layer(embed_dim)
    def load_pretrained(self, pretrained_path):
        # Load the state dict from the pretrained model
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint['model']
    
        # Custom loading for patch_embed
    
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
                if name.startswith("decoder"):
                    continue
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
    
    def _visual(self, x_val):
        x, val = x_val
        B = x.shape[0]
        x = self.patch_embed(x)
    
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
    
        # Process through all blocks except the last
        for blk in self.blocks[:-1]:
            x = blk(x)
    
        # Process through the last block and capture attention scores
        last_block = self.blocks[-1]
        x, att_scores = last_block(x, return_attention_scores=True)
    
        x = self.norm(x)
        final_output = x[:, 0]
    
        # Average attention scores across heads and extract scores for the class token
        avg_att_scores = att_scores.mean(dim=1)  # Averaging across the num_heads dimension
        cls_token_att_scores = avg_att_scores[:, 0, 1:]  # Scores of class token attending to patches
    
        # Calculate order of contribution
        att_order = cls_token_att_scores.argsort(dim=-1, descending=True)
    
        # Forward the class token to the head
        final_output = torch.cat([final_output, val], dim=1)
        final_output = self.head(final_output)
    
        return final_output, att_order
    def forward(self, x_val):
        x,val=x_val
        x = self.forward_features(x)
        x=torch.cat([x,val],dim=1)
        x = self.head(x)
        return x