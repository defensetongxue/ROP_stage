import torch
from .layers import *
from functools import partial

class SentenceModel(nn.Module):
    def __init__(self, 
                 patch_embedding_method='resnet50', 
                 image_embedding_method='resnet50',
                 patch_embeddding_dim=128,
                 image_embedding_dim=128,
                 word_size=5, conbine_method='transformer',
                 num_classes=4):
        super().__init__()
        self.patch_embed = CustomPatchEmbed(
            word_size=word_size,hybird_method=patch_embedding_method,patch_embedding_dim=patch_embeddding_dim)
        
        self.image_embed= build_model(model_name=image_embedding_method, dim= image_embedding_dim)
        if conbine_method =='add':
            assert patch_embeddding_dim == image_embedding_dim
            self.classifier=Add(image_embedding_dim,num_classes=num_classes)  
        elif conbine_method == 'concat':
            self.classifier=Concat(
                embed_dim=patch_embeddding_dim,concat_dim=image_embedding_dim+word_size*patch_embeddding_dim,num_classes=num_classes)
            print(image_embedding_dim+word_size*patch_embeddding_dim)
        elif conbine_method == 'transformer':
            assert patch_embeddding_dim == image_embedding_dim
            self.classifier = Transformer(word_size=word_size+1,num_classes=num_classes,embed_dim=image_embedding_dim)
        else:
            raise
    def forward(self, x):
        img,patch,val=x
        img= self.image_embed(img).unsqueeze(1)
        patch=self.patch_embed(patch)
        x=torch.cat([img,patch],dim=1)
        x = self.classifier(x)
        return x # label,patch_label
    
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
class Transformer(nn.Module):
    def __init__(self, word_size,num_classes,embed_dim=1024, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, word_size + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.word_size=word_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.head=nn.Linear(embed_dim,num_classes)
        self.norm = norm_layer(embed_dim)
        self.seghead=nn.Linear(embed_dim,num_classes)
    def forward_features(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x
    
    def forward(self, x):
        head_embed=x[:,1:,:]
        x = self.forward_features(x)
        class_token=x[:,0,:]
        class_ouput = self.head(class_token)
        patch_label=self.seghead(head_embed)
        return class_ouput,patch_label
    
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
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
        
class Concat(nn.Module):
    def __init__(self,embed_dim,concat_dim,num_classes):
        super().__init__()
        self.fc=nn.Linear(concat_dim,num_classes)
        self.patch_head= nn.Linear(embed_dim,num_classes)
        
    def forward(self,x):
        patch_label= self.patch_head(x[:,1:,:])
        
        x= x.flatten(1,-1)
        x= self.fc(x)
        return x,patch_label

class Add(nn.Module):
    def __init__(self,embed_dim,num_classes):
        super().__init__()
        self.w=nn.Linear(embed_dim,1)
        self.b=nn.Linear(embed_dim,1)
        self.fc1=nn.Linear(embed_dim,num_classes)
        
        self.patch_head= nn.Linear(embed_dim,num_classes)
    def forward(self,x):
        patch_label= self.patch_head(x[:,1:,:])
        
        # bc, w, embed_dim
        w=self.w(x) # bc,w,1
        b=self.b(x) # bc,w,1
        x= x*w+b
        x= torch.sum(x,dim=1)
        x=self.fc1(x)
        return x,patch_label
        