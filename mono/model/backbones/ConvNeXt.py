import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 **kwargs,):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        return features

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

def convnext_tiny(pretrained=True,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs['checkpoint'], map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {}
        unmatched_pretrained_dict = {}
        for k, v in checkpoint['model'].items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                unmatched_pretrained_dict[k] = v
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(
            'Successfully loaded pretrained %d params, and %d paras are unmatched.'
            %(len(pretrained_dict.keys()), len(unmatched_pretrained_dict.keys())))
        print('Unmatched pretrained paras are :', unmatched_pretrained_dict.keys())
    return model

def convnext_small(pretrained=True,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs['checkpoint'], map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {}
        unmatched_pretrained_dict = {}
        for k, v in checkpoint['model'].items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                unmatched_pretrained_dict[k] = v
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(
            'Successfully loaded pretrained %d params, and %d paras are unmatched.'
            %(len(pretrained_dict.keys()), len(unmatched_pretrained_dict.keys())))
        print('Unmatched pretrained paras are :', unmatched_pretrained_dict.keys())
    return model

def convnext_base(pretrained=True, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs['checkpoint'], map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {}
        unmatched_pretrained_dict = {}
        for k, v in checkpoint['model'].items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                unmatched_pretrained_dict[k] = v
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(
            'Successfully loaded pretrained %d params, and %d paras are unmatched.'
            %(len(pretrained_dict.keys()), len(unmatched_pretrained_dict.keys())))
        print('Unmatched pretrained paras are :', unmatched_pretrained_dict.keys())
    return model

def convnext_large(pretrained=True, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs['checkpoint'], map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {}
        unmatched_pretrained_dict = {}
        for k, v in checkpoint['model'].items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                unmatched_pretrained_dict[k] = v
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(
            'Successfully loaded pretrained %d params, and %d paras are unmatched.'
            %(len(pretrained_dict.keys()), len(unmatched_pretrained_dict.keys())))
        print('Unmatched pretrained paras are :', unmatched_pretrained_dict.keys())
    return model

def convnext_xlarge(pretrained=True, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        checkpoint = torch.load(kwargs['checkpoint'], map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {}
        unmatched_pretrained_dict = {}
        for k, v in checkpoint['model'].items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                unmatched_pretrained_dict[k] = v
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(
            'Successfully loaded pretrained %d params, and %d paras are unmatched.'
            %(len(pretrained_dict.keys()), len(unmatched_pretrained_dict.keys())))
        print('Unmatched pretrained paras are :', unmatched_pretrained_dict.keys())
    return model

if __name__ == '__main__':
    import torch
    model = convnext_base(True, in_22k=False).cuda()

    rgb = torch.rand((2, 3, 256, 256)).cuda()
    out = model(rgb)
    print(len(out))
    for i, ft in enumerate(out):
        print(i, ft.shape)
