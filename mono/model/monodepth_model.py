import torch
import torch.nn as nn
from .model_pipelines.__base_model__ import BaseDepthModel

class DepthModel(BaseDepthModel):
    def __init__(self, cfg, **kwards):
        super(DepthModel, self).__init__(cfg)
        model_type = cfg.model.type
        
    def inference(self, data):
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.forward(data)
        return pred_depth, confidence, output_dict

def get_monodepth_model(
    cfg : dict,
    **kwargs
    ) -> nn.Module:
    model = DepthModel(cfg, **kwargs)
    assert isinstance(model, nn.Module)
    return model

def get_configured_monodepth_model(
    cfg: dict,
    ) -> nn.Module:
    model = get_monodepth_model(cfg)
    return model
