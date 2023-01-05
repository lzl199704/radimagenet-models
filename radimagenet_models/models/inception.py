import errno
import os
from typing import Optional

import gdown
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import inception_v3

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = inception_v3(pretrained=False, aux_logits=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:20])
                        
    def forward(self, x):
        return self.backbone(x)

def inception_v3(
    model_dir: Optional[str] = None,
    file_name: str = "RadImageNet-InceptionV3_notop.pth",
    progress: bool = True,
):
    if model_dir is None:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        gdown.download(
            url="",
            output=cached_file,
            quiet=not progress,
        )

    model = Backbone()
    model.load_state_dict(torch.load(cached_file))
    return model
