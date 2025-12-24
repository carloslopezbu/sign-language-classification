import os
from typing import cast

import dotenv
import torch
import torch.nn.functional as F
from dinov3.dinov3.models.vision_transformer import DinoVisionTransformer
from torch import Tensor, nn

dotenv.load_dotenv()

repo = os.getenv("REPO")


class DinoV3BaseViT:
    @staticmethod
    def from_weights(weights: str, **kwargs) -> DinoVisionTransformer:
        model = torch.hub.load(
            repo_or_dir=repo,
            model="dinov3_vitb16",
            weights=weights,
            pretrained=True,
            source="local",
            verbose=True,
            force_reload=True,
            kwargs=kwargs,
        )
        return cast(DinoVisionTransformer, model)


class FaceVisualEncoder(nn.Module):
    pass


class HandVisualEncoder(nn.Module):
    pass


class TemporalUnit(nn.Module):
    pass


dino = DinoV3BaseViT.from_weights("")
for par in dino.parameters():
    par.requires_grad = False
