# /// script
# dependencies = ["kagglehub", "albumentations", "torchmetrics"]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    # packages added via marimo's package management: kagglehub albumentations !pip install -q kagglehub albumentations
    # packages added via marimo's package management: torchmetrics !pip install torchmetrics

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import kagglehub
    import os
    import cv2
    import numpy as np
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return (
        A,
        DataLoader,
        Dataset,
        F,
        ToTensorV2,
        cv2,
        kagglehub,
        nn,
        np,
        os,
        torch,
    )


@app.cell
def _(kagglehub):
    # Descarga del dataset
    path = kagglehub.dataset_download("tapakah68/segmentation-full-body-tiktok-dancing-dataset")
    print("Path to dataset files:", path)
    return (path,)


@app.cell
def _(os, path):
    print(os.listdir(path ))
    print(os.listdir(os.path.join(path, "segmentation_full_body_tik_tok_2615_img", "segmentation_full_body_tik_tok_2615_img")))
    # Configuración de rutas
    IMAGES_PATH = os.path.join(path, "segmentation_full_body_tik_tok_2615_img", "segmentation_full_body_tik_tok_2615_img", "images")
    MASKS_PATH = os.path.join(path, "segmentation_full_body_tik_tok_2615_img", "segmentation_full_body_tik_tok_2615_img", "masks")
    return IMAGES_PATH, MASKS_PATH


@app.cell
def _(Dataset, cv2, np, os):
    class TikTokDataset(Dataset):
        def __init__(self, images_dir, masks_dir, transform=None):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
            self.image_files = sorted(os.listdir(images_dir))
            self.transform = transform

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_name = self.image_files[idx]
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, img_name) # Usualmente mismo nombre

            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32) # Binarizar

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            return image, mask.unsqueeze(0)
    return (TikTokDataset,)


@app.cell
def _(A, DataLoader, IMAGES_PATH, MASKS_PATH, TikTokDataset, ToTensorV2):
    # Transformaciones (DINO espera normalización ImageNet)
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    dataset = TikTokDataset(IMAGES_PATH, MASKS_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    return (dataloader,)


@app.cell
def _(nn, torch):
    WEIGHTS = "/Users/more/Desktop/tfg/models/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
    class DINOv3Segmentation(nn.Module):
        def __init__(self, out_channels=1):
            super().__init__()
            # Cargar el backbone DINOv3 Small
            # Nota: Asegúrate de tener conexión a internet o el repo clonado
            self.backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16plus', weights=WEIGHTS,
                    pretrained=True,
                    source="github",
                    verbose=False,
                    force_reload=True
                    )

            # El embedding de ViT-S es 384.
            # Congelamos el backbone al principio para no destruir los pesos pre-entrenados
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Decoder simple: DINO da features de 16x16 parches.
            # Para 512x512, el mapa de features es 32x32 (512/16 = 32)
            self.decoder = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 64x64
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), # 256x256
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 512x512
                nn.Conv2d(64, out_channels, kernel_size=1)
            )

        def forward(self, x):
            # DINOv3 devuelve las features de los parches
            # n=1 significa la última capa. Reshape=True nos da [B, C, H_patch, W_patch]
            features = self.backbone.get_intermediate_layers(x, n=1, reshape=True)[0]
            out = self.decoder(features)
            return out

    device = torch.device("mps")
    model = DINOv3Segmentation().to(device)
    return device, model


@app.cell
def _(device):
    device
    return


@app.cell
def _(F, dataloader, device, model, torch):
    from torch import optim

    def dice_loss(pred, target, smooth=1.):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1. - dice.mean()

    # Bucle de entrenamiento actualizado
    model.train()
    optimizer = optim.AdamW(model.decoder.parameters(), lr=2e-4, weight_decay=1e-2)
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Combinamos BCE y Dice para mejores bordes
            loss_bce = F.binary_cross_entropy_with_logits(outputs, masks)
            loss_dice = dice_loss(outputs, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataloader):.4f}")
    return


if __name__ == "__main__":
    app.run()
