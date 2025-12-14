import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2


# Cargar el modelo DINOv3 desde archivo .pt local
def load_dinov3_model(weights_path, model_name="dinov3_vitb16", repo_dir="."):
    """
    Carga el modelo DINOv3 desde un checkpoint local
    """
    model = torch.hub.load(
        repo_dir, model_name, source="local", weights=weights_path, pretrained=False
    )
    model.eval()
    return model


# Preprocesar imagen según DINOv3
def make_transform(resize_size=None, patch_size=16):
    """
    Transform para modelos DINOv3
    Si resize_size=None, mantiene tamaño original (gracias a RoPE)
    """
    transforms_list = [v2.ToImage()]

    if resize_size is not None:
        if isinstance(resize_size, int):
            resize_size = (resize_size // patch_size) * patch_size
            resize_size = (resize_size, resize_size)
        transforms_list.append(v2.Resize(resize_size, antialias=True))

    transforms_list.extend(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return v2.Compose(transforms_list)


def preprocess_image(image_path, resize_size=None, patch_size=16):
    """
    Carga y preprocesa una imagen
    """
    img = Image.open(image_path).convert("RGB")

    if resize_size is None:
        w, h = img.size
        new_w = (w // patch_size) * patch_size
        new_h = (h // patch_size) * patch_size
        if new_w != w or new_h != h:
            print(
                f"Ajustando imagen de {w}x{h} a {new_w}x{new_h} (múltiplo de {patch_size})"
            )
            img = img.resize((new_w, new_h), Image.BILINEAR)

    transform = make_transform(resize_size, patch_size)
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img


# Extraer attention maps directamente modificando el forward pass
def get_attention_maps_v2(model, image_tensor):
    """
    Extrae attention maps usando forward hooks mejorados
    Compatible con xformers y diferentes backends de atención
    """
    attention_maps = []

    def hook_fn_forward_qkv(module, input, output):
        """Hook que captura las atenciones después de softmax"""
        # Algunos modelos devuelven la atención como parte del output
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]  # El segundo elemento suele ser la atención
            if attn is not None:
                attention_maps.append(attn.detach())

    # Registrar hooks en todos los bloques
    hooks = []
    for i, block in enumerate(model.blocks):
        # Intentar diferentes lugares donde puede estar la atención
        if hasattr(block, "attn"):
            # Hook en el módulo de atención completo
            hook = block.attn.register_forward_hook(hook_fn_forward_qkv)
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = model.forward_features(image_tensor)

    # Remover hooks
    for hook in hooks:
        hook.remove()

    return attention_maps


# Método alternativo: modificar temporalmente el modelo
def get_attention_maps_v3(model, image_tensor, patch_size=16):
    """
    Método que calcula attention maps manualmente desde las features
    No requiere hooks ni modificaciones del modelo
    """
    attention_maps = []

    # Guardar el output de cada capa
    layer_outputs = []

    def hook_fn(module, input, output):
        # output es un tensor [B, N, D] donde N = num_patches + 1
        layer_outputs.append(output[0].detach())  # Solo batch 0

    hooks = []
    for block in model.blocks:
        hook = block.register_forward_hook(hook_fn)
        hooks.append(hook)

    with torch.no_grad():
        _ = model(image_tensor)

    for hook in hooks:
        hook.remove()

    # Calcular "pseudo-attention" basado en similitud de features
    img_h, img_w = image_tensor.shape[2], image_tensor.shape[3]
    h = img_h // patch_size
    w = img_w // patch_size

    for layer_feat in layer_outputs:
        # layer_feat: [num_patches+1, dim]
        if layer_feat.numel() == 0:
            print(f"   ⚠️  Capa vacía detectada, saltando...")
            continue

        cls_token = layer_feat[0:1]  # [1, dim]
        patch_tokens = layer_feat[1:]  # [num_patches, dim]

        if patch_tokens.numel() == 0:
            print(f"   ⚠️  No hay patch tokens, saltando capa...")
            continue

        # Similitud coseno entre CLS y cada patch
        cls_norm = F.normalize(cls_token, dim=-1)  # [1, dim]
        patch_norm = F.normalize(patch_tokens, dim=-1)  # [num_patches, dim]

        # Producto punto: [1, dim] @ [dim, num_patches] = [1, num_patches]
        similarity = (cls_norm @ patch_norm.mT).squeeze(0)  # [num_patches]

        # Aplicar softmax para simular atención
        pseudo_attention = F.softmax(similarity, dim=0)
        attention_maps.append(pseudo_attention.cpu())

    return attention_maps, (h, w)


# Visualizar attention maps
def visualize_attention_maps(
    image, attention_maps, grid_shape, num_layers=6, save_prefix="attention"
):
    """
    Visualiza attention maps de múltiples capas
    """
    h, w = grid_shape
    num_layers = min(num_layers, len(attention_maps))

    fig, axes = plt.subplots(2, num_layers, figsize=(4 * num_layers, 8))
    if num_layers == 1:
        axes = axes.reshape(2, 1)

    layer_indices = np.linspace(0, len(attention_maps) - 1, num_layers, dtype=int)

    for idx, layer_idx in enumerate(layer_indices):
        attn = attention_maps[layer_idx].numpy()

        # Verificar que no esté vacío
        if attn.size == 0:
            print(f"⚠️  Capa {layer_idx}: attention map vacío, saltando")
            axes[0, idx].axis("off")
            axes[1, idx].axis("off")
            continue

        # Verificar tamaño
        expected_size = h * w
        if len(attn) != expected_size:
            print(
                f"⚠️  Capa {layer_idx}: esperados {expected_size} patches, hay {len(attn)}"
            )
            # Ajustar dimensiones si no coinciden
            actual_h = actual_w = int(np.sqrt(len(attn)))
            if actual_h * actual_w != len(attn):
                print(f"   No es cuadrado, usando dimensiones aproximadas")
                actual_h = int(np.sqrt(len(attn)))
                actual_w = len(attn) // actual_h
            attn_map = attn[: actual_h * actual_w].reshape(actual_h, actual_w)
        else:
            attn_map = attn.reshape(h, w)

        # Normalizar
        attn_map = (attn_map - attn_map.min()) / (
            attn_map.max() - attn_map.min() + 1e-8
        )

        # Imagen original
        axes[0, idx].imshow(image)
        axes[0, idx].set_title(f"Capa {layer_idx}")
        axes[0, idx].axis("off")

        # Attention map
        im = axes[1, idx].imshow(attn_map, cmap="viridis")
        axes[1, idx].set_title(f"Atención CLS→Patches")
        axes[1, idx].axis("off")
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    filename = f"{save_prefix}_grid.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"✓ Guardado: {filename}")
    plt.show()


def visualize_attention_overlay(
    image, attention_map, grid_shape, layer_name="última", save_prefix="attention"
):
    """
    Superpone un attention map sobre la imagen
    """
    h, w = grid_shape
    attn = attention_map.numpy()

    # Verificar y ajustar tamaño
    expected_size = h * w
    if len(attn) != expected_size:
        actual_h = actual_w = int(np.sqrt(len(attn)))
        attn_map = attn.reshape(actual_h, actual_w)
    else:
        attn_map = attn.reshape(h, w)

    # Normalizar
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Redimensionar al tamaño de la imagen
    attn_resized = np.array(
        Image.fromarray(attn_map).resize(image.size, resample=Image.BILINEAR)
    )

    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(attn_resized, cmap="jet")
    axes[1].set_title(f"Attention Map ({layer_name} capa)")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(attn_resized, cmap="jet", alpha=0.6)
    axes[2].set_title("Superposición")
    axes[2].axis("off")

    plt.tight_layout()
    filename = f"{save_prefix}_overlay.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"✓ Guardado: {filename}")
    plt.show()


# Script principal
if __name__ == "__main__":
    # Configuración
    DINOV3_REPO_DIR = "./src/dinov3"  # Ruta al repo clonado
    WEIGHTS_PATH = (
        "./models/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"  # Tu archivo .pt
    )
    IMAGE_PATH = "/Users/more/Desktop/tfg/datamining/samples/cat.webp"
    MODEL_NAME = "dinov3_vith16plus"  # Cambiar según tu modelo
    PATCH_SIZE = 16

    print("=" * 60)
    print("EXTRACCIÓN DE ATTENTION MAPS - DINOv3")
    print("=" * 60)

    print("\n1. Cargando modelo DINOv3...")
    model = load_dinov3_model(WEIGHTS_PATH, MODEL_NAME, DINOV3_REPO_DIR)
    print(f"   Modelo: {MODEL_NAME}")

    print("\n2. Preprocesando imagen...")
    # Usar resolución original (RoPE permite esto)
    image_tensor, original_image = preprocess_image(
        IMAGE_PATH, resize_size=None, patch_size=PATCH_SIZE
    )
    print(f"   Resolución: {original_image.size}")
    print(f"   Tensor shape: {image_tensor.shape}")

    print("\n3. Extrayendo attention maps...")
    print("   Usando método de pseudo-attention (similitud de features)")

    try:
        attention_maps, grid_shape = get_attention_maps_v3(
            model, image_tensor, PATCH_SIZE
        )
        print(f"   ✓ Extraídos {len(attention_maps)} mapas de atención")
        print(
            f"   ✓ Grid de patches: {grid_shape[1]}x{grid_shape[0]} = {grid_shape[0] * grid_shape[1]}"
        )

        if attention_maps:
            print("\n4. Generando visualizaciones...")

            # Grid de múltiples capas
            visualize_attention_maps(
                original_image,
                attention_maps,
                grid_shape,
                num_layers=6,
                save_prefix="attention",
            )

            # Overlay de la última capa
            visualize_attention_overlay(
                original_image,
                attention_maps[-1],
                grid_shape,
                layer_name="última",
                save_prefix="attention",
            )

            # Overlay de una capa intermedia
            mid_layer = len(attention_maps) // 2
            visualize_attention_overlay(
                original_image,
                attention_maps[mid_layer],
                grid_shape,
                layer_name=f"intermedia ({mid_layer})",
                save_prefix="attention_mid",
            )

            print("\n" + "=" * 60)
            print("✓ PROCESO COMPLETADO")
            print("=" * 60)
            print("\nArchivos generados:")
            print("  - attention_grid.png")
            print("  - attention_overlay.png")
            print("  - attention_mid_overlay.png")

        else:
            print("   ⚠️  No se pudieron extraer attention maps")

    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback

        traceback.print_exc()
