import marimo

__generated_with = "0.18.3"
app = marimo.App()


@app.cell
def imports():
    # Librerías necesarias
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import v2
    import torchvision as tv
    #import torchcodec as tc
    import av
    return F, Image, os, plt, torch, tv, v2


@app.cell
def _():
    DINOV3_REPO_DIR = "./src/model/dinov3"  # Ruta al repo clonado

    WEIGHTS_PATH = "/Users/more/Desktop/tfg/models/dinov3/dinov3_vitb16plus_pretrain_lvd1689m-4057cbaa.pth"

    IMAGE_PATH = "/Users/more/Desktop/tfg/datamining/samples/cat.webp"
    MODEL_NAME = "dinov3_vits16plus"  # Cambiar según tu modelo

    WEIGHTS_PATH="/Users/more/Desktop/tfg/models/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    return IMAGE_PATH, WEIGHTS_PATH


@app.cell
def _(WEIGHTS_PATH, os, torch):


    import dotenv
    dotenv.load_dotenv()

    repo = os.getenv("REPO")


    class DinoV3BaseViT:
        @staticmethod
        def from_path(weights: str, **kwargs) :
            model = torch.hub.load(
                repo_or_dir=repo,
                model="dinov3_vitb16",
                weights=weights,
                pretrained=True,
                source="local",
                verbose=False,
                force_reload=True,
                kwargs=kwargs,
            )
            return model

    model = DinoV3BaseViT.from_path(weights=WEIGHTS_PATH).to("mps")
    return (model,)


@app.cell
def _(model):
    model.eval()
    return


@app.cell
def _(model):
    model.num_features
    return


@app.cell
def _(torch, v2):
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize(mean=(0.485, 0.456, 0.406), 
                 std=(0.229, 0.224, 0.225)),])
    return (transform,)


@app.cell
def _(IMAGE_PATH, Image, transform):
    img = Image.open(IMAGE_PATH).convert("RGB")
    img = transform(img).unsqueeze(0).to("mps")
    return (img,)


@app.cell
def _(img, model, torch):
    with torch.no_grad():
        cls_token = model(img) 
        res = model.forward_features(img)

    dim = res
    #res = res.view(dim, -1).permute(1, 0)
    return cls_token, res


@app.cell
def _(cls_token):
    cls_token
    return


@app.cell
def _(res):
    list(res.keys())
    return


@app.cell
def _(res):
    res["x_norm_patchtokens"].shape
    return


@app.cell
def _(cls_token, res, torch):
    torch.mean(cls_token - res["x_norm_clstoken"])
    return


@app.cell
def _(torch, v2):
    trans = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return (trans,)


@app.cell
def _(tv):

    video = tv.io.read_video(filename="./datamining/CNSE/metadata/slices/-9SqXjh8Y-I-slice-0.mp4", pts_unit="sec", output_format="TCHW")
    return (video,)


@app.cell
def _(video):
    video[0].shape
    return


@app.cell
def _(video):
    video[0][0].shape
    return


@app.cell
def _(model, torch, transform, video):
    with torch.no_grad():
        in_video = transform(video[0]).to("mps")
        out=model(in_video)
    return (out,)


@app.cell
def _(out):
    out.shape
    return


@app.cell
def _(video):
    video[-1]
    return


@app.cell
def _(F, model, plt, torch, trans, video):
    import math
    def visualize_attention(model, img_tensor, patch_size=16, device="cuda", output_path="attention_map.png"):
        """
        Visualiza la atención del token CLS del último bloque del modelo.
        Versión robusta que autocalcula el tamaño de la cuadrícula (Grid Size).
        """
        # Asegurar que el modelo esté en evaluación y en el dispositivo correcto
        model.eval()
        model.to(device)
        img_tensor = img_tensor.to(device)

        # Obtener dimensiones originales para calcular el ratio de aspecto
        _, _, H_orig, W_orig = img_tensor.shape

        with torch.no_grad():
            # 1. Preparar tokens (CLS + REGISTERS + PATCHES)
            # prepare_tokens_with_masks maneja el padding/resize interno si lo hay
            x, (H_padded, W_padded) = model.prepare_tokens_with_masks(img_tensor)

            # RoPE (Embeddings posicionales)
            if model.rope_embed is not None:
                rope_sincos = model.rope_embed(H=H_padded, W=W_padded)
            else:
                rope_sincos = None

            # 2. Pasar por los bloques hasta el penúltimo (N-1)
            for i, blk in enumerate(model.blocks[:-1]):
                x = blk(x, rope_sincos)

            # 3. Calcular atención manualmente en el ÚLTIMO bloque
            last_block = model.blocks[-1]

            try:
                # A. Normalización (Pre-Norm)
                # Buscamos 'norm1' o 'norm' o la primera LayerNorm disponible
                norm_layer = getattr(last_block, 'norm1', getattr(last_block, 'norm', None))
                if norm_layer is None:
                    for m in last_block.children():
                        if isinstance(m, (torch.nn.LayerNorm, type(model.norm))):
                            norm_layer = m
                            break

                x_norm = norm_layer(x)

                # B. Proyección QKV
                attn_layer = getattr(last_block, 'attn', None)
                qkv_layer = getattr(attn_layer, 'qkv', None)

                B, N, C = x_norm.shape
                num_heads = model.num_heads
                head_dim = C // num_heads

                # Calcular Q, K, V
                # Shape: [3, B, Heads, N, Head_Dim]
                qkv = qkv_layer(x_norm).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # C. Atención: (Q @ K.T) / scale
                attn_score = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))
                attn_probs = attn_score.softmax(dim=-1) # [B, Heads, N, N]

            except AttributeError as e:
                print(f"❌ Error accediendo a capas internas: {e}")
                return

            # 4. Extraer y procesar el mapa de atención
            n_storage = model.n_storage_tokens

            # Fila 0 (CLS queries), Columnas [1+Registros : Final] (Patch Keys)
            # Esto elimina el CLS y los Registros del mapa visual
            cls_attn = attn_probs[0, :, 0, 1 + n_storage :] # [Heads, Num_Visual_Tokens]

            # Promedio sobre las cabezas (Heads)
            cls_attn_mean = cls_attn.mean(dim=0) # [Num_Visual_Tokens]

            # --- LÓGICA DE RESHAPE ROBUSTA (AQUÍ ESTABA EL ERROR) ---
            num_tokens = cls_attn_mean.shape[0]

            # Calculamos grid aproximado basado en el aspect ratio de la imagen original
            aspect_ratio = W_orig / H_orig

            # w * h = num_tokens  AND  w / h = aspect_ratio
            # w = sqrt(num_tokens * aspect_ratio)
            grid_w = int(math.sqrt(num_tokens * aspect_ratio))
            grid_h = num_tokens // grid_w

            # Ajuste fino si el redondeo falla
            if grid_w * grid_h != num_tokens:
                # Intentamos ajustar h
                grid_h = int(num_tokens / grid_w)
                # Si sigue fallando, probamos w
                if grid_w * grid_h != num_tokens:
                     # Fallback: Asumir cuadrado y recortar si sobra (caso raro)
                     grid_w = int(math.sqrt(num_tokens))
                     grid_h = grid_w
                     print(f"⚠️ Advertencia: Grid irregular ({num_tokens} tokens). Forzando {grid_w}x{grid_h}.")
                     cls_attn_mean = cls_attn_mean[:grid_w*grid_h]

            print(f"✅ Grid detectado: {grid_h}x{grid_w} (Total tokens usados: {grid_w*grid_h})")

            # Reshape final
            attn_map = cls_attn_mean.reshape(grid_h, grid_w).detach().cpu().numpy()

        # 5. Visualización
        plt.figure(figsize=(12, 6))

        # A. Imagen Original
        # Des-normalización estándar de ImageNet para visualización
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        img_disp = img_tensor[0] * std + mean
        img_disp = img_disp.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        plt.subplot(1, 2, 1)
        plt.imshow(img_disp)
        plt.title("Imagen Original")
        plt.axis('off')

        # B. Mapa de Atención
        plt.subplot(1, 2, 2)
        plt.imshow(img_disp)

        # Interpolamos el mapa pequeño (ej: 37x37) al tamaño real (224x224)
        attn_map_resized = F.interpolate(
            torch.tensor(attn_map).unsqueeze(0).unsqueeze(0),
            size=(img_disp.shape[0], img_disp.shape[1]),
            mode='bicubic',
            align_corners=False
        ).squeeze().numpy()

        plt.imshow(attn_map_resized, cmap='inferno', alpha=0.6)
        plt.title("Atención del Token [CLS]")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"💾 Mapa guardado en: {output_path}")
        plt.show()


    # --- USO ---
    visualize_attention(model, trans(video[0][0][0:50,40:170].unsqueeze(0)), device="mps") # Si estás en Mac usa mps
    return (math,)


@app.cell
def _(F, math, model, plt, torch, trans, video):


    def visualize_last_layer_attention(model, img_tensor, output_path="attention_last_layer.png", device="cuda"):
        """
        Visualiza la atención del token [CLS] de la ÚLTIMA capa del modelo.
        NO realiza Rollout, solo mira la decisión final inmediata.

        Args:
            model: El modelo DinoVisionTransformer cargado.
            img_tensor: Tensor (1, 3, H, W) normalizado.
            device: 'cuda', 'mps' (Mac) o 'cpu'.
        """
        model.eval()
        model.to(device)
        img_tensor = img_tensor.to(device)

        # Guardamos dimensiones originales para calcular el aspecto (ancho/alto) luego
        _, _, H_orig, W_orig = img_tensor.shape

        with torch.no_grad():
            # 1. Tokenización y RoPE
            # prepare_tokens_with_masks devuelve los tokens y las dimensiones "interas" (que pueden tener padding)
            x, (H_internal, W_internal) = model.prepare_tokens_with_masks(img_tensor)

            # Preparar embeddings posicionales (RoPE) si el modelo los usa
            rope_sincos = None
            if model.rope_embed is not None:
                rope_sincos = model.rope_embed(H=H_internal, W=W_internal)

            # 2. Pasar por todos los bloques MENOS el último
            # Llevamos la información hasta la puerta del último bloque
            for blk in model.blocks[:-1]:
                x = blk(x, rope_sincos)

            # 3. Introspección manual del ÚLTIMO bloque
            last_block = model.blocks[-1]

            try:
                # A. Normalización (LayerNorm/RMSNorm antes de la atención)
                # Buscamos la capa de norma. En timm suele ser norm1.
                norm_layer = getattr(last_block, 'norm1', getattr(last_block, 'norm', None))
                # Fallback si no tiene nombre estándar
                if norm_layer is None:
                    for m in last_block.children():
                        if isinstance(m, (torch.nn.LayerNorm, type(model.norm))):
                            norm_layer = m
                            break

                x_norm = norm_layer(x)

                # B. Obtener Q, K, V
                # Accedemos a la lineal qkv dentro del bloque de atención
                attn_module = getattr(last_block, 'attn', None)
                qkv_layer = getattr(attn_module, 'qkv', None)

                B, N, C = x_norm.shape
                num_heads = model.num_heads
                head_dim = C // num_heads

                # Calculamos y separamos Q, K, V
                # Shape resultante: [3, Batch, Heads, Tokens, Dim_per_Head]
                qkv = qkv_layer(x_norm).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # C. Calcular Matriz de Atención Cruda
                # Attention = Softmax(Q @ K.T / sqrt(dim))
                scale = head_dim ** -0.5
                attn_scores = (q @ k.transpose(-2, -1)) * scale
                attn_probs = attn_scores.softmax(dim=-1) # [Batch, Heads, Tokens, Tokens]

            except AttributeError as e:
                print(f"❌ Error accediendo a las capas internas del bloque: {e}")
                return

            # 4. Extraer el mapa de atención del CLS
            # El token CLS está en el índice 0.
            # Queremos saber cuánto atiende el CLS (fila 0) a los parches visuales.

            n_storage = model.n_storage_tokens
            # Indices visuales empiezan después del CLS (1) y los registros (n_storage)
            start_visual = 1 + n_storage

            # Tomamos: Batch 0, Todas las Heads, Fila CLS (0), Columnas Visuales
            cls_attn = attn_probs[0, :, 0, start_visual:] # Shape: [Heads, Num_Visual_Tokens]

            # Promediamos sobre todas las cabezas de atención (Heads) para tener un solo mapa
            attn_mean = cls_attn.mean(dim=0) # Shape: [Num_Visual_Tokens]

            # 5. Cálculo Dinámico de la Cuadrícula (Grid Size) - SOLUCIÓN AL ERROR DE SHAPE
            num_tokens = attn_mean.shape[0]
            aspect_ratio = W_orig / H_orig

            # Matemáticamente: w * h = num_tokens  Y  w / h = aspect_ratio
            # Por tanto: w = sqrt(num_tokens * aspect_ratio)
            grid_w = int(math.sqrt(num_tokens * aspect_ratio))
            grid_h = num_tokens // grid_w

            # Corrección por si el redondeo falla (puede sobrar 1 token a veces)
            if grid_w * grid_h != num_tokens:
                print(f"⚠️ Aviso: Grid irregular detectado ({num_tokens} tokens). Ajustando...")
                # Intentamos ajustar dimensiones
                grid_h = int(num_tokens / grid_w)
                # Si sigue sin cuadrar, recortamos el sobrante (rara vez pasa, pero previene crash)
                if grid_w * grid_h != num_tokens:
                     grid_w = int(math.sqrt(num_tokens))
                     grid_h = grid_w
                     attn_mean = attn_mean[:grid_w*grid_h] # Recorte de seguridad

            print(f"✅ Mapa reconstruido: {grid_h}x{grid_w} (Tokens: {num_tokens})")

            # Reshape final al grid 2D
            attn_map_2d = attn_mean.reshape(grid_h, grid_w).detach().cpu().numpy()

        # 6. Visualización
        plt.figure(figsize=(10, 5))

        # A. Imagen Original
        # Des-normalizar para mostrar (Media/Std de ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        img_show = img_tensor[0] * std + mean
        img_show = img_show.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        plt.subplot(1, 2, 1)
        plt.imshow(img_show)
        plt.title("Input")
        plt.axis('off')

        # B. Mapa de Atención Superpuesto
        plt.subplot(1, 2, 2)
        plt.imshow(img_show)

        # Interpolación bicúbica para suavizar los cuadraditos
        attn_map_resized = F.interpolate(
            torch.tensor(attn_map_2d).unsqueeze(0).unsqueeze(0),
            size=(H_orig, W_orig),
            mode='bicubic',
            align_corners=False
        ).squeeze().numpy()

        plt.imshow(attn_map_resized, cmap='inferno', alpha=0.6)
        plt.title("Atención Última Capa [CLS]")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()
        print(f"Guardado en {output_path}")

    # --- EJEMPLO DE USO ---
    # img = torch.randn(1, 3, 224, 224)
    visualize_last_layer_attention(model, trans(video[0][0][0:50,40:170].unsqueeze(0)), device="mps")
    return


if __name__ == "__main__":
    app.run()
