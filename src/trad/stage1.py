import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import MBartForConditionalGeneration, MBartTokenizer


class VisualEmbedding(nn.Module):
    """Visual Embedding module from Figure 3b"""

    def __init__(self, hidden_dim=1024):
        super().__init__()
        # ResNet18 sin la capa FC
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Temporal blocks
        self.temporal_block1 = nn.Sequential(
            nn.Conv1d(512, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.temporal_block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, T, 224, 224, 3)
        B, T, H, W, C = x.shape

        # Procesar cada frame con ResNet
        x = x.view(B * T, C, H, W)
        x = self.resnet(x)  # (B*T, 512, 1, 1)
        x = x.view(B, T, -1)  # (B, T, 512)

        # Temporal modeling
        x = x.permute(0, 2, 1)  # (B, 512, T)
        x = self.temporal_block1(x)  # (B, 1024, T/2)
        x = self.temporal_block2(x)  # (B, 1024, T/4)
        x = x.permute(0, 2, 1)  # (B, T/4, 1024)

        # Linear projection
        B, T_new, D = x.shape
        x = x.reshape(B * T_new, D)
        x = self.linear(x)
        x = x.reshape(B, T_new, D)

        return x


class VisualEncoder(nn.Module):
    """Visual Encoder: Visual Embedding + Transformer Encoder"""

    def __init__(
        self, hidden_dim=1024, num_layers=3, num_heads=8, ff_dim=4096, dropout=0.1
    ):
        super().__init__()
        self.visual_embedding = VisualEmbedding(hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # CLS token para representación global
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x, mask=None):
        # Visual embedding
        x = self.visual_embedding(x)  # (B, T/4, 1024)

        # Añadir CLS token
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T/4+1, 1024)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        return x


class TextEncoder(nn.Module):
    """Text Encoder basado en mBART"""

    def __init__(self, model_name="facebook/mbart-large-cc25"):
        super().__init__()
        # Cargar encoder de mBART
        mbart = MBartForConditionalGeneration.from_pretrained(model_name)
        self.encoder = mbart.get_encoder()
        self.hidden_dim = self.encoder.config.d_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class TextDecoder(nn.Module):
    """Text Decoder basado en mBART para masked language modeling"""

    def __init__(self, model_name="facebook/mbart-large-cc25"):
        super().__init__()
        self.mbart = MBartForConditionalGeneration.from_pretrained(model_name)
        self.hidden_dim = self.mbart.config.d_model

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.mbart(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs


class PositionalEncoding(nn.Module):
    """Positional Encoding para Transformer"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class VLPModel(nn.Module):
    """
    Visual-Language Pretraining Model (Stage 1)
    Combina contrastive learning (CLIP-style) con masked language modeling
    """

    def __init__(self, visual_hidden_dim=1024, text_model="facebook/mbart-large-cc25"):
        super().__init__()

        # Visual Encoder
        self.visual_encoder = VisualEncoder(hidden_dim=visual_hidden_dim)

        # Text Encoder (para contrastive learning)
        self.text_encoder = TextEncoder(text_model)

        # Text Decoder (para masked language modeling)
        self.text_decoder = TextDecoder(text_model)

        # Projection heads para el espacio multimodal compartido
        text_hidden_dim = self.text_encoder.hidden_dim
        self.visual_projection = nn.Linear(visual_hidden_dim, 512)
        self.text_projection = nn.Linear(text_hidden_dim, 512)

        # Temperature parameter para contrastive loss
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(
        self, videos, input_ids, attention_mask, masked_input_ids=None, labels=None
    ):
        """
        Args:
            videos: (B, T, H, W, C) - videos de lenguaje de señas
            input_ids: (B, U) - tokens de texto
            attention_mask: (B, U) - máscara de atención
            masked_input_ids: (B, U) - tokens de texto con máscaras
            labels: (B, U) - etiquetas para MLM
        """
        # 1. Contrastive Learning (CLIP-style)
        # Codificar video
        visual_features = self.visual_encoder(videos)  # (B, T/4+1, 1024)
        visual_cls = visual_features[:, 0, :]  # CLS token

        # Codificar texto
        text_features = self.text_encoder(
            input_ids, attention_mask
        )  # (B, U, hidden_dim)
        # Usar el token EOS (último token válido)
        text_eos_idx = attention_mask.sum(dim=1) - 1
        text_cls = text_features[torch.arange(text_features.size(0)), text_eos_idx]

        # Proyectar al espacio compartido
        visual_embed = F.normalize(self.visual_projection(visual_cls), dim=-1)
        text_embed = F.normalize(self.text_projection(text_cls), dim=-1)

        # Calcular contrastive loss
        contrastive_loss = self.contrastive_loss(visual_embed, text_embed)

        # 2. Masked Language Modeling
        mlm_loss = None
        if masked_input_ids is not None and labels is not None:
            decoder_outputs = self.text_decoder(
                input_ids=masked_input_ids, attention_mask=attention_mask, labels=labels
            )
            mlm_loss = decoder_outputs.loss

        return {
            "contrastive_loss": contrastive_loss,
            "mlm_loss": mlm_loss,
            "visual_embed": visual_embed,
            "text_embed": text_embed,
        }

    def contrastive_loss(self, visual_embed, text_embed):
        """
        Symmetric cross-entropy loss (Ecuación 3 del paper)
        """
        # Calcular similitudes
        logits = torch.matmul(visual_embed, text_embed.t()) / self.temperature

        # Labels: diagonal (pares correctos)
        labels = torch.arange(logits.size(0), device=logits.device)

        # Loss simétrico
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)

        loss = (loss_v2t + loss_t2v) / 2
        return loss


def mask_tokens(input_ids, tokenizer, mask_prob=0.15):
    """
    Aplica masking a los tokens según la estrategia BERT
    80% -> [MASK], 10% -> token aleatorio, 10% -> sin cambio
    """
    labels = input_ids.clone()

    # Crear máscara de probabilidad
    probability_matrix = torch.full(labels.shape, mask_prob)

    # No enmascarar tokens especiales
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Solo calcular loss en tokens enmascarados

    # 80% de las veces: reemplazar con [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% de las veces: reemplazar con token aleatorio
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # 10% restante: mantener sin cambio

    return input_ids, labels


# Función de entrenamiento
def train_vlp_stage1(
    model, dataloader, optimizer, scheduler, device, epochs=80, lambda_mlm=0.1
):
    """
    Entrena el modelo VLP (Stage 1)

    Args:
        model: VLPModel
        dataloader: DataLoader con (videos, input_ids, attention_mask)
        optimizer: optimizador
        scheduler: learning rate scheduler
        device: dispositivo (cuda/cpu)
        epochs: número de épocas
        lambda_mlm: peso para el MLM loss (Ecuación 9)
    """
    model.train()
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

    for epoch in range(epochs):
        total_loss = 0
        total_contrastive = 0
        total_mlm = 0

        for batch_idx, batch in enumerate(dataloader):
            videos = batch["videos"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Crear versión enmascarada del texto
            masked_input_ids, labels = mask_tokens(input_ids.clone(), tokenizer)

            # Forward pass
            outputs = model(
                videos=videos,
                input_ids=input_ids,
                attention_mask=attention_mask,
                masked_input_ids=masked_input_ids,
                labels=labels,
            )

            # Combinar losses (Ecuación 9)
            loss = outputs["contrastive_loss"] + lambda_mlm * outputs["mlm_loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Estadísticas
            total_loss += loss.item()
            total_contrastive += outputs["contrastive_loss"].item()
            total_mlm += outputs["mlm_loss"].item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Contrastive: {outputs['contrastive_loss'].item():.4f}, "
                    f"MLM: {outputs['mlm_loss'].item():.4f}"
                )

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        avg_contrastive = total_contrastive / len(dataloader)
        avg_mlm = total_mlm / len(dataloader)

        print(f"\nEpoch [{epoch + 1}/{epochs}] Summary:")
        print(
            f"Avg Loss: {avg_loss:.4f}, Avg Contrastive: {avg_contrastive:.4f}, Avg MLM: {avg_mlm:.4f}\n"
        )


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crear modelo
    model = VLPModel(visual_hidden_dim=1024).to(device)

    # Optimizador (según paper: SGD con momentum 0.9)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )

    # Scheduler (cosine annealing según paper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=80, eta_min=1e-5
    )

    print("Modelo VLP Stage 1 creado exitosamente")
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
