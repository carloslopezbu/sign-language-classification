import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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


class VisualEncoder(nn.Module):
    """
    Visual Encoder: Visual Embedding + Transformer Encoder
    En Stage 2 se hereda de Stage 1 y se fine-tunea completamente
    """

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
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, src_key_padding_mask=None):
        # Visual embedding
        x = self.visual_embedding(x)  # (B, T/4, 1024)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        return x


class TextDecoder(nn.Module):
    """
    Text Decoder: Word Embedding + Transformer Decoder
    Se hereda de Stage 1 (mBART decoder)
    """

    def __init__(
        self,
        vocab_size,
        hidden_dim=1024,
        num_layers=3,
        num_heads=8,
        ff_dim=4096,
        dropout=0.1,
        max_len=200,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Word Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        """
        Args:
            tgt: (B, U) - target sequence tokens
            memory: (B, M, hidden_dim) - encoder output
            tgt_mask: causal mask para autoregresivo
            memory_key_padding_mask: máscara de padding del encoder
            tgt_key_padding_mask: máscara de padding del decoder
        """
        # Word embedding
        tgt = self.embedding(tgt)  # (B, U, hidden_dim)

        # Positional encoding
        tgt = self.pos_encoder(tgt)

        # Transformer decoder
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, U, hidden_dim)

        # Project to vocabulary
        logits = self.output_projection(output)  # (B, U, vocab_size)

        return logits

    def generate_square_subsequent_mask(self, sz, device):
        """Genera máscara causal para decoding autoregresivo"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask


class GFSLTModel(nn.Module):
    """
    Gloss-Free Sign Language Translation Model (Stage 2)
    Arquitectura encoder-decoder basada en Transformer (Figura 3a)
    """

    def __init__(
        self,
        vocab_size,
        visual_hidden_dim=1024,
        text_hidden_dim=1024,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=8,
        ff_dim=4096,
        dropout=0.1,
        label_smoothing=0.2,
    ):
        super().__init__()

        # Visual Encoder (hereda de Stage 1)
        self.visual_encoder = VisualEncoder(
            hidden_dim=visual_hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # Text Decoder (hereda de Stage 1)
        self.text_decoder = TextDecoder(
            vocab_size=vocab_size,
            hidden_dim=text_hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        self.label_smoothing = label_smoothing

        # Tokens especiales
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2

    def forward(
        self, videos, target_ids, video_padding_mask=None, target_padding_mask=None
    ):
        """
        Forward pass para entrenamiento

        Args:
            videos: (B, T, H, W, C) - videos de lenguaje de señas
            target_ids: (B, U) - secuencia objetivo (con BOS al inicio)
            video_padding_mask: (B, T/4) - máscara de padding para video
            target_padding_mask: (B, U) - máscara de padding para target
        """
        # Encode video (Ecuación 5)
        encoder_output = self.visual_encoder(
            videos, src_key_padding_mask=video_padding_mask
        )
        # encoder_output: (B, M, hidden_dim) donde M = T/4

        # Prepare decoder input (shift right)
        decoder_input = target_ids[:, :-1]  # Remover último token
        target_output = target_ids[:, 1:]  # Remover BOS

        # Crear máscaras
        device = videos.device
        tgt_seq_len = decoder_input.size(1)
        tgt_mask = self.text_decoder.generate_square_subsequent_mask(
            tgt_seq_len, device
        )

        # Ajustar padding masks si existen
        if target_padding_mask is not None:
            decoder_padding_mask = target_padding_mask[:, :-1]
        else:
            decoder_padding_mask = None

        # Decode (Ecuación 6)
        logits = self.text_decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=video_padding_mask,
            tgt_key_padding_mask=decoder_padding_mask,
        )  # (B, U-1, vocab_size)

        # Calculate loss (Ecuación 8)
        loss = self.compute_loss(logits, target_output, decoder_padding_mask)

        return {"logits": logits, "loss": loss, "encoder_output": encoder_output}

    def compute_loss(self, logits, targets, padding_mask=None):
        """
        Calcula cross-entropy loss con label smoothing
        """
        # Reshape para calcular loss
        vocab_size = logits.size(-1)
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Aplicar label smoothing
        if self.label_smoothing > 0:
            loss = self.label_smoothing_loss(logits_flat, targets_flat, padding_mask)
        else:
            loss = F.cross_entropy(
                logits_flat, targets_flat, ignore_index=self.pad_idx, reduction="mean"
            )

        return loss

    def label_smoothing_loss(self, logits, targets, padding_mask=None):
        """Label smoothing loss"""
        vocab_size = logits.size(-1)

        # Crear distribución suavizada
        confidence = 1.0 - self.label_smoothing
        smoothing_value = self.label_smoothing / (vocab_size - 1)

        # One-hot encoding
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        smooth_one_hot = one_hot * confidence + (1 - one_hot) * smoothing_value

        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Loss
        loss = -(smooth_one_hot * log_probs).sum(dim=-1)

        # Ignorar padding
        if padding_mask is not None:
            mask = ~padding_mask.reshape(-1)
            loss = loss.masked_select(mask).mean()
        else:
            mask = targets != self.pad_idx
            loss = loss.masked_select(mask).mean()

        return loss

    @torch.no_grad()
    def beam_search(
        self,
        videos,
        beam_size=5,
        max_len=100,
        length_penalty=1.0,
        video_padding_mask=None,
    ):
        """
        Beam search decoding para inferencia (Sección 4.2)

        Args:
            videos: (B, T, H, W, C) - un solo video (B=1)
            beam_size: tamaño del beam
            max_len: longitud máxima de generación
            length_penalty: penalización por longitud
            video_padding_mask: máscara de padding
        """
        self.eval()
        device = videos.device

        # Encode video
        encoder_output = self.visual_encoder(
            videos, src_key_padding_mask=video_padding_mask
        )
        # encoder_output: (1, M, hidden_dim)

        # Expandir para beam search
        encoder_output = encoder_output.repeat(
            beam_size, 1, 1
        )  # (beam_size, M, hidden_dim)
        if video_padding_mask is not None:
            video_padding_mask = video_padding_mask.repeat(beam_size, 1)

        # Inicializar beam con BOS token
        beam_seqs = torch.full(
            (beam_size, 1), self.bos_idx, dtype=torch.long, device=device
        )
        beam_scores = torch.zeros(beam_size, device=device)
        completed_seqs = []
        completed_scores = []

        for step in range(max_len):
            # Crear máscaras
            tgt_seq_len = beam_seqs.size(1)
            tgt_mask = self.text_decoder.generate_square_subsequent_mask(
                tgt_seq_len, device
            )

            # Forward pass
            logits = self.text_decoder(
                tgt=beam_seqs,
                memory=encoder_output,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=video_padding_mask,
            )  # (beam_size, seq_len, vocab_size)

            # Obtener logits del último token
            next_token_logits = logits[:, -1, :]  # (beam_size, vocab_size)
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            # Calcular scores
            # Aplicar length penalty: score = log_prob / length^penalty
            current_length = beam_seqs.size(1)
            scores = beam_scores.unsqueeze(1) + log_probs
            scores = scores / (current_length**length_penalty)

            # Flatten scores para seleccionar top-k
            scores_flat = scores.view(-1)  # (beam_size * vocab_size)

            # Seleccionar top beam_size candidatos
            top_scores, top_indices = scores_flat.topk(beam_size)

            # Convertir índices a (beam_idx, token_idx)
            beam_indices = top_indices // logits.size(-1)
            token_indices = top_indices % logits.size(-1)

            # Construir nuevos beams
            new_beam_seqs = []
            new_beam_scores = []

            for i in range(beam_size):
                beam_idx = beam_indices[i].item()
                token_idx = token_indices[i].item()
                score = top_scores[i].item()

                # Añadir token a la secuencia
                seq = torch.cat(
                    [beam_seqs[beam_idx], torch.tensor([token_idx], device=device)]
                )

                # Si es EOS, añadir a completados
                if token_idx == self.eos_idx:
                    completed_seqs.append(seq)
                    completed_scores.append(score * (current_length**length_penalty))
                else:
                    new_beam_seqs.append(seq)
                    new_beam_scores.append(
                        score * ((current_length + 1) ** length_penalty)
                    )

            # Si no quedan beams activos, terminar
            if len(new_beam_seqs) == 0:
                break

            # Actualizar beams
            beam_seqs = torch.nn.utils.rnn.pad_sequence(
                new_beam_seqs, batch_first=True, padding_value=self.pad_idx
            )
            beam_scores = torch.tensor(new_beam_scores, device=device)

            # Actualizar encoder output si el número de beams cambió
            if beam_seqs.size(0) < beam_size:
                encoder_output = encoder_output[: beam_seqs.size(0)]
                if video_padding_mask is not None:
                    video_padding_mask = video_padding_mask[: beam_seqs.size(0)]

        # Añadir beams restantes a completados
        for i in range(len(new_beam_seqs)):
            completed_seqs.append(new_beam_seqs[i])
            completed_scores.append(new_beam_scores[i])

        # Seleccionar mejor secuencia
        if len(completed_seqs) == 0:
            return beam_seqs[0]

        best_idx = torch.tensor(completed_scores).argmax().item()
        return completed_seqs[best_idx]

    @torch.no_grad()
    def translate(
        self, videos, tokenizer, beam_size=5, max_len=100, length_penalty=1.0
    ):
        """
        Traducir un video de lenguaje de señas a texto

        Args:
            videos: (1, T, H, W, C) - video de entrada
            tokenizer: tokenizer para decodificar
            beam_size: tamaño del beam
            max_len: longitud máxima
            length_penalty: penalización por longitud
        """
        # Beam search
        output_ids = self.beam_search(videos, beam_size, max_len, length_penalty)

        # Decodificar
        output_ids = output_ids.cpu().numpy()
        translation = tokenizer.decode(output_ids, skip_special_tokens=True)

        return translation

    def load_pretrained_stage1(self, stage1_model_path):
        """
        Cargar parámetros pre-entrenados de Stage 1
        """
        print("Cargando parámetros de Stage 1...")
        checkpoint = torch.load(stage1_model_path, map_location="cpu")

        # Cargar Visual Encoder
        visual_encoder_state = {}
        for key, value in checkpoint.items():
            if key.startswith("visual_encoder."):
                new_key = key.replace("visual_encoder.", "")
                # No cargar CLS token si existe
                if "cls_token" not in new_key:
                    visual_encoder_state[new_key] = value

        self.visual_encoder.load_state_dict(visual_encoder_state, strict=False)
        print(f"✓ Visual Encoder cargado ({len(visual_encoder_state)} parámetros)")

        # Cargar Text Decoder (si es compatible)
        text_decoder_state = {}
        for key, value in checkpoint.items():
            if key.startswith("text_decoder."):
                new_key = key.replace("text_decoder.", "")
                text_decoder_state[new_key] = value

        if len(text_decoder_state) > 0:
            self.text_decoder.load_state_dict(text_decoder_state, strict=False)
            print(f"✓ Text Decoder cargado ({len(text_decoder_state)} parámetros)")
        else:
            print("⚠ Text Decoder no encontrado en checkpoint de Stage 1")

        print("Parámetros de Stage 1 cargados exitosamente")


def train_gfslt_stage2(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=200,
    save_path="gfslt_stage2.pt",
):
    """
    Entrenar GFSLT Model (Stage 2)

    Args:
        model: GFSLTModel
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        optimizer: optimizador
        scheduler: learning rate scheduler
        device: dispositivo
        epochs: número de épocas (200 según paper)
        save_path: ruta para guardar el modelo
    """
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            videos = batch["videos"].to(device)
            target_ids = batch["target_ids"].to(device)
            video_padding_mask = batch.get("video_padding_mask", None)
            target_padding_mask = batch.get("target_padding_mask", None)

            if video_padding_mask is not None:
                video_padding_mask = video_padding_mask.to(device)
            if target_padding_mask is not None:
                target_padding_mask = target_padding_mask.to(device)

            # Forward pass
            outputs = model(videos, target_ids, video_padding_mask, target_padding_mask)
            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / train_steps

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                videos = batch["videos"].to(device)
                target_ids = batch["target_ids"].to(device)
                video_padding_mask = batch.get("video_padding_mask", None)
                target_padding_mask = batch.get("target_padding_mask", None)

                if video_padding_mask is not None:
                    video_padding_mask = video_padding_mask.to(device)
                if target_padding_mask is not None:
                    target_padding_mask = target_padding_mask.to(device)

                outputs = model(
                    videos, target_ids, video_padding_mask, target_padding_mask
                )
                val_loss += outputs["loss"].item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps

        # Update learning rate
        scheduler.step()

        print(f"\nEpoch [{epoch + 1}/{epochs}] Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✓ Mejor modelo guardado en {save_path}\n")


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 2887  # Para PHOENIX14T (alemán)

    # Crear modelo
    model = GFSLTModel(
        vocab_size=vocab_size,
        visual_hidden_dim=1024,
        text_hidden_dim=1024,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=8,
        ff_dim=4096,
        dropout=0.1,
        label_smoothing=0.2,
    ).to(device)

    # Cargar parámetros pre-entrenados de Stage 1
    # model.load_pretrained_stage1('vlp_stage1.pt')

    # Optimizador (SGD con momentum 0.9)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0
    )

    # Scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=1e-6
    )

    print("Modelo GFSLT Stage 2 creado exitosamente")
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Ejemplo de inferencia
    print("\nEjemplo de uso para inferencia:")
    print("translation = model.translate(video_tensor, tokenizer, beam_size=5)")
