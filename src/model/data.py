import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class VideoAugmentation:
    """
    Strong data augmentation para videos de lenguaje de señas
    Implementa las transformaciones mencionadas en el paper (Sección 3.1)
    """

    def __init__(self, mode="train", img_size=224):
        self.mode = mode
        self.img_size = img_size

        if mode == "train":
            # Strong augmentation para Stage 1 y Stage 2
            self.spatial_transform = A.Compose(
                [
                    A.Resize(256, 256),
                    A.RandomCrop(img_size, img_size),
                    # Geometric transformations
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5
                    ),
                    # Color space transformations
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=0.5
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.5,
                    ),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            # Inference: solo resize y center crop
            self.spatial_transform = A.Compose(
                [
                    A.Resize(256, 256),
                    A.CenterCrop(img_size, img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def temporal_augmentation(self, frames, mode="train"):
        """
        Temporal transformations: temporal sampling
        """
        if mode != "train":
            return frames

        # Temporal subsampling aleatorio
        if random.random() < 0.3 and len(frames) > 16:
            # Mantener al menos 16 frames
            step = random.randint(2, 3)
            frames = frames[::step]

        return frames

    def __call__(self, frames):
        """
        Aplicar augmentation a un video completo

        Args:
            frames: List de frames (numpy arrays) [T, H, W, C]
        Returns:
            frames: Tensor [T, C, H, W]
        """
        # Temporal augmentation
        frames = self.temporal_augmentation(frames, self.mode)

        # Spatial augmentation por frame
        augmented_frames = []
        for frame in frames:
            # Aplicar transformaciones espaciales
            augmented = self.spatial_transform(image=frame)
            augmented_frames.append(augmented["image"])

        # Stack frames: [T, C, H, W]
        frames_tensor = torch.stack(augmented_frames)

        return frames_tensor


class SignLanguageDataset(Dataset):
    """
    Dataset para Sign Language Translation

    Estructura esperada:
    dataset_root/
    ├── videos/
    │   ├── train/
    │   │   ├── video_001.mp4
    │   │   ├── video_002.mp4
    │   │   └── ...
    │   ├── dev/
    │   └── test/
    └── annotations/
        ├── train.json
        ├── dev.json
        └── test.json

    Formato de annotations JSON:
    [
        {
            "video_id": "video_001",
            "video_path": "train/video_001.mp4",
            "text": "das wetter ist heute sonnig",
            "num_frames": 120
        },
        ...
    ]
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        tokenizer=None,
        max_video_length: int = 300,
        max_text_length: int = 100,
        augmentation: bool = True,
        img_size: int = 224,
        sample_rate: int = 1,
    ):
        """
        Args:
            dataset_root: ruta raíz del dataset
            split: 'train', 'dev', o 'test'
            tokenizer: tokenizador para el texto
            max_video_length: longitud máxima de frames de video
            max_text_length: longitud máxima de tokens de texto
            augmentation: aplicar data augmentation
            img_size: tamaño de imagen (224 según paper)
            sample_rate: muestrear cada N frames (1 = todos los frames)
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.tokenizer = tokenizer
        self.max_video_length = max_video_length
        self.max_text_length = max_text_length
        self.sample_rate = sample_rate

        # Cargar anotaciones
        annotation_file = self.dataset_root / "annotations" / f"{split}.json"
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        print(f"Loaded {len(self.annotations)} samples for {split} split")

        # Augmentation
        aug_mode = "train" if (split == "train" and augmentation) else "test"
        self.transform = VideoAugmentation(mode=aug_mode, img_size=img_size)

        # Tokens especiales
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def __len__(self):
        return len(self.annotations)

    def load_video(self, video_path: str) -> List[np.ndarray]:
        """
        Cargar video y extraer frames

        Args:
            video_path: ruta al video
        Returns:
            frames: lista de frames [H, W, C] en formato RGB
        """
        full_path = self.dataset_root / "videos" / video_path

        cap = cv2.VideoCapture(str(full_path))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {full_path}")

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Muestrear según sample_rate
            if frame_count % self.sample_rate == 0:
                # Convertir BGR a RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            frame_count += 1

            # Limitar longitud máxima
            if len(frames) >= self.max_video_length:
                break

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No se pudieron extraer frames del video: {full_path}")

        return frames

    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenizar texto y preparar para el modelo

        Args:
            text: texto a tokenizar
        Returns:
            dict con 'input_ids' y 'attention_mask'
        """
        if self.tokenizer is None:
            # Tokenización simple por palabras (para testing)
            tokens = text.lower().split()
            input_ids = (
                [self.bos_token_id]
                + [hash(t) % 10000 for t in tokens]
                + [self.eos_token_id]
            )
        else:
            # Usar tokenizador real
            encoded = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            return {"input_ids": input_ids, "attention_mask": attention_mask}

        # Padding manual
        if len(input_ids) < self.max_text_length:
            padding_length = self.max_text_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_length
        else:
            input_ids = input_ids[: self.max_text_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Obtener un sample del dataset

        Returns:
            dict con:
                - video: [T, C, H, W]
                - input_ids: [max_text_length]
                - attention_mask: [max_text_length]
                - video_length: escalar (T)
                - text: string original
                - video_id: string
        """
        annotation = self.annotations[idx]

        # Cargar video
        frames = self.load_video(annotation["video_path"])

        # Aplicar augmentation
        video_tensor = self.transform(frames)  # [T, C, H, W]

        # Tokenizar texto
        text_data = self.tokenize_text(annotation["text"])

        return {
            "video": video_tensor,
            "input_ids": text_data["input_ids"],
            "attention_mask": text_data["attention_mask"],
            "video_length": video_tensor.shape[0],
            "text": annotation["text"],
            "video_id": annotation["video_id"],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function para crear batches con padding dinámico

    Args:
        batch: lista de samples del dataset
    Returns:
        batch dict con tensors paddeados
    """
    # Obtener longitudes máximas
    max_video_length = max([item["video_length"] for item in batch])

    # Preparar tensors
    videos = []
    input_ids = []
    attention_masks = []
    video_padding_masks = []
    text_padding_masks = []
    texts = []
    video_ids = []

    for item in batch:
        video = item["video"]  # [T, C, H, W]
        T, C, H, W = video.shape

        # Padding de video
        if T < max_video_length:
            padding = torch.zeros(max_video_length - T, C, H, W)
            video = torch.cat([video, padding], dim=0)
            # Máscara de padding (True = padding)
            video_mask = torch.cat(
                [
                    torch.zeros(T, dtype=torch.bool),
                    torch.ones(max_video_length - T, dtype=torch.bool),
                ]
            )
        else:
            video_mask = torch.zeros(T, dtype=torch.bool)

        videos.append(video)
        input_ids.append(item["input_ids"])
        attention_masks.append(item["attention_mask"])
        video_padding_masks.append(video_mask)
        text_padding_masks.append(item["attention_mask"] == 0)  # True = padding
        texts.append(item["text"])
        video_ids.append(item["video_id"])

    # Stack en batches
    batch_data = {
        "videos": torch.stack(videos),  # [B, T, C, H, W]
        "input_ids": torch.stack(input_ids),  # [B, U]
        "attention_mask": torch.stack(attention_masks),  # [B, U]
        "video_padding_mask": torch.stack(video_padding_masks),  # [B, T]
        "text_padding_mask": torch.stack(text_padding_masks),  # [B, U]
        "texts": texts,
        "video_ids": video_ids,
    }

    # Transponer videos a [B, T, H, W, C] para el modelo
    batch_data["videos"] = batch_data["videos"].permute(0, 1, 3, 4, 2)

    return batch_data


def create_annotation_file(
    video_dir: str, text_file: str, output_file: str, video_extension: str = ".mp4"
):
    """
    Crear archivo de anotaciones desde videos y archivo de texto

    Args:
        video_dir: directorio con videos
        text_file: archivo con textos (formato: video_id|texto)
        output_file: ruta de salida para JSON
        video_extension: extensión de los videos

    Formato del text_file:
    video_001|das wetter ist heute sonnig
    video_002|morgen wird es regnen
    ...
    """
    video_dir = Path(video_dir)
    annotations = []

    # Leer archivo de texto
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) != 2:
                print(f"Skipping invalid line: {line}")
                continue

            video_id, text = parts
            video_path = video_dir / f"{video_id}{video_extension}"

            if not video_path.exists():
                print(f"Warning: Video not found: {video_path}")
                continue

            # Obtener número de frames
            cap = cv2.VideoCapture(str(video_path))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            annotations.append(
                {
                    "video_id": video_id,
                    "video_path": str(video_path.relative_to(video_dir.parent)),
                    "text": text,
                    "num_frames": num_frames,
                }
            )

    # Guardar JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"Created annotation file with {len(annotations)} entries: {output_file}")


# Ejemplo de uso
if __name__ == "__main__":
    from transformers import MBartTokenizer

    # 1. Crear archivo de anotaciones (si es necesario)
    """
    create_annotation_file(
        video_dir='path/to/videos/train',
        text_file='path/to/train.txt',
        output_file='path/to/annotations/train.json'
    )
    """

    # 2. Crear dataset
    # Para Stage 1 (VLP)
    print("=" * 50)
    print("Stage 1 Dataset (VLP)")
    print("=" * 50)

    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

    train_dataset_stage1 = SignLanguageDataset(
        dataset_root="path/to/dataset",
        split="train",
        tokenizer=tokenizer,
        max_video_length=300,
        max_text_length=100,
        augmentation=True,  # Strong augmentation para Stage 1
        img_size=224,
        sample_rate=1,
    )

    train_loader_stage1 = DataLoader(
        train_dataset_stage1,
        batch_size=16,  # Batch size de 16 según paper
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Para Stage 2 (GFSLT)
    print("\n" + "=" * 50)
    print("Stage 2 Dataset (GFSLT)")
    print("=" * 50)

    train_dataset_stage2 = SignLanguageDataset(
        dataset_root="path/to/dataset",
        split="train",
        tokenizer=tokenizer,
        max_video_length=300,
        max_text_length=100,
        augmentation=True,  # También con augmentation para Stage 2
        img_size=224,
        sample_rate=1,
    )

    train_loader_stage2 = DataLoader(
        train_dataset_stage2,
        batch_size=8,  # Batch size de 8 según paper
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 4. Visualizar un sample
    print("\n" + "=" * 50)
    print("Sample Example")
    print("=" * 50)

    sample = train_dataset_stage1[0]
    print(f"Video shape: {sample['video'].shape}")  # [T, C, H, W]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Text: {sample['text']}")
    print(f"Video ID: {sample['video_id']}")

    # 5. Visualizar un batch
    print("\n" + "=" * 50)
    print("Batch Example")
    print("=" * 50)

    batch = next(iter(train_loader_stage1))
    print(f"Batch videos shape: {batch['videos'].shape}")  # [B, T, H, W, C]
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")  # [B, U]
    print(f"Batch video_padding_mask shape: {batch['video_padding_mask'].shape}")
    print(f"Number of texts: {len(batch['texts'])}")

    # 6. Crear DataLoader para validación/test (sin augmentation)
    print("\n" + "=" * 50)
    print("Validation Dataset")
    print("=" * 50)

    val_dataset = SignLanguageDataset(
        dataset_root="path/to/dataset",
        split="dev",
        tokenizer=tokenizer,
        max_video_length=300,
        max_text_length=100,
        augmentation=False,  # Sin augmentation para validación
        img_size=224,
        sample_rate=1,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Validation dataset size: {len(val_dataset)}")
    print("\nDataset setup completado!")
