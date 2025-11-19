import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import cv2

    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    __all__ = ["InceptionI3d"]


    class MaxPool3dSamePadding(nn.MaxPool3d):
        def compute_pad(self, dim, s):
            if s % self.stride[dim] == 0:
                return max(self.kernel_size[dim] - self.stride[dim], 0)
            else:
                return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

        def forward(self, x):
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)

            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            pad_h_f = pad_h // 2
            pad_h_b = pad_h - pad_h_f
            pad_w_f = pad_w // 2
            pad_w_b = pad_w - pad_w_f

            pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
            x = F.pad(x, pad)
            return super(MaxPool3dSamePadding, self).forward(x)


    class Unit3D(nn.Module):
        def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1),
                     stride=(1, 1, 1), activation_fn=F.relu, use_batch_norm=True,
                     use_bias=False):
            super(Unit3D, self).__init__()

            self._output_channels = output_channels
            self._kernel_shape = kernel_shape
            self._stride = stride
            self._use_batch_norm = use_batch_norm
            self._activation_fn = activation_fn
            self._use_bias = use_bias

            self.conv3d = nn.Conv3d(
                in_channels=in_channels,
                out_channels=self._output_channels,
                kernel_size=self._kernel_shape,
                stride=self._stride,
                padding=0,
                bias=self._use_bias,
            )

            if self._use_batch_norm:
                self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

        def compute_pad(self, dim, s):
            if s % self._stride[dim] == 0:
                return max(self._kernel_shape[dim] - self._stride[dim], 0)
            else:
                return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

        def forward(self, x):
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)

            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            pad_h_f = pad_h // 2
            pad_h_b = pad_h - pad_h_f
            pad_w_f = pad_w // 2
            pad_w_b = pad_w - pad_w_f

            pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
            x = F.pad(x, pad)
            x = self.conv3d(x)

            if self._use_batch_norm:
                x = self.bn(x)
            if self._activation_fn is not None:
                x = self._activation_fn(x)
            return x


    class InceptionModule(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(InceptionModule, self).__init__()

            # Branch 0: 1x1x1 conv
            self.b0 = Unit3D(in_channels, out_channels[0], kernel_shape=[1, 1, 1])

            # Branch 1: 1x1x1 conv -> 3x3x3 conv
            self.b1a = Unit3D(in_channels, out_channels[1], kernel_shape=[1, 1, 1])
            self.b1b = Unit3D(out_channels[1], out_channels[2], kernel_shape=[3, 3, 3])

            # Branch 2: 1x1x1 conv -> 3x3x3 conv
            self.b2a = Unit3D(in_channels, out_channels[3], kernel_shape=[1, 1, 1])
            self.b2b = Unit3D(out_channels[3], out_channels[4], kernel_shape=[3, 3, 3])

            # Branch 3: MaxPool -> 1x1x1 conv
            self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
            self.b3b = Unit3D(in_channels, out_channels[5], kernel_shape=[1, 1, 1])

        def forward(self, x):
            b0 = self.b0(x)
            b1 = self.b1b(self.b1a(x))
            b2 = self.b2b(self.b2a(x))
            b3 = self.b3b(self.b3a(x))
            return torch.cat([b0, b1, b2, b3], dim=1)


    class InceptionI3d(nn.Module):
        """I3D: Inflated 3D ConvNet para clasificación de videos."""

        def __init__(self, num_classes=400, spatiotemporal_squeeze=True,
                     in_channels=3, dropout_keep_prob=0.5, num_in_frames=64,
                     include_embds=False):
            super().__init__()

            self._num_classes = num_classes
            self._spatiotemporal_squeeze = spatiotemporal_squeeze
            self.include_embds = include_embds

            # Configuración de la arquitectura (más compacto)
            # Stem
            self.stem = nn.Sequential(
                Unit3D(in_channels, 64, kernel_shape=[7, 7, 7], stride=(2, 2, 2)),
                MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0),
                Unit3D(64, 64, kernel_shape=[1, 1, 1]),
                Unit3D(64, 192, kernel_shape=[3, 3, 3]),
                MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0),
            )

            # Inception modules (organizados por bloques)
            self.inception_3 = nn.Sequential(
                InceptionModule(192, [64, 96, 128, 16, 32, 32]),   # 3b: 256
                InceptionModule(256, [128, 128, 192, 32, 96, 64]),  # 3c: 480
            )

            self.pool_4a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)

            self.inception_4 = nn.Sequential(
                InceptionModule(480, [192, 96, 208, 16, 48, 64]),   # 4b: 512
                InceptionModule(512, [160, 112, 224, 24, 64, 64]),  # 4c: 512
                InceptionModule(512, [128, 128, 256, 24, 64, 64]),  # 4d: 512
                InceptionModule(512, [112, 144, 288, 32, 64, 64]),  # 4e: 528
                InceptionModule(528, [256, 160, 320, 32, 128, 128]), # 4f: 832
            )

            self.pool_5a = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)

            self.inception_5 = nn.Sequential(
                InceptionModule(832, [256, 160, 320, 32, 128, 128]), # 5b: 832
                InceptionModule(832, [384, 192, 384, 48, 128, 128]), # 5c: 1024
            )

            # Head
            last_duration = int(math.ceil(num_in_frames / 8))
            last_size = 7
            self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
            self.dropout = nn.Dropout(dropout_keep_prob)
            self.logits = Unit3D(1024, self._num_classes, kernel_shape=[1, 1, 1],
                                activation_fn=None, use_batch_norm=False, use_bias=True)

        def forward(self, x):
            x = self.stem(x)
            x = self.inception_3(x)
            x = self.pool_4a(x)
            x = self.inception_4(x)
            x = self.pool_5a(x)
            x = self.inception_5(x)

            embds = self.dropout(self.avgpool(x))
            x = self.logits(embds)

            if self._spatiotemporal_squeeze:
                logits = x.squeeze(3).squeeze(3).squeeze(2)

            if self.include_embds:
                return {"logits": logits, "embds": embds}
            else:
                return {"logits": logits}

        def load_old_state_dict(self, old_state_dict):
            """Carga un state_dict del modelo original (verboso) al simplificado."""
            # Mapeo: nombre_antiguo -> nombre_nuevo
            name_mapping = {
                # Stem
                'Conv3d_1a_7x7': 'stem.0',
                'MaxPool3d_2a_3x3': 'stem.1',
                'Conv3d_2b_1x1': 'stem.2',
                'Conv3d_2c_3x3': 'stem.3',
                'MaxPool3d_3a_3x3': 'stem.4',

                # Inception 3
                'Mixed_3b': 'inception_3.0',
                'Mixed_3c': 'inception_3.1',

                # Pool 4a
                'MaxPool3d_4a_3x3': 'pool_4a',

                # Inception 4
                'Mixed_4b': 'inception_4.0',
                'Mixed_4c': 'inception_4.1',
                'Mixed_4d': 'inception_4.2',
                'Mixed_4e': 'inception_4.3',
                'Mixed_4f': 'inception_4.4',

                # Pool 5a
                'MaxPool3d_5a_2x2': 'pool_5a',

                # Inception 5
                'Mixed_5b': 'inception_5.0',
                'Mixed_5c': 'inception_5.1',

                # Head (avgpool y dropout no tienen params)
                'logits': 'logits',
            }

            # Crear nuevo state_dict con nombres mapeados
            new_state_dict = {}
            for old_name, param in old_state_dict.items():
                # Buscar el prefijo que coincida
                new_name = old_name
                for old_prefix, new_prefix in name_mapping.items():
                    if old_name.startswith(old_prefix + '.'):
                        # Reemplazar el prefijo
                        new_name = old_name.replace(old_prefix + '.', new_prefix + '.')
                        break
                    elif old_name == old_prefix:
                        new_name = new_prefix
                        break

                new_state_dict[new_name] = param

            # Cargar el state_dict mapeado
            self.load_state_dict(new_state_dict, strict=True)
            return self
    return F, InceptionI3d, cv2, math, np, torch


@app.cell
def _():
    checkpoint: str = 'i3d.pth.tar'
    return (checkpoint,)


@app.cell
def _(InceptionI3d, checkpoint: str, torch):
    model = InceptionI3d(num_classes=1064)

    state_dict = torch.load(checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model = model.load_old_state_dict(state_dict)
    device = torch.device('cpu')
    model = model.to(device)
    torch.compile(model=model, backend='mps')
    model.eval()
    print(model)
    return (model,)


@app.cell
def _(np, torch):
    def im_to_numpy(img):
        img = to_numpy(img)
        img = np.transpose(img, (1, 2, 0))  # H*W*C
        return img


    def im_to_torch(img):
        img = np.transpose(img, (2, 0, 1))  # C*H*W
        img = to_torch(img).float()
        if img.max() > 1:
            img /= 255
        return img


    def to_numpy(tensor):
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        elif type(tensor).__module__ != "numpy":
            raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
        return tensor


    def to_torch(ndarray):
        if type(ndarray).__module__ == "numpy":
            return torch.from_numpy(ndarray)
        elif not torch.is_tensor(ndarray):
            raise ValueError(f"Cannot convert {type(ndarray)} to torch tensor")
        return ndarray


    def color_normalize(x, mean, std):
        """Normalize a tensor of images by subtracting the mean, dividing by std. dev.
        """
        if x.dim() in {3, 4}:
            if x.size(0) == 1:
                x = x.repeat(3, 1, 1)
            assert x.size(0) == 3, "For single video format, expected RGB along first dim"
            for t, m, s in zip(x, mean, std):
                t.sub_(m)
                t.div_(s)
        elif x.dim() == 5:
            assert (
                x.shape[1] == 3
            ), "For batched video format, expected RGB along second dim"
            x[:, 0].sub_(mean[0]).div_(std[0])
            x[:, 1].sub_(mean[1]).div_(std[1])
            x[:, 2].sub_(mean[2]).div_(std[2])
        return x
    return color_normalize, im_to_numpy, im_to_torch, to_torch


@app.cell
def _(color_normalize, cv2, im_to_numpy, math, np, to_torch, torch):



    def prepare_input(
        rgb: torch.Tensor,
        resize_res: int = 256,
        inp_res: int = 224,
        mean: torch.Tensor = 0.5 * torch.ones(3),
        std=1.0 * torch.ones(3),
    ):
        """
        Process the video:
        1) Resize to [resize_res x resize_res]
        2) Center crop with [inp_res x inp_res]
        3) Color normalize using mean/std
        """
        iC, iF, iH, iW = rgb.shape
        # Resize
        rgb_resized = np.zeros((iF, resize_res, resize_res, iC))
        for t in range(iF):
            tmp = rgb[:, t, :, :]
            rgb_resized[t] = cv2.resize(im_to_numpy(tmp), (resize_res, resize_res))

        rgb = np.transpose(rgb_resized, (3, 0, 1, 2))
        # Center crop coords
        ulx = int((resize_res - inp_res) / 2)
        uly = int((resize_res - inp_res) / 2)
        # Crop 256x256
        rgb = rgb[:, :, uly : uly + inp_res, ulx : ulx + inp_res]
        rgb = to_torch(rgb).float()
        assert rgb.max() <= 1
        rgb = color_normalize(rgb, mean, std)
        return rgb

    def sliding_windows(rgb: torch.Tensor, num_in_frames: int, stride: int,) -> tuple:
        """
        Return sliding windows and corresponding (middle) timestamp
        """
        C, nFrames, H, W = rgb.shape
        # If needed, pad to the minimum clip length
        if nFrames < num_in_frames:
            rgb_ = torch.zeros(C, num_in_frames, H, W)
            rgb_[:, :nFrames] = rgb
            rgb_[:, nFrames:] = rgb[:, -1].unsqueeze(1)
            rgb = rgb_
            nFrames = rgb.shape[1]

        num_clips = math.ceil((nFrames - num_in_frames) / stride) + 1
        plural = ""
        if num_clips > 1:
            plural = "s"
        print(f"{num_clips} clip{plural} resulted from sliding window processing.")

        rgb_slided = torch.zeros(num_clips, 3, num_in_frames, H, W)
        t_mid = []
        # For each clip
        for j in range(num_clips):
            # Check if num_clips becomes 0
            actual_clip_length = min(num_in_frames, nFrames - j * stride)
            if actual_clip_length == num_in_frames:
                t_beg = j * stride
            else:
                t_beg = nFrames - num_in_frames
            t_mid.append(t_beg + num_in_frames / 2)
            rgb_slided[j] = rgb[:, t_beg : t_beg + num_in_frames, :, :]
        return rgb_slided, np.array(t_mid)
    return prepare_input, sliding_windows


@app.cell
def _(cv2, im_to_torch, torch):
    import shutil
    import subprocess
    from pathlib import Path



    def load_rgb_video(video_path: Path, fps: int) -> torch.Tensor:
        """
        Load a video as a torch tensor (3, T, H, W). If the video FPS does not match
        the target FPS, create a temporary ffmpeg copy with the desired frame rate.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if abs(cap_fps - fps) > 1e-2:
            tmp_video = video_path.with_suffix(f".tmp{video_path.suffix}")
            shutil.copy(video_path, tmp_video)
            cmd = [
                "ffmpeg", "-y", "-i", str(tmp_video),
                "-pix_fmt", "yuv420p",
                "-filter:v", f"fps=fps={fps}",
                str(video_path)
            ]
            print(f"Generating new copy of video with frame rate {fps}")
            subprocess.run(cmd, check=True)
            tmp_video.unlink(missing_ok=True)
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            assert abs(cap_fps - fps) < 1e-2, f"ffmpeg failed to produce {fps} FPS video"

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, :, [2, 1, 0]]  # BGR → RGB
            frames.append(im_to_torch(frame))
        cap.release()

        if not frames:
            raise ValueError(f"No frames read from {video_path}")

        rgb = torch.stack(frames).permute(1, 0, 2, 3)
        print(f"Loaded {len(frames)} frames from {video_path} ({cap_height}x{cap_width} @ {cap_fps:.2f} fps)")
        return rgb
    return (load_rgb_video,)


@app.cell
def _(F, video):
    # Prepare for interpolation: add batch dim → (1, 3, T, H, W)
    video_5d = video.unsqueeze(0)

    # Resize: keep same T, change H,W to 256
    T = video_5d.shape[2]
    video_resized = F.interpolate(video_5d, size=(T, 256, 256), mode='trilinear', align_corners=False)

    # Remove batch dim → (3, T, 256, 256)
    video_resized = video_resized.squeeze(0)
    return


@app.cell
def _(load_rgb_video, prepare_input, sliding_windows, torch):


    # video: (3, T, H, W)
    video = load_rgb_video("maldicion16.mp4", 16)
    video = prepare_input(video)

    slices = sliding_windows(video, 16,1)
    print(slices[0].shape)
    frames16fps = slices[0].to(torch.device('cpu'))

    return frames16fps, video


@app.cell
def _(frames16fps, model, torch):

    if False:
        with torch.no_grad():
            outs = model(frames16fps)
    return (outs,)


@app.cell
def _(outs):
    outs
    return


if __name__ == "__main__":
    app.run()
