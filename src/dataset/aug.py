from .load import cv2, np, RGBVideoLoader

# Obtenido  de https://www.kaggle.com/code/ahmedabdelfattah20/image-augmentation-using-opencv


class Augmentation:
    def __init__(
        self,
        original: list[np.ndarray],
        blurred: list[np.ndarray],
        noised: list[np.ndarray],
        warped: list[np.ndarray],
        resized: list[np.ndarray],
    ) -> None:
        self.original: list[np.ndarray] = original
        self.blurred: list[np.ndarray] = blurred
        self.noised: list[np.ndarray] = noised
        self.warped: list[np.ndarray] = warped
        self.resized: list[np.ndarray] = resized


class VideoAugmentator:
    def __init__(self) -> None:
        pass

    def __call__(self, dataloader: RGBVideoLoader) -> Augmentation:
        return Augmentation(
            original=dataloader.videos,
            blurred=[
                np.array([self.Blur(frame) for frame in video])
                for video in dataloader.videos
            ],
            noised=[
                np.array([self.RandomNoise(frame) for frame in video])
                for video in dataloader.videos
            ],
            warped=[
                np.array([self.WarpAffine(frame) for frame in video])
                for video in dataloader.videos
            ],
            resized=[
                np.array([self.RandomResize(frame) for frame in video])
                for video in dataloader.videos
            ],
        )

    @staticmethod
    def Blur(frame: np.ndarray):
        ksize = np.random.randint(5, 15)
        return cv2.blur(frame, (ksize, ksize))

    @staticmethod
    def RandomNoise(frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        h = (h + np.random.randint(0, 100, h.shape, np.uint8)) % 180
        s = np.clip(s + np.random.randint(0, 20, s.shape, np.uint8), 0, 255)
        v = np.clip(v + np.random.randint(0, 10, v.shape, np.uint8), 0, 255)

        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def WarpAffine(frame: np.ndarray):
        shape: tuple[int, int, int] = frame.shape
        (cols, rows, _) = shape
        cbound: int = int(cols * 0.25)
        rbound: int = int(rows * 0.25)

        tx: int = np.random.randint(-cbound, cbound)
        ty: int = np.random.randint(-rbound, rbound)
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        return cv2.warpAffine(frame, M, (cols, rows))

    @staticmethod
    def RandomResize(frame: np.ndarray):
        shape: tuple[int, int, int] = frame.shape
        (cols, rows, _) = shape
        cbound: int = int(cols * 0.25)
        rbound: int = int(rows * 0.25)

        tx: int = np.random.randint(-cbound, cbound)
        ty: int = np.random.randint(-rbound, rbound)
        x, y = max(tx, 0), max(ty, 0)
        w, h = cols - abs(tx), rows - abs(ty)
        rsize = frame[y : y + h, x : x + w]
        rsize = cv2.resize(rsize, (cols, rows))
        return rsize
