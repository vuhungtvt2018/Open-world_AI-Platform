# embedding.py
import torch
import timm
import numpy as np
from timm.data import resolve_model_data_config
from typing import Optional
from PIL import Image
import torchvision.transforms as T
from config import settings

class ImageEmbedder:
    def __init__(self, model_name: str = None, device: Optional[str] = None):
        # Chọn device
        self.device = device or (settings.DEVICE if torch.cuda.is_available() or settings.DEVICE == "cpu" else "cpu")

        # Khởi tạo model DINOv2 (trả về feature, dùng CLS/token pool)
        self.model_name = model_name or settings.MODEL_NAME
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=0,          # -> forward() trả về features
            global_pool='token'     # lấy CLS token (ổn định cho embedding)
        )
        self.model.eval().to(self.device)

        # Lấy cấu hình tiền xử lý từ timm (mean/std, input_size, interpolation…)
        self.cfg = resolve_model_data_config(self.model)
        # Kích thước mục tiêu (DINOv2 ViT-S/14 = 518)
        self.target = int(self.cfg.get("input_size", [3, 518, 518])[1])
        # Pad theo mean (0..1) -> (0..255)
        mean = self.cfg.get("mean", (0.485, 0.456, 0.406))
        self.pad_color = tuple(int(round(m * 255)) for m in mean)

        # Transform chỉ gồm ToTensor + Normalize (KHÔNG resize/crop nữa)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=self.cfg.get("std", (0.229, 0.224, 0.225))),
        ])

    def _letterbox_to_target(self, img_pil: Image.Image) -> Image.Image:
        """
        Giữ tỉ lệ, scale lên/xuống sao cho cạnh dài = target, rồi pad vào khung vuông target x target.
        Pad bằng ImageNet mean để vùng pad ~0 sau Normalize.
        """
        img = img_pil.convert("RGB")
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("Invalid image with zero width/height")

        # scale sao cho max(h, w) -> target
        s = self.target / max(w, h)
        new_w = int(round(w * s))
        new_h = int(round(h * s))
        # Bicubic (đúng pretrained_cfg.interpolation)
        img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)

        # Tạo canvas target x target, pad = mean
        canvas = Image.new("RGB", (self.target, self.target), color=self.pad_color)
        left = (self.target - new_w) // 2
        top  = (self.target - new_h) // 2
        canvas.paste(img_resized, (left, top))
        return canvas

    @torch.inference_mode()
    def embed_pil(self, img_pil: Image.Image) -> list[float]:
        # 1) Letterbox -> 518x518 RGB (không méo, không crop mất biên)
        img_518 = self._letterbox_to_target(img_pil)
        # 2) ToTensor + Normalize
        tensor = self.transform(img_518).unsqueeze(0).to(self.device)  # (1,3,518,518)
        # 3) Forward -> feature
        feat = self.model(tensor)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        vec = feat.squeeze(0).detach().cpu().float().numpy()
        # 4) L2 normalize để dùng cosine/IP trong FAISS
        norm = np.linalg.norm(vec) + 1e-12
        vec = (vec / norm).astype(np.float32)
        return vec.tolist()

# Singleton embedder để tránh load model nhiều lần
_embedder: Optional[ImageEmbedder] = None

def get_embedder() -> ImageEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ImageEmbedder()
    return _embedder

def embed_pil(img_pil: Image.Image) -> list[float]:
    return get_embedder().embed_pil(img_pil)
