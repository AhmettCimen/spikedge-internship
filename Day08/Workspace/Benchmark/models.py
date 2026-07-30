"""
models.py — Ego-Track Segmentasyon Model Tanımları
====================================================
DeepLabV3+, U-Net ve SegFormer modelleri.
segmentation_models_pytorch (SMP) kütüphanesi kullanılır.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class DiceBCELoss(nn.Module):
    """
    Dice Loss + Binary Cross Entropy Loss birleşimi.
    Ego-track segmentasyonu için optimize edilmiş kayıp fonksiyonu.
    
    Dice Loss: Sınıf dengesizliğine dayanıklı
    BCE Loss: Piksel bazlı öğrenme kararlılığı
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5,
                 smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (
            pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        )
        return 1 - dice.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


class BoundaryAwareLoss(nn.Module):
    """
    Sınır-farkındalıklı kayıp fonksiyonu.
    Ray sınırlarına yakın piksellere daha fazla ağırlık verir.
    """
    
    def __init__(self, boundary_weight: float = 2.0, dilation: int = 5):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.dilation = dilation
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def get_boundary_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Hedef maskeden sınır bölgesi çıkarır."""
        # Erosion ile içeriği küçült
        kernel_size = 2 * self.dilation + 1
        padding = self.dilation
        
        # MaxPool ile dilation, -MaxPool(-x) ile erosion
        dilated = torch.nn.functional.max_pool2d(
            target, kernel_size=kernel_size, stride=1, padding=padding
        )
        eroded = -torch.nn.functional.max_pool2d(
            -target, kernel_size=kernel_size, stride=1, padding=padding
        )
        
        boundary = dilated - eroded
        return boundary
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred, target)
        
        boundary = self.get_boundary_mask(target)
        weight = 1.0 + (self.boundary_weight - 1.0) * boundary
        
        weighted_bce = (bce * weight).mean()
        return weighted_bce


class CombinedLoss(nn.Module):
    """
    Tüm kayıp bileşenlerini birleştiren ana kayıp fonksiyonu.
    
    L = λ1 * L_dice_bce + λ2 * L_boundary
    """
    
    def __init__(self, 
                 dice_bce_weight: float = 0.7,
                 boundary_weight: float = 0.3,
                 use_boundary: bool = True):
        super().__init__()
        self.dice_bce_weight = dice_bce_weight
        self.boundary_weight = boundary_weight
        self.use_boundary = use_boundary
        
        self.dice_bce = DiceBCELoss()
        self.boundary_loss = BoundaryAwareLoss() if use_boundary else None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.dice_bce_weight * self.dice_bce(pred, target)
        
        if self.use_boundary and self.boundary_loss is not None:
            loss += self.boundary_weight * self.boundary_loss(pred, target)
        
        return loss


def create_model(architecture: str = "deeplabv3plus",
                 encoder: str = "resnet50",
                 pretrained: str = "imagenet",
                 in_channels: int = 3,
                 classes: int = 1) -> nn.Module:
    """
    Segmentasyon modeli oluşturur.
    
    Args:
        architecture: Model mimarisi
            - "deeplabv3plus": DeepLabV3+ (önerilen)
            - "unet": U-Net
            - "unetplusplus": U-Net++
            - "fpn": Feature Pyramid Network
            - "pspnet": Pyramid Scene Parsing Network
            - "manet": Multi-Scale Attention Net
            - "linknet": LinkNet
        encoder: Backbone encoder
            - "resnet50": ResNet-50 (dengeli)
            - "resnet34": ResNet-34 (hafif)
            - "efficientnet-b3": EfficientNet-B3 (verimli)
            - "efficientnet-b0": EfficientNet-B0 (çok hafif)
            - "mobilenet_v2": MobileNetV2 (edge cihaz)
            - "timm-regnety_016": RegNetY-016 (hızlı)
        pretrained: Ön eğitim ağırlıkları ("imagenet" veya None)
        in_channels: Giriş kanal sayısı
        classes: Çıkış sınıf sayısı (binary segmentasyon için 1)
    
    Returns:
        PyTorch segmentasyon modeli
    """
    import segmentation_models_pytorch as smp
    
    architecture_map = {
        "deeplabv3plus": smp.DeepLabV3Plus,
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "manet": smp.MAnet,
        "linknet": smp.Linknet,
    }
    
    if architecture.lower() not in architecture_map:
        raise ValueError(
            f"Bilinmeyen mimari: {architecture}. "
            f"Desteklenen: {list(architecture_map.keys())}"
        )
    
    model_class = architecture_map[architecture.lower()]
    
    model = model_class(
        encoder_name=encoder,
        encoder_weights=pretrained,
        in_channels=in_channels,
        classes=classes,
    )
    
    return model


def get_model_configs() -> Dict[str, Dict]:
    """
    Deneylerde kullanılacak model konfigürasyonlarını döner.
    
    Returns:
        Model adı -> config dict eşlemesi
    """
    configs = {
        "DeepLabV3+_ResNet50": {
            "architecture": "deeplabv3plus",
            "encoder": "resnet50",
            "description": "Ana model — Güçlü multi-scale özellik çıkarma",
        },
        "UNet_EfficientNetB3": {
            "architecture": "unet",
            "encoder": "efficientnet-b3",
            "description": "Karşılaştırma — Verimli encoder ile U-Net",
        },
        "DeepLabV3+_ResNet34": {
            "architecture": "deeplabv3plus",
            "encoder": "resnet34",
            "description": "Hafif versiyon — Daha hızlı eğitim ve inference",
        },
        "UNet_MobileNetV2": {
            "architecture": "unet",
            "encoder": "mobilenet_v2",
            "description": "Edge cihaz — AGX Orin hedefli hafif model",
        },
        "FPN_ResNet50": {
            "architecture": "fpn",
            "encoder": "resnet50",
            "description": "Feature Pyramid — Çoklu ölçek ray tespiti",
        },
    }
    
    return configs


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Model parametre sayısını hesaplar."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_mb": total * 4 / (1024 ** 2),  # float32 varsayımı
    }


def export_to_onnx(model: nn.Module, save_path: str,
                    input_size: tuple = (1, 3, 640, 640),
                    opset_version: int = 12):
    """
    Modeli ONNX formatına dışa aktarır.
    
    Args:
        model: PyTorch modeli
        save_path: ONNX dosya kayıt yolu
        input_size: Örnek giriş boyutu (B, C, H, W)
        opset_version: ONNX opset versiyonu
    """
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    if next(model.parameters()).is_cuda:
        dummy_input = dummy_input.cuda()
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"[ONNX] Model dışa aktarıldı: {save_path}")
