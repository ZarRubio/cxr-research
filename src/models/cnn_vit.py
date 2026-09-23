"""
src/models/cnn_vit.py

Modelo híbrido CNN-ViT para clasificación de radiografías de tórax.
Arquitectura: Secuencial CNN → ViT (Opción A).

Diseño:
  1. CNN Backbone (DenseNet121 preentrenado en NIH via TorchXRayVision):
     extrae features locales de la imagen CXR.
     Output: feature map (B, 1024, 7, 7).

  2. Patch Embedding:
     Convierte el feature map en una secuencia de patches para el Transformer.
     7×7 = 49 patches, proyectados a embedding_dim=512.

  3. Vision Transformer (implementación propia):
     [CLS] token + positional encoding + N bloques Transformer.
     Cada bloque: LayerNorm → Multi-Head Self-Attention → MLP.
     El [CLS] token agrega información global de todos los patches.

  4. Cabeza clasificadora:
     LayerNorm → Linear(embedding_dim, num_classes).

Justificación del diseño:
  El CNN pre-procesa la imagen en features de dominio (CXR), reduciendo
  la tarea del ViT de "aprender features desde píxeles" a "relacionar
  features ya significativas". Esto reduce drásticamente el volumen de
  datos necesario para entrenar el ViT desde ~14M imágenes (ViT original)
  a ~3,900 imágenes de CXR.

Referencias:
  - Dosovitskiy et al. (2021). An Image is Worth 16x16 Words. ICLR.
  - Chen et al. (2021). Crossvit: Cross-attention multi-scale ViT. ICCV.
  - TorchXRayVision: Cohen et al. (2022).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchxrayvision as xrv


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention con proyecciones Q, K, V.
    Implementación propia para máxima trazabilidad y comprensión.
    """
    def __init__(self, embedding_dim: int, num_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        assert embedding_dim % num_heads == 0, \
            f"embedding_dim={embedding_dim} no divisible entre num_heads={num_heads}"

        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.qkv  = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) — N tokens de dimensión D.
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape

        # Proyectar a Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, N, d_head)
        q, k, v = qkv.unbind(0)             # cada: (B, H, N, d_head)

        # Atención escalada
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Agregación ponderada
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """
    Bloque Transformer estándar:
      LayerNorm → MHSA → residual
      LayerNorm → MLP  → residual
    """
    def __init__(self, embedding_dim: int, num_heads: int,
                 mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn  = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # residual MHSA
        x = x + self.mlp(self.norm2(x))    # residual MLP
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer aplicado a patches de features (no píxeles).

    Args:
        n_patches:     número de patches de entrada (49 para 7×7 feature map).
        patch_dim:     dimensión de cada patch (1024 del DenseNet121).
        embedding_dim: dimensión del espacio de atención.
        num_heads:     cabezas de atención.
        num_layers:    bloques Transformer.
        mlp_dim:       dimensión interna del MLP.
        num_classes:   número de clases de salida.
        dropout:       regularización.
    """
    def __init__(self, n_patches: int, patch_dim: int,
                 embedding_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 4, mlp_dim: int = 1024,
                 num_classes: int = 4, dropout: float = 0.1):
        super().__init__()

        self.n_patches     = n_patches
        self.embedding_dim = embedding_dim

        # Proyección de patches → espacio de atención
        self.patch_embedding = nn.Linear(patch_dim, embedding_dim)

        # [CLS] token learnable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding learnable (n_patches + 1 por el CLS)
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, n_patches + 1, embedding_dim)
        )
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Bloques Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

        # Cabeza clasificadora
        self.head = nn.Linear(embedding_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialización estándar para Transformers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, n_patches, patch_dim) — patches del feature map CNN.
        Returns:
            logits: (B, num_classes).
        """
        B = patches.shape[0]

        # Proyectar patches al espacio de atención
        x = self.patch_embedding(patches)  # (B, 49, 512)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 512)
        x   = torch.cat([cls, x], dim=1)         # (B, 50, 512)

        # Agregar positional encoding
        x = self.dropout(x + self.pos_encoding)

        # Pasar por todos los bloques Transformer
        for block in self.blocks:
            x = block(x)

        # Tomar el [CLS] token y normalizar
        cls_out = self.norm(x[:, 0])  # (B, 512)

        return self.head(cls_out)     # (B, num_classes)


# ---------------------------------------------------------------------------
# Modelo híbrido CNN-ViT
# ---------------------------------------------------------------------------
class CNNViT(nn.Module):
    """
    Modelo híbrido CNN-ViT para clasificación de radiografías de tórax.

    Flujo:
      imagen (B, 1, 224, 224)
        → DenseNet121 features (B, 1024, 7, 7)
        → flatten spatial: (B, 49, 1024)
        → VisionTransformer: (B, 49, 1024) → (B, num_classes)

    Soporta entrenamiento en 2 fases:
      Fase 1: CNN congelada, solo ViT + head (set_phase(1))
      Fase 2: Todo desbloqueado con LR diferenciado (set_phase(2))
    """

    def __init__(self, backbone_weights: str = "densenet121-res224-nih",
                 embedding_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 4, mlp_dim: int = 1024,
                 num_classes: int = 4, dropout: float = 0.1):
        super().__init__()

        # 1. CNN Backbone preentrenado
        densenet = xrv.models.DenseNet(weights=backbone_weights)
        self.cnn_features = densenet.features
        self.cnn_out_dim  = 1024  # canales del feature map del DenseNet121

        # 2. Vision Transformer
        # 7×7 feature map → 49 patches
        self.vit = VisionTransformer(
            n_patches=49,
            patch_dim=self.cnn_out_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self._current_phase = None

    def set_phase(self, phase: int) -> None:
        """
        Configura qué parámetros se entrenan.

        Fase 1: CNN congelada, solo ViT + head.
        Fase 2: Todo desbloqueado.
        """
        assert phase in {1, 2}, "phase debe ser 1 o 2"
        self._current_phase = phase

        if phase == 1:
            # Congelar CNN
            for param in self.cnn_features.parameters():
                param.requires_grad = False
            # Descongelar ViT
            for param in self.vit.parameters():
                param.requires_grad = True

        else:  # phase 2
            # Descongelar todo
            for param in self.parameters():
                param.requires_grad = True

    def get_param_groups(self, lr_cnn: float, lr_vit: float) -> list[dict]:
        """
        Devuelve grupos de parámetros con LR diferenciado para la Fase 2.

        CNN recibe lr_cnn (muy bajo para no destruir preentrenamiento).
        ViT recibe lr_vit (más alto para seguir aprendiendo).
        """
        return [
            {"params": self.cnn_features.parameters(), "lr": lr_cnn,
             "name": "cnn_backbone"},
            {"params": self.vit.parameters(), "lr": lr_vit,
             "name": "vit"},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 224, 224) en rango [-1024, 1024] (formato xrv).
        Returns:
            logits: (B, num_classes).
        """
        # Extraer feature map del DenseNet
        features = self.cnn_features(x)           # (B, 1024, 7, 7)

        # ReLU (igual que el forward original del DenseNet)
        import torch.nn.functional as F
        features = F.relu(features, inplace=False)  # (B, 1024, 7, 7)

        # Convertir feature map en secuencia de patches
        B, C, H, W = features.shape
        patches = features.flatten(2).transpose(1, 2)  # (B, 49, 1024)

        # Pasar por el ViT
        return self.vit(patches)                   # (B, num_classes)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_summary(self) -> dict:
        cnn_total = sum(p.numel() for p in self.cnn_features.parameters())
        vit_total = sum(p.numel() for p in self.vit.parameters())
        return {
            "cnn_total":   cnn_total,
            "vit_total":   vit_total,
            "total":       cnn_total + vit_total,
            "trainable":   self.trainable_params(),
            "phase":       self._current_phase,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_cnn_vit(cfg) -> CNNViT:
    """Construye el modelo CNN-ViT desde la config del Sprint 4."""
    vit_cfg = cfg.model.get("vit", {})
    model = CNNViT(
        backbone_weights=cfg.model["backbone"],
        embedding_dim=vit_cfg.get("embedding_dim", 512),
        num_heads=vit_cfg.get("num_heads", 8),
        num_layers=vit_cfg.get("num_layers", 4),
        mlp_dim=vit_cfg.get("mlp_dim", 1024),
        num_classes=cfg.model["num_classes"],
        dropout=vit_cfg.get("dropout", 0.1),
    )
    return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    code_dir = Path(__file__).resolve().parent.parent.parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNViT(num_classes=4).to(device)
    model.set_phase(1)
    summary = model.param_summary()
    print(f"Modelo CNN-ViT (fase 1):")
    print(f"  CNN params:      {summary['cnn_total']:>12,}")
    print(f"  ViT params:      {summary['vit_total']:>12,}")
    print(f"  Total:           {summary['total']:>12,}")
    print(f"  Entrenables:     {summary['trainable']:>12,} (solo ViT)")

    model.set_phase(2)
    print(f"  Entrenables (fase 2): {model.trainable_params():>8,} (todo)")

    # Forward dummy
    model.set_phase(1)
    model.eval()
    x = torch.randn(4, 1, 224, 224, device=device) * 500
    with torch.no_grad():
        logits = model(x)
    print(f"\nForward OK: input={x.shape} → output={logits.shape}")
    probs = torch.softmax(logits, dim=1)
    print(f"Probs[0]: {probs[0].tolist()}")
    print(f"Suma: {probs[0].sum().item():.3f}")
