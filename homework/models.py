"""
Implement the following models for classification.

This homework extends HW2 by adding advanced neural network architectures.
The ClassificationLoss implementation is provided from HW2.
Implement the two new models below.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.functional.cross_entropy(logits, target)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        A single Transformer encoder block with multi-head attention and MLP.

        You need to implement this for your Vision Transformer.

        Hint: A transformer block typically consists of:
        1. Layer normalization
        2. Multi-head self-attention (use torch.nn.MultiheadAttention with batch_first=True)
        3. Residual connection
        4. Layer normalization
        5. MLP (Linear -> GELU -> Dropout -> Linear -> Dropout)
        6. Residual connection

        Args:
            embed_dim: embedding dimension
            num_heads: number of attention heads
            mlp_ratio: ratio of MLP hidden dimension to embedding dimension
            dropout: dropout probability
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, embed_dim) input sequence

        Returns:
            (batch_size, sequence_length, embed_dim) output sequence
        """
        norm1 = self.layer_norm1(x)
        attention, _ = self.attention(norm1, norm1, norm1)
        x = x + attention
        norm2 = self.layer_norm2(x)
        x = x + self.mlp(norm2)
        return x




class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 256):
        """
        Convert image to sequence of patch embeddings using a simple approach

        This is provided as a helper for implementing the Vision Transformer.
        You can use this directly in your ViTClassifier implementation.

        Args:
            img_size: size of input image (assumed square)
            patch_size: size of each patch
            in_channels: number of input channels (3 for RGB)
            embed_dim: embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape into patches: (B, C, H//p, p, W//p, p) -> (B, C, H//p, W//p, p, p)
        x = x.reshape(B, C, H//p, p, W//p, p).permute(0, 1, 2, 4, 3, 5)
        # Flatten patches: (B, C, H//p, W//p, p*p) -> (B, H//p * W//p, C * p * p)
        x = x.reshape(B, self.num_patches, C * p * p)

        # Linear projection
        return self.projection(x)


class MLPClassifierDeepResidual(nn.Module):
    # Code from Lecture to define a block
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.LayerNorm(out_channels),
                torch.nn.ReLU(),
            )
            if in_channels != out_channels:
                self.skip = torch.nn.Linear(in_channels, out_channels)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x):
            return self.skip(x) + self.model(x)
        
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        HIDDEN_DIM = 64
        HIDDEN_LAYERS = 4
        """
        An MLP with multiple hidden layers and residual connections

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(h*w*3, HIDDEN_DIM, bias=False))
        for i in range(HIDDEN_LAYERS):
            layers.append(self.Block(HIDDEN_DIM, HIDDEN_DIM))
        layers.append(torch.nn.Linear(HIDDEN_DIM, num_classes, bias=False))

        self.model = torch.nn.Sequential(*layers)


        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)


class ViTClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        patch_size: int = 8,
        embed_dim: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """
        Vision Transformer (ViT) classifier

        Args:
            h: int, height of input image
            w: int, width of input image
            num_classes: int, number of classes

        Hint - you can add more arguments to the constructor such as:
            patch_size: int, size of image patches
            embed_dim: int, embedding dimension
            num_layers: int, number of transformer layers
            num_heads: int, number of attention heads

        Note: You can use the provided PatchEmbedding class. You'll need to implement the TransformerBlock class.
        """
        super().__init__()

        self.patch_embedding = PatchEmbedding(img_size=h, patch_size=patch_size, in_channels=3, embed_dim=embed_dim,)
        num_patches = self.patch_embedding.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.linear = nn.Linear(embed_dim, num_classes)

        # ChatGPT Used to help with initialization values
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) input image

        Returns:
            tensor (b, num_classes) logits
        """

        B = x.shape[0]

        x = self.patch_embedding(x)

        # Add CLS token and positional embeddings
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.positional_embedding

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Get output from CLS token
        cls_out = x[:, 0]

        logits = self.linear(cls_out)
        return logits


model_factory = {
    "mlp_deep_residual": MLPClassifierDeepResidual,
    "vit": ViTClassifier,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r