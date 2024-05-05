import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import xavier_uniform as xu
from flax.typing import PRNGKey
from numpy import dtype


class FeatureExtractor(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        # fmt: off
        self.conv0 = nn.Conv(512, 10, 5, "VALID", use_bias=False, kernel_init=xu(), dtype=self.dtype)
        self.norm0 = nn.GroupNorm(512, epsilon=1e-5, use_bias=True)
        self.conv1 = nn.Conv(512, 3, 2, "VALID", use_bias=False, kernel_init=xu(), dtype=self.dtype)
        self.conv2 = nn.Conv(512, 3, 2, "VALID", use_bias=False, kernel_init=xu(), dtype=self.dtype)
        self.conv3 = nn.Conv(512, 3, 2, "VALID", use_bias=False, kernel_init=xu(), dtype=self.dtype)
        self.conv4 = nn.Conv(512, 3, 2, "VALID", use_bias=False, kernel_init=xu(), dtype=self.dtype)
        self.conv5 = nn.Conv(512, 2, 2, "VALID", use_bias=False, kernel_init=xu(), dtype=self.dtype)
        self.conv6 = nn.Conv(512, 2, 2, "VALID", use_bias=False, kernel_init=xu(), dtype=self.dtype)
        # fmt: on

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv0(x)
        x = nn.gelu(self.norm0(x))
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        x = nn.gelu(self.conv3(x))
        x = nn.gelu(self.conv4(x))
        x = nn.gelu(self.conv5(x))
        x = nn.gelu(self.conv6(x))
        return x


class FeatureProjection(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.projection = nn.Dense(768, use_bias=True, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=0.1)

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x, deterministic=not train)
        return x


class PositionalConvEmbedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        conv = nn.Conv(
            features=768,
            kernel_size=128,
            strides=1,
            padding=128 // 2,
            feature_group_count=16,
            use_bias=True,
            kernel_init=xu(),
            dtype=self.dtype,
        )
        self.conv = nn.WeightNorm(conv, feature_axes=0, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv(x)
        x = nn.gelu(x[:, :-1, :])
        return x


def scaled_dot_product(
    q: jnp.ndarray,  # B T Nh Dh
    k: jnp.ndarray,  # B T Nh Dh
    v: jnp.ndarray,  # B T Nh Dh
    mask: jnp.ndarray = None,
    softmax_dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    attn_logits = jnp.einsum("...qhd,...khd->...hqk", q, k)  # B Nh T T
    attn_logits = attn_logits * (q.shape[-1] ** -0.5)  # B Nh T T
    if mask is not None:
        attn_logits = jnp.where(
            mask, attn_logits, jnp.finfo(softmax_dtype).min
        )  # B Nh T T
    attention = nn.softmax(attn_logits, axis=-1)  # B Nh T T
    values = jnp.einsum("...hqk,...khd->...qhd", attention, v)  # B T Nh Dh
    return values, attention


def expand_mask(mask):
    if mask.ndim < 2:
        raise ValueError("Mask should have at least 2 dimensions")
    if mask.ndim == 2:  # B T (e.g. padding mask)
        mask = mask[:, None, :]  # B 1 T
    if mask.ndim == 3:  # B T T (e.g. causal mask)
        mask = mask[:, None, :, :]  # B 1 T T
    return mask


class MultiheadAttention(nn.Module):
    embed_dim: int
    num_heads: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert (
            self.embed_dim % self.num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv_proj = nn.Dense(
            3 * self.embed_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.o_proj = nn.Dense(
            self.embed_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(
        self, x: jnp.ndarray, mask: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        B, T, D = x.shape
        if mask is not None:
            mask = expand_mask(mask)

        qkv = self.qkv_proj(x)  # B T 3D
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)  # B T 3 Nh Dh

        values, attention = scaled_dot_product(
            qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=mask
        )  # (B T Nh Dh), (B Nh T T)
        values = values.reshape(B, T, D)
        o = self.o_proj(values)  # B T D

        return o, attention  # (B T D), (B Nh T T)


class EncoderBlock(nn.Module):
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attn = MultiheadAttention(
            embed_dim=self.input_dim, num_heads=self.num_heads, dtype=self.dtype
        )
        self.linear = [
            nn.Dense(self.dim_feedforward, dtype=self.dtype),
            nn.Dropout(self.dropout_prob),
            # nn.gelu is applied here
            nn.Dense(self.input_dim, dtype=self.dtype),
        ]
        self.norm1 = nn.LayerNorm(dtype=self.dtype)
        self.norm2 = nn.LayerNorm(dtype=self.dtype)
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(
        self, x: jnp.ndarray, mask: jnp.ndarray = None, train: bool = True
    ) -> jnp.ndarray:
        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        linear_out = x
        for l in self.linear:
            linear_out = (
                l(linear_out)
                if not isinstance(l, nn.Dropout)
                else nn.gelu(l(linear_out, deterministic=not train))
            )
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            EncoderBlock(
                self.input_dim,
                self.num_heads,
                self.dim_feedforward,
                self.dropout_prob,
                dtype=self.dtype,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray = None,
        train: bool = True,
    ) -> jnp.ndarray:
        for l in self.layers:
            x = l(x, mask=mask, train=train)
        return x

    def get_attention_maps(
        self, x: jnp.ndarray, mask: jnp.ndarray = None, train: bool = True
    ) -> jnp.ndarray:
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask)
            attention_maps.append(attn_map)
            x = l(x, mask=mask, train=train)
        return attention_maps


class HuBERTEncoder(nn.Module):
    num_layers: int = 12
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.feature_extractor = FeatureExtractor(dtype=self.dtype)
        self.feature_projection = FeatureProjection(dtype=self.dtype)
        self.positional_embedding = PositionalConvEmbedding(dtype=self.dtype)
        self.norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=0.1)
        self.encoder = TransformerEncoder(
            num_layers=self.num_layers,
            input_dim=768,
            num_heads=12,
            dim_feedforward=3072,
            dropout_prob=0.1,
            dtype=self.dtype,
        )
        self.mask_embedding = self.param(
            "mask_embedding",
            nn.initializers.uniform(),
            (768,),
        )

    def __call__(
        self,
        x: jnp.ndarray,
        feature_mask: jnp.ndarray = None,
        padding_mask: jnp.ndarray = None,
        train: bool = True,
    ) -> jnp.ndarray:
        x = jnp.pad(x, ((0, 0), (0, 0), ((400 - 320) // 2, (400 - 320) // 2)))  # B 1 t
        x = jnp.transpose(x, (0, 2, 1))  # B t 1
        x = self.feature_extractor(x)  # B T 512
        x = self.feature_projection(x, train=train)  # B T 768
        if feature_mask is not None:
            x = jnp.where(feature_mask[:, :, None], self.mask_embedding, x)  # B T 768
        x = self.positional_embedding(x)  # B T 768
        x = self.norm(x)  # B T 768
        x = self.dropout(x, deterministic=not train)  # B T 768
        x = self.encoder(x, mask=padding_mask, train=train)  # B T 768
        return x


def cosine_similarity(
    x1: jnp.ndarray, x2: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
    x1_normalized = x1 / jnp.clip(
        jnp.linalg.norm(x1, axis=-1, keepdims=True), a_min=eps
    )
    x2_normalized = x2 / jnp.clip(
        jnp.linalg.norm(x2, axis=-1, keepdims=True), a_min=eps
    )
    return jnp.sum(x1_normalized * x2_normalized, axis=-1)


class HuBERTForTraining(nn.Module):
    num_label_embeddings: int = 504  # number of target labels
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hubert_encoder = HuBERTEncoder(dtype=self.dtype)
        self.proj = nn.Dense(256, dtype=self.dtype)
        self.label_embeddings = self.param(
            "label_embeddings",
            nn.initializers.normal(1),
            (self.num_label_embeddings, 256),
        )

    def logits(self, x: jnp.ndarray) -> jnp.ndarray:
        logits = cosine_similarity(
            x[:, :, None, :], self.label_embeddings[None, None, :, :]
        )
        return logits / 0.1

    def __call__(
        self,
        x: jnp.ndarray,
        padding_mask: jnp.ndarray,
        feature_mask: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        x = self.hubert_encoder(
            x,
            padding_mask=padding_mask,
            feature_mask=feature_mask,
            train=train,
        )  # B T 768
        x = self.proj(x)  # B T 256
        logits = self.logits(x)  # B T N
        return logits


# util to create padding mask from waveform_lengths
def make_padding_mask(T: int, waveform_lengths: jnp.ndarray) -> jnp.ndarray:
    arange_T = jnp.arange(T)[None, :]  # 1 T
    unpadded_lengths = waveform_lengths // 320  # B
    unpadded_lengths = unpadded_lengths[:, None]  # B 1
    mask = arange_T < unpadded_lengths  # B T
    return mask


if __name__ == "__main__":
    model = HuBERTEncoder()

    # sample train data
    x = jnp.zeros((2, 1, 16000))
    padding_mask = jnp.zeros([2, 16000 // 320], dtype=jnp.bool_)
    feature_mask = jnp.zeros([2, 16000 // 320], dtype=jnp.bool_)

    # create rngs
    main_rng = jax.random.PRNGKey(0)
    main_rng, init_rng, dropout_init_rng = jax.random.split(main_rng, 3)

    params = model.init(
        {"params": init_rng, "dropout": dropout_init_rng},
        x,
        train=True,
    )["params"]

    main_rng, dropout_apply_rng = jax.random.split(main_rng)

    y = model.apply(
        {"params": params},
        x,
        padding_mask=padding_mask,
        feature_mask=feature_mask,
        rngs={"dropout": dropout_apply_rng},
        train=True,
    )

    print(y.shape)
