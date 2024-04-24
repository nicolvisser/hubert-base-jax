import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import xavier_uniform as xu
from flax.typing import PRNGKey


class FeatureExtractor(nn.Module):
    def setup(self) -> None:
        self.conv0 = nn.Conv(
            512, (10,), (5,), "VALID", use_bias=False, kernel_init=xu()
        )
        self.norm0 = nn.GroupNorm(512, epsilon=1e-5, use_bias=True)
        self.conv1 = nn.Conv(512, (3,), (2,), "VALID", use_bias=False, kernel_init=xu())
        self.conv2 = nn.Conv(512, (3,), (2,), "VALID", use_bias=False, kernel_init=xu())
        self.conv3 = nn.Conv(512, (3,), (2,), "VALID", use_bias=False, kernel_init=xu())
        self.conv4 = nn.Conv(512, (3,), (2,), "VALID", use_bias=False, kernel_init=xu())
        self.conv5 = nn.Conv(512, (2,), (2,), "VALID", use_bias=False, kernel_init=xu())
        self.conv6 = nn.Conv(512, (2,), (2,), "VALID", use_bias=False, kernel_init=xu())

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
    def setup(self):
        self.norm = nn.LayerNorm(epsilon=1e-5)
        self.projection = nn.Dense(768, use_bias=True)
        self.dropout = nn.Dropout(rate=0.1)

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x, deterministic=not train)
        return x


class PositionalConvEmbedding(nn.Module):
    def setup(self):
        conv = nn.Conv(
            768,
            kernel_size=(128,),
            strides=(1,),
            padding=(128 // 2,),
            feature_group_count=16,
            use_bias=True,
            kernel_init=xu(),
        )
        self.conv = nn.WeightNorm(conv, feature_axes=0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv(x)
        x = nn.gelu(x[:, :-1, :])
        return x


def scaled_dot_product(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    B, Nh, T, Dh = q.shape
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))  # B Nh T T
    attn_logits = attn_logits / math.sqrt(Dh)  # B Bh T T
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)  # B Nh T T
    attention = nn.softmax(attn_logits, axis=-1)  # B Nh T T
    values = jnp.matmul(attention, v)  # B Nh T Dh
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
    embed_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads (h)

    def setup(self):
        assert (
            self.embed_dim % self.num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv_proj = nn.Dense(
            3 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.o_proj = nn.Dense(
            self.embed_dim,
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
        qkv = qkv.reshape(B, T, 3, D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # 3 B Nh T Dh

        values, attention = scaled_dot_product(
            qkv[0], qkv[1], qkv[2], mask=mask
        )  # (B Nh T Dh), (B Nh T T)
        values = values.transpose(0, 2, 1, 3)  # B T Nh Dh
        values = values.reshape(B, T, D)
        o = self.o_proj(values)  # B T D

        return o, attention  # (B T D), (B Nh T T)


class EncoderBlock(nn.Module):
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        self.self_attn = MultiheadAttention(
            embed_dim=self.input_dim, num_heads=self.num_heads
        )
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            # nn.gelu is applied here
            nn.Dense(self.input_dim),
        ]
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
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

    def setup(self):
        self.layers = [
            EncoderBlock(
                self.input_dim, self.num_heads, self.dim_feedforward, self.dropout_prob
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


def _make_feature_mask(
    key: PRNGKey,
    sequence_length: int,
    mask_length: int,
    num_masked_spans: int,
) -> jnp.ndarray:
    # get random start indices for masked spans
    start_indices = jax.random.choice(
        key, a=jnp.arange(sequence_length), shape=(num_masked_spans,), replace=False
    )
    # add the indices of the spans
    mask_indices = (jnp.arange(mask_length)[None, :] + start_indices[:, None]).ravel()
    # make the mask (True for masked indices, False for unmasked indices)
    mask = jnp.zeros(sequence_length, dtype=jnp.bool_)
    mask = mask.at[mask_indices].set(True)
    return mask


def _make_feature_mask_batch(
    key: PRNGKey,
    batch_size: int,
    sequence_length: int,
    mask_prob: float,
    mask_length: int,
) -> jnp.ndarray:
    # compute number of masked spans
    num_spans_key, key = jax.random.split(key, 2)
    num_masked_spans = int(
        mask_prob * sequence_length / mask_length
        + jax.random.uniform(num_spans_key, shape=())
    )
    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length
    # make the mask for each row
    row_keys = jax.random.split(key, batch_size)
    mask = jax.vmap(_make_feature_mask, in_axes=(0, None, None, None))(
        row_keys, sequence_length, mask_length, num_masked_spans
    )
    return mask


class FeatureMasking(nn.Module):
    mask_prob: float = 0.8
    mask_length: int = 10
    rng_collection: str = "mask_rng"

    def setup(self):
        self.mask_embedding = self.param(
            "mask_embedding",
            nn.initializers.uniform(),
            (768,),
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: Optional[bool] = True,
        rng: Optional[PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if train:
            B, T, D = x.shape
            if rng is None:
                rng = self.make_rng(self.rng_collection)
            mask = _make_feature_mask_batch(
                rng, B, T, self.mask_prob, self.mask_length
            )  # B T
            x = x.at[mask].set(self.mask_embedding)
        return x


class HuBERTEncoder(nn.Module):
    mask: bool = False
    num_layers: int = 12  # only change this during inference

    def setup(self):
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.feature_masking = FeatureMasking(0.8, 10)
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = nn.LayerNorm(epsilon=1e-5)
        self.dropout = nn.Dropout(rate=0.1)
        self.encoder = TransformerEncoder(
            num_layers=self.num_layers,
            input_dim=768,
            num_heads=12,
            dim_feedforward=3072,
            dropout_prob=0.1,
        )

    def make_padding_mask(
        self, T: int, unpadded_num_samples: jnp.ndarray
    ) -> jnp.ndarray:
        arange_T = jnp.arange(T)[None, :]  # 1 T
        unpadded_lengths = unpadded_num_samples // 320
        unpadded_lengths = unpadded_lengths[:, None]  # B 1
        mask = arange_T < unpadded_lengths  # B T
        return mask

    def make_span_mask() -> jnp.ndarray:
        pass

    def __call__(
        self,
        x: jnp.ndarray,
        unpadded_num_samples: jnp.array = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """
        Encodes the input waveform using the HuBERT model.

        Args:
            x (jnp.ndarray): The input sequence to be encoded. Shape: (B, 1, max_num_samples)
            unpadded_num_samples (jnp.array, optional): The number of unpadded samples in each batch. Shape: (B,)
            layer (int, optional): The layer index to extract the output from. If None, returns the final encoded representation. Default: None
            train (bool, optional): Whether to apply dropout and use the training mode of the model. Default: True

        Returns:
            jnp.ndarray: The encoded representation of the input sequence. Shape: (B, T, 768) where T is approximately max_num_samples // 320

        """
        x = jnp.pad(x, ((0, 0), (0, 0), ((400 - 320) // 2, (400 - 320) // 2)))  # B 1 t
        x = jnp.transpose(x, (0, 2, 1))  # B t 1
        x = self.feature_extractor(x)  # B T 512
        x = self.feature_projection(x, train=train)  # B T 768
        x = self.feature_masking(x, train=train)  # B T 768
        x = self.positional_embedding(x)  # B T 768
        x = self.norm(x)  # B T 768
        x = self.dropout(x, deterministic=not train)  # B T 768
        padding_mask = (
            None
            if unpadded_num_samples is None
            else self.make_padding_mask(x.shape[1], unpadded_num_samples)
        )  # B T
        x = self.encoder(x, mask=padding_mask, train=train)  # B T 768
        return x


# def cosine_similarity(
#     x1: jnp.ndarray, x2: jnp.ndarray, eps: float = 1e-8
# ) -> jnp.ndarray:
#     x1_normalized = x1 / jnp.clip(
#         jnp.linalg.norm(x1, axis=-1, keepdims=True), a_min=eps
#     )
#     x2_normalized = x2 / jnp.clip(
#         jnp.linalg.norm(x2, axis=-1, keepdims=True), a_min=eps
#     )
#     return jnp.sum(x1_normalized * x2_normalized, axis=-1)


# class HuBERTPredictor(nn.Module):
#     num_label_embeddings: int = 504

#     def setup(self):
#         self.hubert_encoder = HuBERTEncoder(mask=True)
#         self.proj = nn.Dense(256)
#         self.label_embeddings = self.param(
#             "label_embeddings",
#             nn.initializers.normal(1),
#             (self.num_label_embeddings, 256),
#         )

#     def logits(self, x: jnp.ndarray) -> jnp.ndarray:
#         logits = cosine_similarity(
#             x[:, :, None, :], self.label_embeddings[None, None, :, :]
#         )
#         return logits / 0.1

#     def __call__(
#         self, x: jnp.ndarray, padding_mask: jnp.ndarray = None, train: bool = True
#     ) -> jnp.ndarray:
#         x = self.hubert_encoder(x, padding_mask=padding_mask, train=train)  # B T 768
#         x = self.proj(x)  # B T 256
#         logits = self.logits(x)  # B T N
#         return logits, None


if __name__ == "__main__":
    model = HuBERTEncoder()

    # sample data
    x = jnp.zeros((2, 1, 16000))
    unpadded_lengths = jnp.array([16000, 10000])

    # create rngs
    main_rng = jax.random.PRNGKey(0)
    main_rng, init_rng, dropout_init_rng, mask_init_rng = jax.random.split(main_rng, 4)

    params = model.init(
        {"params": init_rng, "dropout": dropout_init_rng, "mask_rng": mask_init_rng},
        x,
        train=True,
    )["params"]

    main_rng, dropout_apply_rng, mask_apply_rng = jax.random.split(main_rng, 3)

    # if you don't bind the model, you can apply it like this
    # y = model.apply(
    #     {"params": params},
    #     x,
    #     unpadded_num_samples=unpadded_lengths,
    #     rngs={"dropout": dropout_apply_rng, "mask_rng": mask_apply_rng},
    # )

    binded_mod = model.bind(
        {"params": params},
        rngs={"dropout": dropout_apply_rng, "mask_rng": mask_apply_rng},
    )

    y = binded_mod(x, unpadded_num_samples=unpadded_lengths, train=True)

    print(y.shape)
