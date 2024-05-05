from pathlib import Path

import flax
import jax.numpy as jnp
import soundfile as sf

from model import HuBERTEncoder, make_padding_mask

OVERWRITE_EXPECTED = False

checkpoint_path = Path("checkpoints/hubert_bshall.bin")
wav1_input_path = Path("test-data/84-121123-0000.flac")
wav2_input_path = Path("test-data/1272-128104-0000.flac")
features1_output_path = Path("test-data/84-121123-0000.npy")
features2_output_path = Path("test-data/1272-128104-0000.npy")
features_batched_output_path = Path("test-data/batched.npy")

assert checkpoint_path.exists(), f"File {checkpoint_path} does not exist"
assert wav1_input_path.exists(), f"File {wav1_input_path} does not exist"

with open(checkpoint_path, "rb") as f:
    params = flax.serialization.from_bytes(HuBERTEncoder, f.read())

wav1, sr = sf.read(wav1_input_path)
wav1 = jnp.array(wav1)
wav1 = wav1[None, None, :]

wav2, sr = sf.read(wav2_input_path)
wav2 = jnp.array(wav2)
wav2 = wav2[None, None, :]

expected_features1 = jnp.load(features1_output_path)
expected_features2 = jnp.load(features2_output_path)
extected_features_batched = jnp.load(features_batched_output_path)

model = HuBERTEncoder(num_layers=12)

# Test unbatched - wav1
features1 = model.apply({"params": params}, wav1, padding_mask=None, train=False)
if OVERWRITE_EXPECTED:
    jnp.save(features1_output_path, features1)
else:
    if not jnp.allclose(features1, expected_features1, rtol=1e-5, atol=1e-5):
        mse1 = jnp.mean((features1 - expected_features1) ** 2)
        print(f"Mismatch for {wav1_input_path}: MSE = {mse1:.8f}")
    else:
        print(f"Test passed for {wav1_input_path}")

# Test unbatched - wav2
features2 = model.apply({"params": params}, wav2, padding_mask=None, train=False)
if OVERWRITE_EXPECTED:
    jnp.save(features2_output_path, features2)
else:
    if not jnp.allclose(features2, expected_features2, rtol=1e-5, atol=1e-5):
        mse2 = jnp.mean((features2 - expected_features2) ** 2)
        print(f"Mismatch for {wav2_input_path}: MSE = {mse2:.8f}")
    else:
        print(f"Test passed for {wav2_input_path}")

# Test batched
wav1_num_samples = wav1.shape[-1]
wav2_num_samples = wav2.shape[-1]
lengths_samples = jnp.array([wav1_num_samples, wav2_num_samples])
max_num_samples = jnp.max(lengths_samples)
wav1 = jnp.pad(wav1, ((0, 0), (0, 0), (0, max_num_samples - wav1_num_samples)))
wav2 = jnp.pad(wav2, ((0, 0), (0, 0), (0, max_num_samples - wav2_num_samples)))
wavs = jnp.concatenate([wav1, wav2], axis=0)
padding_mask = make_padding_mask(lengths_samples)

features = model.apply({"params": params}, wavs, padding_mask=padding_mask, train=False)
if OVERWRITE_EXPECTED:
    jnp.save(features_batched_output_path, features)
else:
    if not jnp.allclose(features, extected_features_batched, rtol=1e-5, atol=1e-5):
        mse_batched = jnp.mean((features - extected_features_batched) ** 2)
        print(f"Mismatch for batched wavs: MSE = {mse_batched:.8f}")
    else:
        print(f"Test passed for batched wavs")
