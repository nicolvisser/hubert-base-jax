import os
from pathlib import Path

import click
import flax
import torch


def unflatten_dict(flat_dict: dict, sep: str = "."):
    result = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}  # Create a new nested level
            current = current[part]
        current[parts[-1]] = value  # Assign value at the deepest level
    return result


@click.command()
@click.option(
    "--checkpoint_path",
    required=False,
    default="checkpoints/hubert_bshall.bin",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the Torch checkpoint with a .bin extension",
)
def scrape_bshall_torch_weights(checkpoint_path: Path):
    # assert checkpoint_path has extension .bin
    assert checkpoint_path.suffix == ".bin", "Checkpoint must have a .bin extension"

    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/bshall/hubert/releases/download/v0.2/hubert-discrete-96b248c5.pt",
        progress=True,
    )
    torch_params = checkpoint["hubert"]

    # covert all tensors to numpy
    for k, v in torch_params.items():
        torch_params[k] = v.numpy()

    # fmt: off
    jax_params = {}
    jax_params['feature_extractor.conv0.kernel'] = torch_params["feature_extractor.conv0.weight"].T
    jax_params['feature_extractor.norm0.scale'] = torch_params["feature_extractor.norm0.weight"]
    jax_params['feature_extractor.norm0.bias'] = torch_params["feature_extractor.norm0.bias"]
    jax_params['feature_extractor.conv1.kernel'] = torch_params["feature_extractor.conv1.weight"].T
    jax_params['feature_extractor.conv2.kernel'] = torch_params["feature_extractor.conv2.weight"].T
    jax_params['feature_extractor.conv3.kernel'] = torch_params["feature_extractor.conv3.weight"].T
    jax_params['feature_extractor.conv4.kernel'] = torch_params["feature_extractor.conv4.weight"].T
    jax_params['feature_extractor.conv5.kernel'] = torch_params["feature_extractor.conv5.weight"].T
    jax_params['feature_extractor.conv6.kernel'] = torch_params["feature_extractor.conv6.weight"].T
    jax_params['feature_projection.norm.scale'] = torch_params['feature_projection.norm.weight']
    jax_params['feature_projection.norm.bias'] = torch_params["feature_projection.norm.bias"]
    jax_params['feature_projection.projection.kernel'] = torch_params['feature_projection.projection.weight'].T
    jax_params['feature_projection.projection.bias'] = torch_params["feature_projection.projection.bias"]
    jax_params['positional_embedding.conv.layer_instance.kernel'] = torch_params["positional_embedding.conv.weight_v"].T
    jax_params['positional_embedding.conv.layer_instance.bias'] = torch_params["positional_embedding.conv.bias"]
    jax_params['positional_embedding.conv.layer_instance/kernel/scale'] = torch_params["positional_embedding.conv.weight_g"][0, 0, :]
    jax_params['feature_masking.mask_embedding'] = torch_params['masked_spec_embed']
    jax_params['norm.scale'] = torch_params["norm.weight"]
    jax_params['norm.bias'] = torch_params["norm.bias"]
    jax_params['proj.kernel'] = torch_params["proj.weight"].T
    jax_params['proj.bias'] = torch_params["proj.bias"]
    jax_params['label_embeddings'] = torch_params['label_embedding.weight']
    for i in range(12):
        jax_params[f'encoder.layers_{i}.self_attn.qkv_proj.kernel'] = torch_params[f"encoder.layers.{i}.self_attn.in_proj_weight"].T
        jax_params[f'encoder.layers_{i}.self_attn.qkv_proj.bias'] = torch_params[f"encoder.layers.{i}.self_attn.in_proj_bias"]
        jax_params[f'encoder.layers_{i}.self_attn.o_proj.kernel'] = torch_params[f"encoder.layers.{i}.self_attn.out_proj.weight"].T
        jax_params[f'encoder.layers_{i}.self_attn.o_proj.bias'] = torch_params[f"encoder.layers.{i}.self_attn.out_proj.bias"]
        jax_params[f'encoder.layers_{i}.norm1.scale'] = torch_params[f"encoder.layers.{i}.norm1.weight"]
        jax_params[f'encoder.layers_{i}.norm1.bias'] = torch_params[f"encoder.layers.{i}.norm1.bias"]
        jax_params[f'encoder.layers_{i}.norm2.scale'] = torch_params[f"encoder.layers.{i}.norm2.weight"]
        jax_params[f'encoder.layers_{i}.norm2.bias'] = torch_params[f"encoder.layers.{i}.norm2.bias"]
        jax_params[f'encoder.layers_{i}.linear_0.kernel'] = torch_params[f"encoder.layers.{i}.linear1.weight"].T
        jax_params[f'encoder.layers_{i}.linear_0.bias'] = torch_params[f"encoder.layers.{i}.linear1.bias"]
        jax_params[f'encoder.layers_{i}.linear_2.kernel'] = torch_params[f"encoder.layers.{i}.linear2.weight"].T
        jax_params[f'encoder.layers_{i}.linear_2.bias'] = torch_params[f"encoder.layers.{i}.linear2.bias"]
    # fmt: on

    num_torch_params = sum(p.size for p in torch_params.values())
    num_jax_params = sum(p.size for p in jax_params.values())

    print(f"Number of Torch params: {num_torch_params}")
    print(f"Number of JAX params: {num_jax_params}")

    if num_torch_params == num_jax_params:
        print("Number of params match!")
    else:
        print("Number of params do not match!")

    jax_params = unflatten_dict(jax_params)

    with open(checkpoint_path, "wb") as f:
        f.write(flax.serialization.to_bytes(jax_params))
        print(f"JAX params saved to {checkpoint_path}")


if __name__ == "__main__":
    scrape_bshall_torch_weights()
