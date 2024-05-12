import math

import click
import jax.numpy as jnp
from torch.utils.data import DataLoader

from dataset import WaveformsAndLabelsDataset
from trainer import HuBERTTrainer


# fmt: off
@click.command()
@click.option("--waveforms_dir", required=True)
@click.option("--waveforms_train_match_pattern", default="train-clean-100/**/*.flac")
@click.option("--waveforms_valid_match_pattern", default="dev-clean/**/*.flac")
@click.option("--labels_dir", required=True)
@click.option("--labels_train_match_pattern", default="train-clean-100/**/*.npy")
@click.option("--labels_vaild_match_pattern", default="dev-clean/**/*.npy")
@click.option("--num_labels", type=int, required=True)
@click.option("--label_rate", type=int, required=True)
@click.option("--num_workers", default=0, type=int)
@click.option("--batch_size", type=int, required=True)
@click.option("--warmup_steps", default=20000, type=int)
@click.option("--num_steps", default=250000, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--dtype", type=click.Choice(["float32", "bfloat16"]))
# fmt: on
def train(
    waveforms_dir,
    waveforms_train_match_pattern,
    waveforms_valid_match_pattern,
    labels_dir,
    labels_train_match_pattern,
    labels_vaild_match_pattern,
    num_labels,
    label_rate,
    num_workers,
    batch_size,
    warmup_steps,
    num_steps,
    seed,
    dtype,
):

    dtype = jnp.float32 if dtype == "float32" else jnp.bfloat16

    train_dataset = WaveformsAndLabelsDataset(
        waveforms_dir=waveforms_dir,
        waveforms_match_pattern=waveforms_train_match_pattern,
        labels_dir=labels_dir,
        labels_match_pattern=labels_train_match_pattern,
        label_rate=label_rate,
        num_unique_labels=num_labels,
        random_crop=True,
        max_sample_size=256000,
        min_sample_size=16000,
    )

    valid_dataset = WaveformsAndLabelsDataset(
        waveforms_dir=waveforms_dir,
        waveforms_match_pattern=waveforms_valid_match_pattern,
        labels_dir=labels_dir,
        labels_match_pattern=labels_vaild_match_pattern,
        label_rate=label_rate,
        num_unique_labels=num_labels,
        random_crop=False,
        max_sample_size=256000,
        min_sample_size=16000,
    )

    example_batch = next(
        iter(
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                collate_fn=train_dataset.collate_fn,
            )
        )
    )

    trainer = HuBERTTrainer(
        model_name="hubert-base",
        example_batch=example_batch,
        num_label_embeddings=num_labels,
        seed=seed,
        checkpoint_dir="./checkpoints",
        lr_peak_value=5e-4,
        lr_end_value=0,
        warmup_steps=warmup_steps,
        decay_steps=num_steps,
        gradient_clip_value=1.0,
        dtype=dtype,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
        drop_last=False,
        num_workers=num_workers,
    )

    num_epochs = math.ceil(num_steps / len(train_loader))
    trainer.train_model(train_loader, valid_loader, num_epochs=num_epochs)


if __name__ == "__main__":
    train()
