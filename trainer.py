from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints
from flax.training.train_state import TrainState
from flax.typing import PRNGKey
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Batch
from model import HuBERTForTraining


class TrainerModule:

    def __init__(
        self,
        model_name: str,
        example_batch: Batch,
        num_label_embeddings: int,  # number of target labels
        seed: int = 42,
        checkpoint_dir: str = "./checkpoints",
        lr_peak_value: float = 5e-4,
        lr_end_value: float = 0,
        warmup_steps=32000,  # 20k iter1, 32k iter2 (8% of total steps)
        decay_steps=230000,  # 230k iter1, 368k iter2 (92% of total steps)
        gradient_clip_value: float = 1.0,  # TODO: Confirm this value
    ):

        self.model_name = model_name
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir).absolute()
        self.lr_peak_value = lr_peak_value
        self.lr_end_value = lr_end_value
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.gradient_clip_value = gradient_clip_value

        self.model = HuBERTForTraining(num_label_embeddings=num_label_embeddings)

        self.log_dir = self.checkpoint_dir / model_name
        self.logger = SummaryWriter(log_dir=self.log_dir)

        self.create_functions()
        self.init_model(example_batch)

    def get_loss_function(self):
        def calculate_loss(params: dict, rng: PRNGKey, batch: Batch, train: bool):
            rng, dropout_apply_rng, mask_apply_rng = jax.random.split(rng, 3)

            logits = self.model.apply(
                {"params": params},
                x=batch.padded_waveforms,
                padding_mask=batch.padding_mask,
                feature_mask=batch.feature_mask,
                train=train,
                rngs={"dropout": dropout_apply_rng, "mask_rng": mask_apply_rng},
            )

            losses = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits,
                labels=batch.padded_labels,
            )
            correct = logits.argmax(axis=-1) == batch.padded_labels

            num_masked_entries = jnp.sum(batch.feature_mask)

            masked_loss = (
                jnp.where(batch.feature_mask, losses, 0).sum() / num_masked_entries
            )
            masked_accuracy = (
                jnp.where(batch.feature_mask, correct, 0).sum() / num_masked_entries
            )

            return masked_loss, (masked_accuracy, rng)

        return calculate_loss

    def create_functions(self):
        calculate_loss = self.get_loss_function()

        @jax.jit
        def train_step(
            state: TrainState, rng: PRNGKey, batch: Batch
        ) -> Tuple[TrainState, PRNGKey, jnp.ndarray, jnp.ndarray]:
            loss_fn = lambda params: calculate_loss(params, rng, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, accuracy, rng = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads)
            return state, rng, loss, accuracy

        @jax.jit
        def eval_step(
            state: TrainState, rng: PRNGKey, batch: Batch
        ) -> Tuple[jnp.ndarray, PRNGKey]:
            _, (accuracy, rng) = calculate_loss(state.params, rng, batch, train=False)
            return accuracy, rng

        self.train_step = train_step
        self.eval_step = eval_step

    def init_model(self, example_batch: Batch):
        self.rng = jax.random.PRNGKey(self.seed)

        self.rng, init_rng, dropout_init_rng, mask_init_rng = jax.random.split(
            self.rng, 4
        )

        params = self.model.init(
            {
                "params": init_rng,
                "dropout": dropout_init_rng,
                "mask_rng": mask_init_rng,
            },
            x=example_batch.padded_waveforms,
            padding_mask=example_batch.padding_mask,
            feature_mask=example_batch.feature_mask,
            train=True,
        )["params"]

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr_peak_value,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.lr_end_value,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.gradient_clip_value),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=0.9,
                b2=0.98,
                eps=1e-06,
                weight_decay=0.01,
            ),
        )

        self.state = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer
        )

    def train_model(self, train_loader, val_loader, num_epochs):
        best_val_loss = float("inf")
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 1 == 0:  # validate every 1 epoch
                val_loss = self.eval_model(val_loader)
                self.logger.add_scalar("val/loss", val_loss, global_step=epoch_idx)
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def train_epoch(self, train_loader, epoch):
        losses, accuracies = [], []
        with tqdm(total=len(train_loader), desc="Training", leave=False) as pbar:
            for batch in train_loader:
                self.state, self.rng, loss, accuracy = self.train_step(
                    self.state, self.rng, batch
                )
                losses.append(loss)
                accuracies.append(accuracy)
                pbar.update(1)
                pbar.set_postfix(
                    loss=np.mean(jax.device_get(losses)),
                    accuracy=np.mean(jax.device_get(accuracies)),
                )

        avg_loss = np.stack(jax.device_get(losses)).mean()
        avg_accuracy = np.stack(jax.device_get(accuracies)).mean()
        self.logger.add_scalar("train/loss", avg_loss, global_step=epoch)
        self.logger.add_scalar("train/accuracy", avg_accuracy, global_step=epoch)

    def eval_model(self, data_loader):
        total_loss, count = 0, 0
        for batch in data_loader:
            batch: Batch = batch  # for type hinting
            loss, self.rng = self.eval_step(self.state, self.rng, batch)
            total_loss += loss * batch.batch_size()
            count += batch.batch_size()
        eval_loss = (total_loss / count).item()
        return eval_loss

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir, target=self.state.params, step=step
        )

    # TODO: load model


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    import math

    from dataset import TrainingDataset

    num_labels = 500
    num_workers = 2
    batch_size = 2
    warmup_steps = 32000
    decay_steps = 230000
    num_steps = warmup_steps + decay_steps  # 250000

    dataset = TrainingDataset(
        waveforms_dir="/media/SSD/datasets/LibriSpeech/dev-clean/1272",
        labels_dir="/home/nicolvisser/repos/hubert-base-jax/data/mfcc-cluster-ids/dev-clean/1272",
        label_rate=100,
        num_unique_labels=num_labels,
        random_crop=True,
        max_sample_size=256000,
        min_sample_size=16000,
    )

    example_batch = next(
        iter(DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn))
    )

    trainer = TrainerModule(
        model_name="hubert-base",
        example_batch=example_batch,
        num_label_embeddings=num_labels,
        seed=42,
        checkpoint_dir="./checkpoints",
        lr_peak_value=5e-4,
        lr_end_value=0,
        warmup_steps=32000,
        decay_steps=230000,
        gradient_clip_value=1.0,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
        num_workers=num_workers,
    )

    num_epochs = math.ceil(num_steps / len(train_loader))

    trainer.train_model(train_loader, val_loader, num_epochs=num_epochs)
