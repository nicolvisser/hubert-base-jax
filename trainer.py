from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import checkpoints
from flax.training.train_state import TrainState
from flax.typing import PRNGKey
from tqdm import tqdm

from dataset import Batch
from model import HuBERTForTraining


class HuBERTTrainer:

    def __init__(
        self,
        model_name: str,
        example_batch: Batch,
        num_label_embeddings: int,  # number of target labels
        seed: int = 42,
        checkpoint_dir: str = "./checkpoints",
        lr_peak_value: float = 5e-4,  # 5e-4
        lr_end_value: float = 0,
        warmup_steps: int = 20000,  # 20k iter1, 32k iter2 (8% of total steps)
        decay_steps: int = 250000,  # 250k iter1, 400k iter2 (total steps)
        gradient_clip_value: float = 1.0,  # TODO: Confirm this value
        dtype: jnp.dtype = jnp.float32,
    ):

        self.model_name = model_name
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir).absolute()
        self.lr_peak_value = lr_peak_value
        self.lr_end_value = lr_end_value
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.gradient_clip_value = gradient_clip_value

        self.model = HuBERTForTraining(
            num_label_embeddings=num_label_embeddings, dtype=dtype
        )

        self.log_dir = self.checkpoint_dir / model_name
        self.logger = wandb.init(
            project="hubert-base-jax",
            name=model_name,
            dir=self.log_dir,
        )

        self.create_functions()
        self.init_model(example_batch)

    @property
    def step(self):
        return self.state.step.item()

    def get_loss_function(self):
        def calculate_loss(params: dict, rng: PRNGKey, batch: Batch, train: bool):
            rng, dropout_apply_rng, masking_apply_rng = jax.random.split(rng, 3)

            rngs = (
                {"dropout": dropout_apply_rng, "masking": masking_apply_rng}
                if train
                else {}
            )

            logits, mask = self.model.apply(
                {"params": params},
                x=batch.padded_waveforms,
                padding_mask=batch.padding_mask,
                train=train,
                rngs=rngs,
            )

            if train:
                # use both padding mask and features mask
                loss_mask = jnp.logical_and(jnp.logical_not(batch.padding_mask), mask)
            else:
                # only use padding mask
                loss_mask = jnp.logical_not(batch.padding_mask)

            num_loss_entries = jnp.sum(loss_mask)

            # calculate masked loss
            losses = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch.padded_labels
            )
            losses = jnp.where(loss_mask, losses, 0.0)
            loss = jnp.sum(losses) / num_loss_entries

            # calculate masked accuracy
            correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.padded_labels)
            correct_pred = jnp.where(loss_mask, correct_pred, False)
            accuracy = jnp.sum(correct_pred) / num_loss_entries

            return loss, (accuracy, rng)

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

        self.rng, init_rng, dropout_init_rng, masking_init_rng = jax.random.split(
            self.rng, 4
        )

        params = self.model.init(
            {
                "params": init_rng,
                "dropout": dropout_init_rng,
                "masking": masking_init_rng,
            },
            x=example_batch.padded_waveforms,
            padding_mask=example_batch.padding_mask,
            train=True,
        )["params"]

        self.lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr_peak_value,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.lr_end_value,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.gradient_clip_value),
            optax.adamw(
                learning_rate=self.lr_schedule,
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
        best_val_accuracy = 0.0
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 1 == 0:  # validate every 1 epoch
                val_accuracy = self.eval_model(val_loader)
                wandb.log(
                    {"val/accuracy": val_accuracy},
                    step=self.step,
                )
                if val_accuracy >= best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    checkpoints.save_checkpoint(
                        ckpt_dir=self.log_dir, target=self.state.params, step=self.step
                    )

    def train_epoch(self, train_loader, epoch):
        with tqdm(total=len(train_loader), desc="Training", leave=False) as pbar:
            for i, batch in enumerate(train_loader):
                self.state, self.rng, loss, accuracy = self.train_step(
                    self.state, self.rng, batch
                )
                if i % 1 == 0:  # log every 1 steps
                    wandb.log(
                        {
                            "epoch": epoch,
                            "trainer/global_step": self.step,
                            "train/loss": loss,
                            "train/accuracy": accuracy,
                            "scheduler/lr": self.lr_schedule(self.step),
                        },
                        step=self.step,
                    )
                pbar.update(1)

    def eval_model(self, data_loader):
        total_accuracy, count = 0, 0
        with tqdm(total=len(data_loader), desc="Validating", leave=False) as pbar:
            for batch in data_loader:
                batch: Batch = batch  # for type hinting
                accuracy, self.rng = self.eval_step(self.state, self.rng, batch)
                total_accuracy += accuracy * batch.batch_size
                count += batch.batch_size
                pbar.update(1)
        eval_accuracy = (total_accuracy / count).item()
        return eval_accuracy
