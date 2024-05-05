import os
import wave

import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm

from model import HuBERTEncoder, make_padding_mask


class SortedChunkedDataset(Dataset):
    def __init__(self, dataset, num_chunks, sort_key_fn: lambda item: item):
        self.dataset = dataset
        print("Sorting dataset...")
        lengths = np.array([sort_key_fn(item) for item in dataset])
        print("Done sorting dataset.")
        self.sorted_idxs = np.argsort(lengths)[::-1]
        sorted_lengths = lengths[self.sorted_idxs]
        N = 2
        while N < len(dataset):
            N *= 2
        chunk_size = N // num_chunks
        starting_indices = np.arange(0, len(dataset), chunk_size)
        starting_lengths = sorted_lengths[starting_indices]
        self.padded_lengths = (
            starting_lengths[:, None] + np.zeros((1, chunk_size)).astype(int)
        ).ravel()[: len(dataset)]

    def __getitem__(self, idx):
        return self.dataset[self.sorted_idxs[idx]], self.padded_lengths[idx]

    def __len__(self):
        return len(self.sorted_idxs)


def collate_fn(batch):
    batch, padded_lengths = zip(*batch)
    max_len = max(padded_lengths)
    B = len(batch)
    stems = [f"{s}-{c}-{u:04d}" for _, _, _, s, c, u in batch]
    waveforms = [jnp.array(w[0]) for w, *_ in batch]
    unpadded_lengths = jnp.array([len(w) for w in waveforms])
    waveforms_padded = jnp.zeros((B, max_len))
    for i, w in enumerate(waveforms):
        waveforms_padded = waveforms_padded.at[i, : len(w)].set(w)
    waveforms_padded = waveforms_padded[:, None, :]
    return waveforms_padded, unpadded_lengths, stems


if __name__ == "__main__":

    dataset = LIBRISPEECH(
        root="/media/SSD/datasets",
        url="dev-clean",
        folder_in_archive="LibriSpeech",
        download=False,
    )

    sort_key_fn = lambda item: item[0].shape[1]
    dataset = SortedChunkedDataset(
        dataset=dataset, num_chunks=4, sort_key_fn=sort_key_fn
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    with open("checkpoints/hubert_bshall.bin", "rb") as f:
        params = flax.serialization.from_bytes(HuBERTEncoder, f.read())

    for layer_idx in range(12):
        model = HuBERTEncoder(num_layers=layer_idx + 1)

        @jax.jit
        def get_features_batch(waveforms, unpadded_lengths):
            features = model.apply(
                {"params": params}, waveforms, padding_mask, train=False
            )
            return features

        output_dir = f"output/layer_{layer_idx}"
        os.makedirs(output_dir, exist_ok=True)
        for waveforms_padded, unpadded_lengths, stems in tqdm(dataloader):
            padding_mask = make_padding_mask(unpadded_lengths)
            features = get_features_batch(waveforms_padded, padding_mask)
            feature_lengths = [l // 320 for l in unpadded_lengths]
            # save features
            # for f, l, s in zip(features, feature_lengths, stems):
            #     output_path = os.path.join(output_dir, f"{s}.npy")
            #     np.save(output_path, f[:l])
