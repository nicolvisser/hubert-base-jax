import random
from pathlib import Path

import chex
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from masking import compute_mask_indices


class TrainingDataset(Dataset):
    def __init__(
        self,
        waveforms_dir: str,
        labels_dir: str,
        label_rate: int,
        num_unique_labels: int,
        waveforms_extension: str = ".flac",
        random_crop: bool = True,
        max_sample_size=256000,
        min_sample_size=32000,
    ):
        super().__init__()
        self.waveforms_dir = waveforms_dir
        self.labels_dir = labels_dir
        self.label_rate = label_rate
        self.num_unique_labels = num_unique_labels
        self.random_crop = random_crop
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size

        assert self.max_sample_size % 320 == 0
        assert self.min_sample_size % 320 == 0

        waveforms_paths = {
            p.stem: p for p in Path(waveforms_dir).rglob(f"*{waveforms_extension}")
        }
        labels_paths = {p.stem: p for p in Path(labels_dir).rglob("*.npy")}

        print(f"Found {len(waveforms_paths)} waveform files")
        print(f"Found {len(labels_paths)} label files")

        common_stems = set(waveforms_paths.keys()) & set(labels_paths.keys())

        print(f"Found {len(common_stems)} common stems")
        assert len(common_stems) > 0

        self.waveforms_paths = [waveforms_paths[stem] for stem in common_stems]
        self.labels_paths = [labels_paths[stem] for stem in common_stems]

        assert label_rate == 50 or label_rate == 100
        self.downsample_labels = True if label_rate == 100 else False

    def __len__(self) -> int:
        return len(self.waveforms_paths)

    def __getitem__(self, idx: int):
        waveform_path = self.waveforms_paths[idx]
        label_path = self.labels_paths[idx]

        waveform, sr = sf.read(waveform_path)
        labels = np.load(label_path)

        assert sr == 16000

        divisor = 160 if self.downsample_labels else 320

        if len(waveform) < self.min_sample_size:
            # too short, sample another file
            return self.__getitem__(random.randint(0, len(self) - 1))

        if len(waveform) > self.max_sample_size:
            cropped_waveform_length = self.max_sample_size
            # too long, crop it
            if self.random_crop:
                # choose a multiple of 320 smaller than len(waveform)
                possible_starts = np.arange(
                    0, len(waveform) - self.max_sample_size, 320
                )
                waveform_start = random.choice(possible_starts)
            else:
                waveform_start = 0
            waveform = waveform[
                waveform_start : waveform_start + cropped_waveform_length
            ]

            # also crop the labels

            cropped_label_length = len(waveform) // divisor
            label_start = waveform_start // divisor
            labels = labels[label_start : label_start + cropped_label_length]

        # crop waveform to a multiple of 320
        waveform = waveform[: len(waveform) // divisor * divisor]

        if self.downsample_labels:
            labels = labels[::2]

        divisor = 320

        expected_label_length = len(waveform) // divisor

        # if for some reason there is still a mismatch between waveform and labels, pad or crop
        num_missing_labels = abs(expected_label_length - len(labels))
        if num_missing_labels > 2:
            print(
                f"Warning: mismatch beteen waveform and labels too large. Check your data!. expected: {expected_label_length}, got: {len(labels)}"
            )
        if len(labels) < expected_label_length:
            # pad labels with randomly sampled labels
            random_labels = np.random.randint(
                0, self.num_unique_labels, num_missing_labels
            )
            labels = np.concatenate([labels, random_labels])
        if len(labels) > expected_label_length:
            # crop labels
            labels = labels[:expected_label_length]

        waveform_padding = self.max_sample_size - len(waveform)
        labels_padding = self.max_sample_size // divisor - len(labels)

        unpadded_labels_length = len(labels)

        padded_waveform = np.pad(waveform, (0, waveform_padding))
        padded_labels = np.pad(labels, (0, labels_padding))

        padding_mask = np.zeros_like(padded_labels, dtype=np.bool_)
        padding_mask[unpadded_labels_length:] = True

        return (
            padded_waveform,
            padded_labels,
            padding_mask,
        )

    def collate_fn(self, batch):
        (
            padded_waveforms,
            padded_labels,
            padding_mask,
        ) = zip(*batch)

        padded_waveforms = np.stack(padded_waveforms)  # B, t
        padded_labels = np.stack(padded_labels)  # B, T
        padding_mask = np.stack(padding_mask)  # B, T

        feature_mask = compute_mask_indices(
            shape=padded_labels.shape,
            padding_mask=padding_mask,
            mask_prob=0.65,
            mask_length=10,
            min_masks=2,
        )

        return Batch(
            padded_waveforms=np.array(padded_waveforms)[:, None, :],  # B, 1, t
            padded_labels=np.array(padded_labels),  # B, T
            padding_mask=np.array(padding_mask),  # B, T
            feature_mask=np.array(feature_mask),  # B, T
        )


@chex.dataclass
class Batch:
    padded_waveforms: np.ndarray
    padded_labels: np.ndarray
    padding_mask: np.ndarray
    feature_mask: np.ndarray

    def batch_size(self):
        return self.padded_waveforms.shape[0]
