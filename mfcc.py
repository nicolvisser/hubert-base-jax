from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

FRAME_LENGTH_MS = 25  # torchaudio.compliance.kaldi.mfcc default
FRAME_SHIFT_MS = 10  # torchaudio.compliance.kaldi.mfcc default


def extract_mfcc(wav, sr):
    assert sr == 16000

    frame_length = int(sr * FRAME_LENGTH_MS / 1000)
    frame_shift = int(sr * FRAME_SHIFT_MS / 1000)
    pad_length = (frame_length - frame_shift) // 2
    wav = F.pad(wav, (pad_length, pad_length))

    with torch.no_grad():
        mfccs = torchaudio.compliance.kaldi.mfcc(
            waveform=wav,
            sample_frequency=sr,
            use_energy=False,
            frame_length=FRAME_LENGTH_MS,
            frame_shift=FRAME_SHIFT_MS,
        )  # (time, freq)
        mfccs = mfccs.transpose(0, 1)  # (freq, time)
        deltas = torchaudio.functional.compute_deltas(mfccs)
        ddeltas = torchaudio.functional.compute_deltas(deltas)
        concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
        concat = concat.transpose(0, 1).contiguous()  # (freq, time)

    return concat


def _extract_and_dump_mfcc(audio_path: str, output_path: str):
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    wav, sr = torchaudio.load(audio_path)
    mfcc = extract_mfcc(wav, sr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mfcc.numpy())


def dump_mfcc_from_audio_dir(
    audio_dir: str,
    output_dir: str,
    extension: str = ".wav",
    show_progress: bool = True,
):
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)

    audio_paths = sorted(list(audio_dir.rglob(f"*{extension}")))
    output_paths = [
        output_dir / audio_path.relative_to(audio_dir).with_suffix(".npy")
        for audio_path in audio_paths
    ]

    args = list(zip(audio_paths, output_paths))

    with tqdm(total=len(args), disable=not show_progress) as pbar:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_extract_and_dump_mfcc, *arg) for arg in args]
            for future in as_completed(futures):
                future.result()  # to raise exception if any
                pbar.update(1)


@click.command()
@click.argument("audio_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--extension", type=str, default=".wav")
def _dump_mfcc_from_audio_dir_cli(
    audio_dir: str,
    output_dir: str,
    extension: str = ".wav",
    show_progress: bool = True,
):
    dump_mfcc_from_audio_dir(audio_dir, output_dir, extension, show_progress)


if __name__ == "__main__":
    _dump_mfcc_from_audio_dir_cli()
