from pathlib import Path

import click
import numpy as np
from sklearn import metrics


def upsample(a, b, a_rate, b_rate):
    if a_rate == b_rate:
        return a, b
    elif a_rate > b_rate:
        assert a_rate % b_rate == 0
        upsample_factor = a_rate // b_rate
        b = np.repeat(b, upsample_factor)
        return a, b
    elif a_rate < b_rate:
        assert b_rate % a_rate == 0
        upsample_factor = b_rate // a_rate
        a = np.repeat(a, upsample_factor)
        return a, b


def make_same_length_if_similar(a, b, max_len_diff=2):
    minlen = min(len(a), len(b))
    if abs(len(a) - len(b)) <= max_len_diff:
        return a[:minlen], b[:minlen]
    else:
        raise ValueError(f"Lengths are too different: {len(a)} vs {len(b)}")


def calculate_cluster_metrics(
    unit_ids_dir, phoneme_ids_dir, unit_label_rate, phoneme_label_rate
):
    unit_ids_dir, phoneme_ids_dir = Path(unit_ids_dir), Path(phoneme_ids_dir)

    unit_ids_paths = {p.stem: p for p in unit_ids_dir.rglob("*.npy")}
    phoneme_ids_paths = {p.stem: p for p in phoneme_ids_dir.rglob("*.npy")}

    common_stems = set(unit_ids_paths.keys()) & set(phoneme_ids_paths.keys())

    assert len(common_stems) > 0
    assert len(common_stems) == len(unit_ids_paths) == len(phoneme_ids_paths)

    def load(common_stem):
        unit_ids = np.load(unit_ids_paths[common_stem])
        phoneme_ids = np.load(phoneme_ids_paths[common_stem])
        unit_ids, phoneme_ids = upsample(
            unit_ids, phoneme_ids, unit_label_rate, phoneme_label_rate
        )
        unit_ids, phoneme_ids = make_same_length_if_similar(unit_ids, phoneme_ids)
        return unit_ids, phoneme_ids

    data = list(map(load, common_stems))
    unit_ids, phoneme_ids = list(zip(*data))
    unit_ids, phoneme_ids = np.concatenate(unit_ids), np.concatenate(phoneme_ids)

    unique_unit_ids = np.unique(unit_ids)
    unique_phoneme_ids = np.unique(phoneme_ids)

    max_unit_id = unique_unit_ids.max()
    max_phoneme_id = unique_phoneme_ids.max()

    assert set(unique_unit_ids) == set(range(max_unit_id + 1))
    assert set(unique_phoneme_ids) == set(range(max_phoneme_id + 1))

    n_yz = metrics.cluster.contingency_matrix(phoneme_ids, unit_ids)
    phone_purity = n_yz.max(axis=0).sum() / n_yz.sum()
    cluster_purity = n_yz.max(axis=1).sum() / n_yz.sum()
    pnmi = metrics.cluster.normalized_mutual_info_score(phoneme_ids, unit_ids)

    print(f"Cluster Purity: {cluster_purity:.2f}")
    print(f"Phone Purity:   {phone_purity:.2f}")
    print(f"PNMI:           {pnmi:.2f}")


@click.command()
@click.option("--unit_ids_dir", required=True, type=click.Path(exists=True))
@click.option("--phoneme_ids_dir", required=True, type=click.Path(exists=True))
@click.option("--unit_label_rate", required=True, type=int)
@click.option("--phoneme_label_rate", required=True, type=int)
def main(unit_ids_dir, phoneme_ids_dir, unit_label_rate, phoneme_label_rate):
    calculate_cluster_metrics(
        unit_ids_dir, phoneme_ids_dir, unit_label_rate, phoneme_label_rate
    )


if __name__ == "__main__":
    main()
