""" Taken from fairseq utils https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/data_utils.py
    as it was used in origninal HuBERT model https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/hubert/hubert.py
"""

from typing import Optional, Tuple

import numpy as np


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[np.ndarray],
    mask_prob: float,
    mask_length: int = 10,
    min_masks: int = 2,
) -> np.ndarray:

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    mask_idcs = []
    for i in range(bsz):
        rng = np.random.default_rng()

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].sum()
            assert sz >= 0, sz
        else:
            sz = all_sz

        num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * sz / float(mask_length)
            + rng.random()
        )
        num_mask = max(min_masks, num_mask)

        lengths = np.full(num_mask, mask_length)

        if sum(lengths) == 0:
            raise ValueError(f"this should never happens")

        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1
        mask_idc = rng.choice(sz - min_len, num_mask, replace=False)

        mask_idc = np.asarray(
            [
                mask_idc[j] + offset
                for j in range(len(mask_idc))
                for offset in range(lengths[j])
            ]
        )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            raise ValueError(
                (
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={None}"
                )
            )
        mask_idcs.append(mask_idc)

    target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

    return mask
