from __future__ import annotations

import pytest
import torch

from l2kv.position_utils import make_cache_position, make_position_ids


def test_logical_positions_have_expected_shapes_and_values() -> None:
    position_ids = make_position_ids(17, 3, "cpu")

    assert position_ids.shape == (1, 3)
    assert position_ids.tolist() == [[17, 18, 19]]
    assert make_cache_position(position_ids).tolist() == [17, 18, 19]


@pytest.mark.parametrize(
    ("start_position", "length"),
    [(-1, 1), (0, 0)],
)
def test_invalid_logical_position_arguments_raise(
    start_position: int,
    length: int,
) -> None:
    with pytest.raises(ValueError):
        make_position_ids(start_position, length, "cpu")


def test_cache_position_requires_single_batch() -> None:
    with pytest.raises(ValueError):
        make_cache_position(torch.zeros((2, 1), dtype=torch.long))
