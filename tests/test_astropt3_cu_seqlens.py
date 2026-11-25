import torch

from nanotron.models.astropt3 import _packed_cu_seqlens


def test_packed_cu_seqlens_collapses_pad_runs_per_row():
    # Row 0: obj1 (len 3), obj2 (len 2), 3 trailing pads
    # Row 1: 8 pads (all-pad tail row)
    # Row 2: 2 leading pads, obj3 (len 4), 2 trailing pads
    input_ids = torch.tensor(
        [
            [1, 2, 3, 1, 4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 4, 0, 0],
        ]
    )
    position_ids = torch.tensor(
        [
            [0, 1, 2, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 3, 0, 0],
        ]
    )

    cu = _packed_cu_seqlens(input_ids, position_ids, pad_token_id=0)
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 3, 5, 8, 16, 18, 22, 24]
    assert (cu[1:] - cu[:-1]).max().item() <= input_ids.shape[1]


def test_packed_cu_seqlens_large_padded_microbatch_stays_under_cuda_grid_limit():
    # 32 x 4096 = 131,072 tokens with >65,535 pad tokens previously exceeded FA2's 65,535 gridDim.z limit.
    mbs, seq_len = 32, 4096
    input_ids = torch.zeros((mbs, seq_len), dtype=torch.long)
    position_ids = torch.zeros((mbs, seq_len), dtype=torch.long)
    # Put two packed objects in row 0; leave rows 1..31 as all-pad draw tail.
    input_ids[0, :300] = 1
    position_ids[0, :150] = torch.arange(150)
    position_ids[0, 150:300] = torch.arange(150)

    cu = _packed_cu_seqlens(input_ids, position_ids, pad_token_id=0)
    num_segments = len(cu) - 1
    # Row 0: 2 objects + 1 trailing pad run; rows 1..31: 1 pad segment each.
    assert num_segments == 34
    assert num_segments < 65535
    assert (cu[1:] - cu[:-1]).max().item() == seq_len


def test_packed_cu_seqlens_without_pad_id_matches_zero_restarts():
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 2]])
    cu = _packed_cu_seqlens(input_ids, position_ids, pad_token_id=None)
    assert cu.tolist() == [0, 2, 3, 4, 5, 8]
