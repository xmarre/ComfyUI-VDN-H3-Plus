from __future__ import annotations

import torch
import torch.nn.functional as F

from vdn_h3.audio_fix_train import AudioFixPair


def test_zero_init_first_update_trains_b_before_a():
    torch.manual_seed(17)
    pair = AudioFixPair(in_features=5, out_features=7, rank=3, alpha=3)
    x = torch.randn(11, 5)
    target = torch.randn(11, 7)

    first_loss = F.mse_loss(pair(x), target)
    first_loss.backward()

    assert pair.lora_A.grad is not None
    assert torch.count_nonzero(pair.lora_A.grad) == 0
    assert pair.lora_B.grad is not None
    assert torch.count_nonzero(pair.lora_B.grad) > 0

    # Model one optimizer update without depending on a particular optimizer. Once B is
    # nonzero, the factorized delta has a gradient path back into A on the next update.
    with torch.no_grad():
        pair.lora_B.add_(pair.lora_B.grad, alpha=-0.1)
    pair.zero_grad(set_to_none=True)

    second_loss = F.mse_loss(pair(x), target)
    second_loss.backward()

    assert pair.lora_A.grad is not None
    assert torch.count_nonzero(pair.lora_A.grad) > 0
    assert pair.lora_B.grad is not None
    assert torch.count_nonzero(pair.lora_B.grad) > 0
