import torch

from smart_llm.llm.pooling import AttentionPool, pool_last, pool_mean, pool


def test_pool_last_picks_last_real_token():
    hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    out = pool_last(hidden, mask)
    # row 0: last real token is index 1; row 1: index 2
    assert torch.allclose(out[0], hidden[0, 1])
    assert torch.allclose(out[1], hidden[1, 2])


def test_pool_mean_respects_mask():
    hidden = torch.ones(1, 4, 5)
    hidden[0, 2:] = 10.0
    mask = torch.tensor([[1, 1, 0, 0]])
    out = pool_mean(hidden, mask)
    assert torch.allclose(out, torch.ones(1, 5))  # padded tokens excluded


def test_attention_pool_shape_and_grad():
    attn = AttentionPool(dim=8, hidden=16)
    hidden = torch.randn(3, 6, 8, requires_grad=True)
    mask = torch.ones(3, 6)
    out = attn(hidden, mask)
    assert out.shape == (3, 8)
    out.sum().backward()
    assert hidden.grad is not None


def test_pool_dispatch():
    hidden = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5)
    assert pool("last", hidden, mask).shape == (2, 4)
    assert pool("mean", hidden, mask).shape == (2, 4)
