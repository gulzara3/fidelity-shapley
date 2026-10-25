import numpy as np
import torch

from fidelity_shapley.models.surrogate import MaskedGRU, batched_pred, train_surrogate


def test_forward_shape_and_mask_concatenation():
    net = MaskedGRU(3, 2, hidden=8)
    assert net.gru.input_size == 6
    out = net(torch.zeros(4, 5, 3), torch.ones(4, 3))
    assert out.shape == (4, 2)


def test_masked_channel_cannot_leak():
    torch.manual_seed(0)
    net = MaskedGRU(3, 1, hidden=8).eval()
    x = torch.randn(2, 5, 3)
    x2 = x.clone()
    x2[:, :, 1] = 100.0                                   # change only channel 1
    m = torch.tensor([[1.0, 0.0, 1.0]] * 2)
    assert torch.allclose(net(x, m), net(x2, m))


def test_training_reduces_error_and_resumes(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(512, 6, 3)).astype(np.float32)
    Y = X[:, -1, :1].copy()
    dev = torch.device("cpu")
    ck = tmp_path / "net.pt"
    net = train_surrogate(X, Y, 3, 1, 0, device=dev, hidden=16, steps=150, bs=64,
                          warmup=10, ckpt=ck)
    err = np.mean((batched_pred(net, X, np.ones(3), device=dev) - Y) ** 2)
    assert err < 0.5 * Y.var()
    again = train_surrogate(X, Y, 3, 1, 0, device=dev, hidden=16, ckpt=ck)   # resumes
    np.testing.assert_allclose(batched_pred(again, X, np.ones(3), device=dev),
                               batched_pred(net, X, np.ones(3), device=dev))
