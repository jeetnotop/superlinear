import torch


class _DummyCache:
    def __init__(self, n_layers: int, transformer_layers: list[int]):
        self._layer_seen_tokens = [0 for _ in range(n_layers)]
        self.transformer_layers = transformer_layers


def test_sync_static_cache_seen_tokens_updates_all_layers() -> None:
    # Import locally to avoid heavy imports at module load time.
    from superlinear.engine.adapters.superlinear import SuperlinearAdapter

    adapter = SuperlinearAdapter()

    cache = _DummyCache(n_layers=6, transformer_layers=[1, 3, 5])
    adapter._sync_static_cache_seen_tokens(cache, current_pos=1234)

    assert cache._layer_seen_tokens == [1234, 1234, 1234, 1234, 1234, 1234]
