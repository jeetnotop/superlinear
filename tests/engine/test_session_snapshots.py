import pytest


torch = pytest.importorskip("torch", reason="torch not installed")


from superlinear.engine.session_snapshots import (
    SnapshotStoreV1,
    export_hybrid_mamba_attention_static_cache,
    import_hybrid_mamba_attention_static_cache,
)


class _DummyCache:
    def __init__(self, *, num_layers: int, transformer_layers: list[int], max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len
        self.transformer_layers = transformer_layers
        self._layer_seen_tokens = [0] * num_layers

        # Attention KV: only allocate for transformer layers; others are (B, 0) placeholders.
        self.key_cache = [torch.empty((1, 0)) for _ in range(num_layers)]
        self.value_cache = [torch.empty((1, 0)) for _ in range(num_layers)]
        for li in transformer_layers:
            self.key_cache[li] = torch.zeros((1, 2, max_seq_len, 4), dtype=torch.float16)
            self.value_cache[li] = torch.zeros((1, 2, max_seq_len, 4), dtype=torch.float16)

        # Mamba state buffers (shapes intentionally small for tests).
        self.conv_states = torch.zeros((num_layers, 1, 8, 4), dtype=torch.float16)
        self.ssm_states = torch.zeros((num_layers, 1, 2, 4, 3), dtype=torch.float32)


def test_cache_export_import_roundtrip():
    cache = _DummyCache(num_layers=4, transformer_layers=[1, 3], max_seq_len=8)
    current_pos = 5

    # Seed deterministic tensors.
    cache.key_cache[1][:, :, :current_pos, :].copy_(torch.randn((1, 2, current_pos, 4), dtype=torch.float16))
    cache.value_cache[1][:, :, :current_pos, :].copy_(torch.randn((1, 2, current_pos, 4), dtype=torch.float16))
    cache.key_cache[3][:, :, :current_pos, :].copy_(torch.randn((1, 2, current_pos, 4), dtype=torch.float16))
    cache.value_cache[3][:, :, :current_pos, :].copy_(torch.randn((1, 2, current_pos, 4), dtype=torch.float16))
    cache.conv_states.copy_(torch.randn_like(cache.conv_states))
    cache.ssm_states.copy_(torch.randn_like(cache.ssm_states))

    payload = export_hybrid_mamba_attention_static_cache(cache=cache, current_pos=current_pos)

    restored = _DummyCache(num_layers=4, transformer_layers=[1, 3], max_seq_len=8)
    restored_pos = import_hybrid_mamba_attention_static_cache(cache=restored, payload=payload)
    assert restored_pos == current_pos

    for li in [1, 3]:
        assert torch.allclose(restored.key_cache[li][:, :, :current_pos, :], cache.key_cache[li][:, :, :current_pos, :])
        assert torch.allclose(
            restored.value_cache[li][:, :, :current_pos, :], cache.value_cache[li][:, :, :current_pos, :]
        )

    assert torch.allclose(restored.conv_states, cache.conv_states)
    assert torch.allclose(restored.ssm_states, cache.ssm_states)

    # Cursor sync is required for session-append prefill correctness.
    assert restored._layer_seen_tokens[1] == current_pos
    assert restored._layer_seen_tokens[3] == current_pos


def test_cache_import_allows_larger_target_allocation():
    # Export from a small cache...
    src = _DummyCache(num_layers=2, transformer_layers=[0], max_seq_len=4)
    current_pos = 3
    src.key_cache[0][:, :, :current_pos, :].copy_(torch.randn((1, 2, current_pos, 4), dtype=torch.float16))
    src.value_cache[0][:, :, :current_pos, :].copy_(torch.randn((1, 2, current_pos, 4), dtype=torch.float16))
    src.conv_states.copy_(torch.randn_like(src.conv_states))
    src.ssm_states.copy_(torch.randn_like(src.ssm_states))

    payload = export_hybrid_mamba_attention_static_cache(cache=src, current_pos=current_pos)

    # ...and import into a larger cache.
    dst = _DummyCache(num_layers=2, transformer_layers=[0], max_seq_len=8)
    restored_pos = import_hybrid_mamba_attention_static_cache(cache=dst, payload=payload)
    assert restored_pos == current_pos
    assert torch.allclose(dst.key_cache[0][:, :, :current_pos, :], src.key_cache[0][:, :, :current_pos, :])
    assert torch.allclose(dst.value_cache[0][:, :, :current_pos, :], src.value_cache[0][:, :, :current_pos, :])
    assert torch.allclose(dst.conv_states, src.conv_states)
    assert torch.allclose(dst.ssm_states, src.ssm_states)
    assert dst._layer_seen_tokens[0] == current_pos


def test_snapshot_store_roundtrip(tmp_path):
    store = SnapshotStoreV1(root_dir=tmp_path, model_id="m", compat={"fingerprint": "fp", "payload": {}})
    cache = _DummyCache(num_layers=2, transformer_layers=[0], max_seq_len=4)
    cache.key_cache[0][:, :, :2, :].copy_(torch.randn((1, 2, 2, 4), dtype=torch.float16))
    cache.value_cache[0][:, :, :2, :].copy_(torch.randn((1, 2, 2, 4), dtype=torch.float16))
    cache.conv_states.copy_(torch.randn_like(cache.conv_states))
    cache.ssm_states.copy_(torch.randn_like(cache.ssm_states))

    cache_payload = export_hybrid_mamba_attention_static_cache(cache=cache, current_pos=2)
    manifest = store.create_snapshot(
        transcript=[{"role": "user", "content": "hi"}],
        cache_payload=cache_payload,
        session={"max_seq_len": 4, "current_pos": 2},
        title="t",
        description="d",
        tags=["x"],
    )

    listed = store.list_snapshots()
    assert [m.snapshot_id for m in listed] == [manifest.snapshot_id]

    mf, transcript, loaded_payload = store.load_snapshot_payload(manifest.snapshot_id)
    assert mf.snapshot_id == manifest.snapshot_id
    assert transcript == [{"role": "user", "content": "hi"}]
    assert loaded_payload["cache_format"] == "hybrid_mamba_attention_static_v1"

    updated = store.patch_metadata(manifest.snapshot_id, title="t2", tags=["a", "b"])
    assert updated.metadata["title"] == "t2"
    assert updated.metadata["tags"] == ["a", "b"]

    store.delete_snapshot(manifest.snapshot_id)
    assert store.list_snapshots() == []

