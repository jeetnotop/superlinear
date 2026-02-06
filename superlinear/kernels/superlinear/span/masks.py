"""Mask utilities for span attention."""

from __future__ import annotations

import torch


def create_span_mask(
    qstart: torch.Tensor,
    qend: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    k: int = 0,
    sw_index: int = 0,
):
    """
    Create a mask generator function for span attention.

    Args:
        qstart: Start indices of spans [B, H, Q, K]
        qend: End indices of spans [B, H, Q, K]
        attention_mask: Optional attention mask [B, L]
        k: Index of the span to use
        sw_index: Sliding window index

    Returns:
        A function create_mask(b, h, q_idx, kv_idx) -> bool
    """
    if attention_mask is not None:
        attention_mask_bool = attention_mask.clone().to(torch.bool)
    else:
        attention_mask_bool = None

    qstart_k = qstart[:, :, :, k].clone()  # Shape: [B, H, Q]
    qend_k = qend[:, :, :, k].clone()      # Shape: [B, H, Q]

    def create_mask(b, h, q_idx, kv_idx):
        start = qstart_k[b, h, q_idx]
        end = qend_k[b, h, q_idx]
        mask = (kv_idx >= start) & (kv_idx <= end)

        if sw_index > 0:
            diff = q_idx - kv_idx + 1
            window = (sw_index + 1) ** 2 - 1
            mask = mask | ((diff <= window) & (diff > 0))

        if attention_mask_bool is not None:
            mask = mask & attention_mask_bool[b][kv_idx]

        return mask

    return create_mask


def create_sort_indices(
    qstart: torch.Tensor,
    qend: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Create sort indices for span-sorted attention.

    Args:
        qstart: Start indices of spans [B, H, Q, K]
        qend: End indices of spans [B, H, Q, K]
        k: Index of the span to use

    Returns:
        sorted_indices: [B, H, L] indices for sorting
    """
    device = qstart.device
    qstart_all = qstart[:, :, :, k].clone()
    qstart_all = torch.where(qstart_all < 0, -1, qstart_all)
    qend_all = qend[:, :, :, k].clone()
    qend_all = torch.where(
        qend_all < 0,
        -torch.arange(len(qend_all[0, 0])).to(device),
        qend_all,
    )

    max_val = max(qstart_all.max().item(), qend_all.max().item()) + 1
    combined_key_all = qstart_all * max_val + qend_all

    sorted_indices = torch.argsort(combined_key_all, dim=-1)
    return sorted_indices


def create_sorted_span_mask(
    qstart: torch.Tensor,
    qend: torch.Tensor,
    sorted_indices: torch.Tensor,
    k: int = 0,
    attention_mask: torch.Tensor | None = None,
    sw_index: int = 0,
):
    """
    Create a mask generator function for sorted span attention.

    Args:
        qstart: Start indices of spans [B, H, Q, K]
        qend: End indices of spans [B, H, Q, K]
        sorted_indices: Sorting indices [B, H, L]
        k: Index of the span to use
        attention_mask: Optional attention mask [B, L]
        sw_index: Sliding window index

    Returns:
        A function create_mask(b, h, q_idx, kv_idx) -> bool
    """
    qstart_k = qstart[:, :, :, k].clone()
    qend_k = qend[:, :, :, k].clone()
    sorted_indices = sorted_indices.clone()

    qstart_k_sorted = torch.gather(qstart_k, dim=-1, index=sorted_indices)
    qend_k_sorted = torch.gather(qend_k, dim=-1, index=sorted_indices)

    if attention_mask is not None:
        attention_mask_bool = attention_mask.clone().to(torch.bool)
    else:
        attention_mask_bool = None

    def create_mask(b, h, q_idx, kv_idx):
        start = qstart_k_sorted[b, h, q_idx]
        end = qend_k_sorted[b, h, q_idx]
        mask = (kv_idx >= start) & (kv_idx <= end)
        if sw_index > 0:
            diff = q_idx - kv_idx + 1
            window = (sw_index + 1) ** 2 - 1
            mask = mask & (diff > window)
        if attention_mask_bool is not None:
            mask = mask & attention_mask_bool[b][kv_idx]
        return mask

    return create_mask


def create_sliding_window_mask(
    sw_index: int = 0,
    attention_mask: torch.Tensor | None = None,
):
    """
    Create a mask generator function for sliding window attention.

    Args:
        sw_index: Sliding window index
        attention_mask: Optional attention mask [B, L]

    Returns:
        A function create_mask(b, h, q_idx, kv_idx) -> bool
    """
    if attention_mask is not None:
        attention_mask_bool = attention_mask.clone().to(torch.bool)
    else:
        attention_mask_bool = None

    window = (sw_index + 1) ** 2 - 1

    def create_mask(b, h, q_idx, kv_idx):
        diff = q_idx - kv_idx + 1
        mask = (diff <= window) & (diff > 0)
        if attention_mask_bool is not None:
            mask = mask & attention_mask_bool[b][kv_idx]
        return mask

    return create_mask


def invert_sorted_matrix(
    sorted_matrix: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Invert a sorted matrix back to original order.

    Args:
        sorted_matrix: Sorted tensor [B, H, L, ...] or [B, H, L]
        sorted_indices: Indices used for sorting [B, H, L]

    Returns:
        Tensor restored to original ordering
    """
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    if sorted_matrix.dim() == 4:
        inverse_indices = inverse_indices.unsqueeze(-1).expand(
            -1, -1, -1, sorted_matrix.shape[-1]
        )
    reconstructed_matrix = torch.gather(sorted_matrix, dim=2, index=inverse_indices)
    return reconstructed_matrix
