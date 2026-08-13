"""Runtime and command-line primitives for synthesizer runners."""

import argparse
import random

import numpy as np
import torch


class WorkflowHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults for optional arguments without confusing required ones."""

    def _get_help_string(self, action):
        if action.required:
            return action.help
        return super()._get_help_string(action)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_device(value: str) -> str:
    """Validate a Torch device exposed by synthesizer runners."""
    if value in {'auto', 'cpu', 'cuda'} or (
        value.startswith('cuda:') and value.removeprefix('cuda:').isdigit()
    ):
        return value
    raise argparse.ArgumentTypeError(
        '--device must be auto, cpu, cuda, or a CUDA device such as cuda:1.'
    )


def parse_dims(value: str) -> tuple[int, ...]:
    """Parse a comma-separated neural-network dimension list."""
    try:
        dims = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    except ValueError as exc:
        raise ValueError(
            f'Invalid dimensions {value!r}; expected values like 128,128.'
        ) from exc
    if not dims or any(dim < 1 for dim in dims):
        raise ValueError(
            f'Invalid dimensions {value!r}; every dimension must be positive.'
        )
    return dims


def parse_columns(value: str | None) -> list[str] | None:
    """Parse a comma-separated column list, with none meaning no columns."""
    if value is None:
        return None
    if value.strip().lower() in {'', 'none', 'null'}:
        return []
    columns = [column.strip() for column in value.split(',') if column.strip()]
    if len(columns) != len(set(columns)):
        raise ValueError('--categorical-columns contains duplicate column names.')
    return columns
