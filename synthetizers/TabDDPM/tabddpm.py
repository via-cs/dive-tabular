"""TAB-DDPM preprocessing, training, checkpointing, and sampling.

The diffusion and denoising modules are the official TAB-DDPM implementation.
This module replaces its legacy Python 3.9 experiment stack with a small API
that works with the current repository and its uv-managed environment.
"""

from __future__ import annotations

import contextlib
import io
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from .modules import MLPDiffusion


TASK_ALIASES = {
    'binary_classification': 'binclass',
    'binclass': 'binclass',
    'multiclass_classification': 'multiclass',
    'multiclass': 'multiclass',
    'regression': 'regression',
}


def resolve_device(value: str) -> torch.device:
    """Resolve and validate an auto/CPU/CUDA device string."""
    if value == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if value == 'cuda':
        value = 'cuda:0'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA device requested but CUDA is unavailable: {value}')
    return device


def load_info_column_types(
    info_path: Path,
    columns: list[str],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Read and validate a complete categorical/numerical partition."""
    if not info_path.exists():
        raise FileNotFoundError(f'TAB-DDPM requires info.json: {info_path}')
    info = json.loads(info_path.read_text(encoding='utf-8'))
    col_types = info.get('col_types')
    resolved: dict[str, str]
    if isinstance(col_types, dict):
        resolved = {
            name: metadata.get('type') if isinstance(metadata, dict) else None
            for name, metadata in col_types.items()
        }
    elif isinstance(col_types, list):
        order = info.get('features_by_order')
        if not isinstance(order, list) or len(order) != len(col_types):
            raise ValueError(
                f'{info_path} must align list-valued col_types with features_by_order.'
            )
        resolved = dict(zip(order, col_types))
    else:
        raise ValueError(
            f'{info_path} must contain an object or list named "col_types".'
        )

    missing = [column for column in columns if column not in resolved]
    extra = [column for column in resolved if column not in columns]
    invalid = {
        column: resolved[column]
        for column in columns
        if column in resolved and resolved[column] not in {'cat', 'num'}
    }
    if missing or extra or invalid:
        raise ValueError(
            f'{info_path} column types do not partition data.csv. '
            f'Missing: {missing}; extra: {extra}; invalid: {invalid}.'
        )
    categorical = [column for column in columns if resolved[column] == 'cat']
    numerical = [column for column in columns if resolved[column] == 'num']
    return categorical, numerical, info


def infer_task_type(
    target: pd.Series,
    target_is_categorical: bool,
    declared_task: str | None,
) -> str:
    """Resolve the TAB-DDPM task from metadata, falling back to target data."""
    if declared_task is not None:
        if declared_task not in TASK_ALIASES:
            raise ValueError(f'Unsupported task in info.json: {declared_task!r}')
        return TASK_ALIASES[declared_task]
    unique = int(target.nunique(dropna=False))
    if unique == 2:
        return 'binclass'
    if target_is_categorical or not pd.api.types.is_numeric_dtype(target) or unique <= 20:
        return 'multiclass'
    return 'regression'


def _sorted_values(series: pd.Series) -> list[Any]:
    values = series.drop_duplicates().tolist()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: (type(value).__name__, str(value)))


def _make_normalizer(name: str, row_count: int, seed: int):
    if name == 'quantile':
        requested = max(min(row_count // 30, 1000), 10)
        return QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(requested, row_count),
            random_state=seed,
        )
    if name == 'standard':
        return StandardScaler()
    if name == 'minmax':
        return MinMaxScaler()
    raise ValueError(f'Unknown numerical normalization: {name!r}')


@dataclass
class TabDDPMPreprocessor:
    """Fit train-only transforms and restore generated rows."""

    column_order: list[str]
    target: str
    categorical_columns: list[str]
    numerical_columns: list[str]
    task_type: str
    normalization: str = 'quantile'
    seed: int = 0
    feature_categorical: list[str] = field(init=False)
    feature_numerical: list[str] = field(init=False)
    category_values: dict[str, list[Any]] = field(default_factory=dict, init=False)
    target_values: list[Any] = field(default_factory=list, init=False)
    normalizer: Any = field(default=None, init=False)
    fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        expected = set(self.column_order)
        categorical = set(self.categorical_columns)
        numerical = set(self.numerical_columns)
        if categorical & numerical or categorical | numerical != expected:
            raise ValueError('Categorical and numerical columns must partition the table.')
        if self.target not in expected:
            raise ValueError(f'Target column is absent from the table: {self.target!r}')
        if self.task_type == 'regression' and self.target not in numerical:
            raise ValueError('A regression target must be numerical in info.json.')
        if self.task_type != 'regression' and self.target not in categorical:
            raise ValueError('A classification target must be categorical in info.json.')
        self.feature_categorical = [
            column for column in self.column_order
            if column in categorical and column != self.target
        ]
        self.feature_numerical = [
            column for column in self.column_order
            if column in numerical and column != self.target
        ]

    @property
    def is_classification(self) -> bool:
        return self.task_type in {'binclass', 'multiclass'}

    @property
    def num_classes(self) -> int:
        return len(self.target_values) if self.is_classification else 0

    @property
    def category_sizes(self) -> list[int]:
        return [len(self.category_values[column]) for column in self.feature_categorical]

    @property
    def model_numerical_columns(self) -> list[str]:
        if self.task_type == 'regression':
            return [self.target, *self.feature_numerical]
        return list(self.feature_numerical)

    @property
    def num_numerical_features(self) -> int:
        return len(self.model_numerical_columns)

    def fit_transform(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit on the training split and return model matrix and labels."""
        if list(train.columns) != self.column_order:
            raise ValueError('Training columns do not match the configured column order.')
        if train.empty:
            raise ValueError('TAB-DDPM cannot train on an empty dataframe.')

        self.category_values = {
            column: _sorted_values(train[column])
            for column in self.feature_categorical
        }
        categorical_parts = []
        for column in self.feature_categorical:
            mapping = {
                value: index for index, value in enumerate(self.category_values[column])
            }
            encoded = train[column].map(mapping)
            if encoded.isna().any():
                raise ValueError(f'Could not encode categorical column {column!r}.')
            categorical_parts.append(encoded.to_numpy(dtype=np.float32)[:, None])

        numerical_columns = self.model_numerical_columns
        if numerical_columns:
            numerical = train[numerical_columns].to_numpy(dtype=np.float64)
            if not np.isfinite(numerical).all():
                raise ValueError('Numerical TAB-DDPM inputs must be finite.')
            self.normalizer = _make_normalizer(self.normalization, len(train), self.seed)
            numerical = self.normalizer.fit_transform(numerical).astype(np.float32)
        else:
            numerical = np.empty((len(train), 0), dtype=np.float32)

        if categorical_parts:
            categorical = np.hstack(categorical_parts).astype(np.float32)
            matrix = np.hstack([numerical, categorical]).astype(np.float32)
        else:
            matrix = numerical

        if self.is_classification:
            self.target_values = _sorted_values(train[self.target])
            target_mapping = {
                value: index for index, value in enumerate(self.target_values)
            }
            labels = train[self.target].map(target_mapping).to_numpy(
                dtype=np.int64, copy=True
            )
            expected_classes = 2 if self.task_type == 'binclass' else 3
            if len(self.target_values) < expected_classes:
                raise ValueError(
                    f'{self.task_type} requires at least {expected_classes} target classes.'
                )
        else:
            labels = np.zeros(len(train), dtype=np.int64)

        self.fitted = True
        return matrix, labels

    def inverse_transform(self, matrix: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        """Convert sampled model values into the encoded repository dataframe."""
        if not self.fitted:
            raise RuntimeError('TAB-DDPM preprocessor has not been fitted.')
        matrix = np.asarray(matrix)
        expected_width = self.num_numerical_features + len(self.feature_categorical)
        if matrix.ndim != 2 or matrix.shape[1] != expected_width:
            raise ValueError(
                f'Generated matrix has shape {matrix.shape}; expected width {expected_width}.'
            )
        result: dict[str, Any] = {}
        n_num = self.num_numerical_features
        if n_num:
            numerical = self.normalizer.inverse_transform(matrix[:, :n_num])
            for index, column in enumerate(self.model_numerical_columns):
                result[column] = numerical[:, index]

        for offset, column in enumerate(self.feature_categorical):
            values = self.category_values[column]
            indices = np.rint(matrix[:, n_num + offset]).astype(int)
            indices = np.clip(indices, 0, len(values) - 1)
            result[column] = np.asarray(values, dtype=object)[indices]

        if self.is_classification:
            indices = np.asarray(labels).astype(int).reshape(-1)
            indices = np.clip(indices, 0, len(self.target_values) - 1)
            result[self.target] = np.asarray(self.target_values, dtype=object)[indices]

        return pd.DataFrame(result, columns=self.column_order)


class TabDDPMSynthesizer:
    """Train and sample the official TAB-DDPM MLP diffusion model."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        preprocessor: TabDDPMPreprocessor,
        *,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
        time_embedding_dim: int = 128,
        num_timesteps: int = 1000,
        scheduler: str = 'cosine',
        device: str = 'auto',
    ) -> None:
        self.preprocessor = preprocessor
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)
        self.time_embedding_dim = int(time_embedding_dim)
        self.num_timesteps = int(num_timesteps)
        self.scheduler = scheduler
        self.device = resolve_device(device)
        self.raw_state_dict: dict[str, torch.Tensor] | None = None
        self.ema_state_dict: dict[str, torch.Tensor] | None = None
        self.class_probabilities: list[float] = [1.0]

    def _category_array(self) -> np.ndarray:
        sizes = self.preprocessor.category_sizes
        return np.asarray(sizes if sizes else [0], dtype=np.int64)

    def _build_model(self) -> MLPDiffusion:
        category_sizes = self._category_array()
        d_in = self.preprocessor.num_numerical_features + int(category_sizes.sum())
        return MLPDiffusion(
            d_in=d_in,
            num_classes=self.preprocessor.num_classes,
            is_y_cond=self.preprocessor.is_classification,
            rtdl_params={
                'd_layers': list(self.hidden_dims),
                'dropout': self.dropout,
            },
            dim_t=self.time_embedding_dim,
        )

    def _build_diffusion(self, model: MLPDiffusion) -> GaussianMultinomialDiffusion:
        return GaussianMultinomialDiffusion(
            num_classes=self._category_array(),
            num_numerical_features=self.preprocessor.num_numerical_features,
            denoise_fn=model,
            num_timesteps=self.num_timesteps,
            gaussian_loss_type='mse',
            scheduler=self.scheduler,
            device=self.device,
        ).to(self.device)

    def fit(
        self,
        train: pd.DataFrame,
        *,
        steps: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        ema_decay: float,
        seed: int,
        log_every: int = 100,
    ) -> pd.DataFrame:
        """Fit the diffusion model for a fixed number of optimizer steps."""
        if steps < 1 or batch_size < 1 or log_every < 1:
            raise ValueError('steps, batch_size, and log_every must be positive.')
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError('ema_decay must be in [0, 1).')
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        matrix, labels = self.preprocessor.fit_transform(train)
        if self.preprocessor.is_classification:
            counts = np.bincount(labels, minlength=self.preprocessor.num_classes)
            self.class_probabilities = (counts / counts.sum()).tolist()

        data = TensorDataset(torch.from_numpy(matrix), torch.from_numpy(labels))
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            data,
            batch_size=min(batch_size, len(data)),
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        model = self._build_model().to(self.device)
        diffusion = self._build_diffusion(model)
        diffusion.train()
        ema_model = deepcopy(model).to(self.device)
        ema_model.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            diffusion.parameters(), lr=lr, weight_decay=weight_decay
        )

        history: list[dict[str, float | int]] = []
        completed = 0
        while completed < steps:
            for batch, target in loader:
                if completed >= steps:
                    break
                batch = batch.to(self.device)
                target = target.long().to(self.device)
                optimizer.zero_grad(set_to_none=True)
                multinomial_loss, gaussian_loss = diffusion.mixed_loss(
                    batch, {'y': target}
                )
                loss = multinomial_loss + gaussian_loss
                loss.backward()
                optimizer.step()

                completed += 1
                current_lr = lr * (1.0 - (completed - 1) / steps)
                for group in optimizer.param_groups:
                    group['lr'] = current_lr
                with torch.no_grad():
                    for ema_parameter, parameter in zip(
                        ema_model.parameters(), model.parameters()
                    ):
                        ema_parameter.mul_(ema_decay).add_(
                            parameter.detach(), alpha=1.0 - ema_decay
                        )

                if completed % log_every == 0 or completed == steps:
                    record = {
                        'step': completed,
                        'multinomial_loss': float(multinomial_loss.detach().cpu()),
                        'gaussian_loss': float(gaussian_loss.detach().cpu()),
                        'loss': float(loss.detach().cpu()),
                        'lr': current_lr,
                    }
                    history.append(record)
                    print(
                        f'Step {completed}/{steps} '
                        f'loss={record["loss"]:.6f} '
                        f'(multinomial={record["multinomial_loss"]:.6f}, '
                        f'gaussian={record["gaussian_loss"]:.6f})'
                    )

        self.raw_state_dict = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        self.ema_state_dict = {
            key: value.detach().cpu() for key, value in ema_model.state_dict().items()
        }
        return pd.DataFrame(history)

    def save(self, checkpoint_path: Path, preprocessor_path: Path) -> None:
        """Save model tensors separately from the sklearn preprocessor."""
        if self.raw_state_dict is None or self.ema_state_dict is None:
            raise RuntimeError('Cannot save TAB-DDPM before fitting it.')
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'format_version': self.CHECKPOINT_VERSION,
            'hidden_dims': list(self.hidden_dims),
            'dropout': self.dropout,
            'time_embedding_dim': self.time_embedding_dim,
            'num_timesteps': self.num_timesteps,
            'scheduler': self.scheduler,
            'class_probabilities': self.class_probabilities,
            'raw_state_dict': self.raw_state_dict,
            'ema_state_dict': self.ema_state_dict,
        }
        torch.save(payload, checkpoint_path)
        joblib.dump(self.preprocessor, preprocessor_path)

    @classmethod
    def load(
        cls,
        checkpoint_path: Path,
        preprocessor_path: Path,
        *,
        device: str = 'auto',
    ) -> 'TabDDPMSynthesizer':
        """Load a trusted local checkpoint and its preprocessing transforms."""
        try:
            payload = torch.load(
                checkpoint_path, map_location='cpu', weights_only=True
            )
        except TypeError:
            payload = torch.load(checkpoint_path, map_location='cpu')
        if payload.get('format_version') != cls.CHECKPOINT_VERSION:
            raise ValueError(f'Unsupported TAB-DDPM checkpoint: {checkpoint_path}')
        preprocessor = joblib.load(preprocessor_path)
        if not isinstance(preprocessor, TabDDPMPreprocessor):
            raise TypeError(f'Unexpected preprocessor in {preprocessor_path}')
        synthesizer = cls(
            preprocessor,
            hidden_dims=tuple(payload['hidden_dims']),
            dropout=payload['dropout'],
            time_embedding_dim=payload['time_embedding_dim'],
            num_timesteps=payload['num_timesteps'],
            scheduler=payload['scheduler'],
            device=device,
        )
        synthesizer.class_probabilities = list(payload['class_probabilities'])
        synthesizer.raw_state_dict = payload['raw_state_dict']
        synthesizer.ema_state_dict = payload['ema_state_dict']
        return synthesizer

    def sample(
        self,
        num_rows: int,
        *,
        batch_size: int,
        seed: int,
        checkpoint_variant: str = 'ema',
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Generate rows and restore them to the encoded dataframe schema."""
        if num_rows < 1 or batch_size < 1:
            raise ValueError('num_rows and sample batch size must be positive.')
        if checkpoint_variant not in {'ema', 'raw'}:
            raise ValueError('checkpoint_variant must be "ema" or "raw".')
        state = (
            self.ema_state_dict if checkpoint_variant == 'ema'
            else self.raw_state_dict
        )
        if state is None:
            raise RuntimeError('TAB-DDPM has no fitted checkpoint to sample.')

        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        model = self._build_model().to(self.device)
        model.load_state_dict(state)
        diffusion = self._build_diffusion(model)
        diffusion.eval()
        distribution = torch.tensor(self.class_probabilities, dtype=torch.float32)
        effective_batch = min(batch_size, num_rows)
        output = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with output:
            matrix, labels = diffusion.sample_all(
                num_rows, effective_batch, distribution, ddim=False
            )
        return self.preprocessor.inverse_transform(matrix.numpy(), labels.numpy())
