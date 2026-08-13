"""Train a Gaussian Copula synthesizer and immediately sample it.

The runner intentionally performs one workflow only: prepare a train/test split,
fit SDV's Gaussian Copula model, and write consecutive synthetic samples. The
fitted pickle is not retained because this statistical model is inexpensive to
refit from the stored training split and configuration.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import sdv
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from synthesizer_workflow import (
    WorkflowHelpFormatter,
    decode_categoricals,
    prepare_data,
    save_json,
    set_seed,
    write_training_artifacts,
)


TABLE_NAME = 'table'
DISTRIBUTIONS = {
    'beta',
    'gamma',
    'gaussian_kde',
    'norm',
    'truncnorm',
    'uniform',
}
ARGPARSE_FORMATTER = WorkflowHelpFormatter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=ARGPARSE_FORMATTER,
    )
    inputs = parser.add_argument_group(
        'dataset inputs',
        'Required dataset/output locations and optional exact-file overrides.',
    )
    inputs.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help=(
            'Required. Dataset directory used to infer data.csv, info.json, '
            'and utility_feature.json when exact overrides are omitted.'
        ),
    )
    inputs.add_argument(
        '--data-file',
        type=Path,
        default=None,
        help='Optional. Exact input CSV; defaults to <data-dir>/data.csv.',
    )
    inputs.add_argument(
        '--train-file',
        type=Path,
        default=None,
        help=(
            'Optional. Exact frozen training CSV. Must be supplied with '
            '--test-file; bypasses random splitting.'
        ),
    )
    inputs.add_argument(
        '--test-file',
        type=Path,
        default=None,
        help=(
            'Optional. Exact frozen test CSV. Must be supplied with '
            '--train-file; bypasses random splitting.'
        ),
    )
    inputs.add_argument(
        '--info-file',
        type=Path,
        default=None,
        help=(
            'Optional. Exact required dataset information JSON; defaults to '
            '<data-dir>/info.json.'
        ),
    )
    inputs.add_argument(
        '--utility-feature-file',
        type=Path,
        default=None,
        help=(
            'Optional. Exact required utility configuration JSON; defaults to '
            '<data-dir>/utility_feature.json.'
        ),
    )
    inputs.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Required. Experiment directory receiving splits and generated artifacts.',
    )

    data = parser.add_argument_group(
        'data preparation',
        'Controls matching the CTGAN/TVAE split workflow.',
    )
    data.add_argument(
        '--test-size',
        type=float,
        default=0.3,
        help='Fraction of cleaned rows reserved for test.csv.',
    )
    data.add_argument(
        '--seed',
        type=int,
        default=42,
        help=(
            'Seed for row limiting, train/test splitting, and the Gaussian '
            'Copula sampling stream.'
        ),
    )
    data.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Randomly limit cleaned data before splitting; defaults to all rows.',
    )
    data.add_argument(
        '--no-stratify',
        action='store_true',
        help='Disable target-stratified splitting.',
    )
    data.add_argument(
        '--comment',
        type=str,
        default='',
        help='Free-form note saved in train_config.json.',
    )

    model = parser.add_argument_group(
        'Gaussian Copula model',
        'Minimal marginal-distribution controls. Rounding and min/max are enforced.',
    )
    model.add_argument(
        '--default-distribution',
        choices=sorted(DISTRIBUTIONS),
        default='beta',
        help='Default marginal distribution used by SDV.',
    )
    model.add_argument(
        '--numerical-distributions-file',
        type=Path,
        default=None,
        help=(
            'Optional JSON object mapping numerical column names to marginal '
            'distribution names.'
        ),
    )

    sampling = parser.add_argument_group(
        'sampling',
        'Controls for consecutive deterministic synthetic files.',
    )
    sampling.add_argument(
        '--num-files',
        type=int,
        default=3,
        help='Number of synthetic_N.csv files to generate.',
    )
    sampling.add_argument(
        '--num-rows',
        type=int,
        default=None,
        help='Rows per synthetic file; defaults to the training row count.',
    )
    sampling.add_argument(
        '--sample-batch-size',
        type=int,
        default=None,
        help='Optional rows sampled per SDV batch.',
    )
    sampling.add_argument(
        '--sample-seed',
        type=int,
        default=None,
        help='Optional. Seed for the first synthetic file; defaults to --seed.',
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _jsonable(value: Any) -> Any:
    """Convert NumPy/Path-rich SDV results to stable JSON primitives."""
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return value.item()
    return value


def _load_info_types(
    info_path: Path | None,
    columns: list[str],
) -> tuple[list[str], list[str]]:
    """Load a complete categorical/numerical split from info.json."""
    if info_path is None:
        raise FileNotFoundError(
            'Gaussian Copula requires info.json (or --info-file) so column '
            'types are explicit.'
        )

    info = json.loads(info_path.read_text(encoding='utf-8'))
    col_types = info.get('col_types')
    if not isinstance(col_types, dict):
        raise ValueError(f'{info_path} must contain an object named "col_types".')

    missing = [column for column in columns if column not in col_types]
    extra = [column for column in col_types if column not in columns]
    if missing or extra:
        raise ValueError(
            f'{info_path} column types do not match data columns. '
            f'Missing: {missing}; extra: {extra}.'
        )

    invalid = {
        column: metadata.get('type') if isinstance(metadata, dict) else None
        for column, metadata in col_types.items()
        if not isinstance(metadata, dict) or metadata.get('type') not in {'cat', 'num'}
    }
    if invalid:
        raise ValueError(
            f'{info_path} contains unsupported column types: {invalid}. '
            'Expected every type to be "cat" or "num".'
        )

    categorical = [column for column in columns if col_types[column]['type'] == 'cat']
    numerical = [column for column in columns if col_types[column]['type'] == 'num']
    return categorical, numerical


def _build_sdv_metadata(
    train: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> Metadata:
    """Build metadata without SDV type or primary-key inference."""
    categorical = set(categorical_columns)
    numerical = set(numerical_columns)
    expected = set(train.columns)
    if categorical & numerical or categorical | numerical != expected:
        raise ValueError('Categorical and numerical columns must partition the table.')

    metadata_dict = {
        'tables': {
            TABLE_NAME: {
                'columns': {
                    column: {
                        'sdtype': 'categorical' if column in categorical else 'numerical'
                    }
                    for column in train.columns
                }
            }
        },
        'relationships': [],
        'METADATA_SPEC_VERSION': 'V1',
    }
    metadata = Metadata.load_from_dict(metadata_dict)
    metadata.validate()
    metadata.validate_table(train, table_name=TABLE_NAME)
    return metadata


def _load_numerical_distributions(
    path: Path | None,
    numerical_columns: list[str],
) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f'Numerical distributions file not found: {path}')

    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path} must contain a JSON object.')

    unknown_columns = sorted(set(value) - set(numerical_columns))
    invalid_distributions = {
        column: distribution
        for column, distribution in value.items()
        if distribution not in DISTRIBUTIONS
    }
    if unknown_columns:
        raise ValueError(
            'Numerical distribution overrides reference non-numerical or '
            f'unknown columns: {unknown_columns}'
        )
    if invalid_distributions:
        raise ValueError(
            f'Unsupported numerical distribution overrides: {invalid_distributions}'
        )
    return {str(column): str(distribution) for column, distribution in value.items()}


def _seed_sampling(synthesizer: GaussianCopulaSynthesizer, seed: int) -> None:
    """Seed SDV 1.32's protected sampling hook, failing on incompatible APIs."""
    setter = getattr(synthesizer, '_set_random_state', None)
    if not callable(setter):
        raise RuntimeError(
            f'SDV {sdv.__version__} does not expose the expected protected '
            '_set_random_state method. Pin a supported version or update the runner.'
        )
    setter(seed)


def _write_samples(
    synthesizer: GaussianCopulaSynthesizer,
    output_dir: Path,
    *,
    num_files: int,
    num_rows: int,
    batch_size: int | None,
    seed: int = 42,
) -> list[Path]:
    synthetic_dir = output_dir / 'synthetic'
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    for existing_path in synthetic_dir.glob('synthetic_*.csv'):
        existing_path.unlink()

    paths = []
    for index in range(num_files):
        _seed_sampling(synthesizer, seed + index)
        synthetic = synthesizer.sample(
            num_rows=num_rows,
            batch_size=batch_size,
        )
        path = synthetic_dir / f'synthetic_{index}.csv'
        synthetic.to_csv(path, index=False)
        paths.append(path)
    return paths


def run(args: argparse.Namespace) -> list[Path]:
    if not 0 < args.test_size < 1:
        raise ValueError('--test-size must be between 0 and 1.')
    if args.num_files < 1:
        raise ValueError('--num-files must be at least 1.')
    if args.num_rows is not None and args.num_rows < 1:
        raise ValueError('--num-rows must be at least 1.')
    if args.sample_batch_size is not None and args.sample_batch_size < 1:
        raise ValueError('--sample-batch-size must be at least 1.')

    started = time.perf_counter()
    set_seed(args.seed)
    prepared = prepare_data(
        data_dir=args.data_dir,
        data_file=args.data_file,
        train_file=args.train_file,
        test_file=args.test_file,
        info_file=args.info_file,
        utility_feature_file=args.utility_feature_file,
        categorical_columns_override=None,
        test_size=args.test_size,
        seed=args.seed,
        max_rows=args.max_rows,
        no_stratify=args.no_stratify,
    )
    categorical_columns, numerical_columns = _load_info_types(
        prepared.info_path,
        list(prepared.train.columns),
    )
    if categorical_columns != prepared.categorical_columns:
        raise ValueError(
            'Resolved categorical columns differ from the split in info.json: '
            f'{prepared.categorical_columns} != {categorical_columns}'
        )
    if numerical_columns != prepared.numerical_columns:
        raise ValueError(
            'Resolved numerical columns differ from the split in info.json: '
            f'{prepared.numerical_columns} != {numerical_columns}'
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_metadata = write_training_artifacts(
        args.output_dir,
        prepared,
        write_label_maps=False,
    )
    train = decode_categoricals(prepared.train, prepared.label_maps)
    if train.isna().any().any():
        raise ValueError('Training data still contains missing values after preprocessing.')

    metadata = _build_sdv_metadata(train, categorical_columns, numerical_columns)
    metadata.save_to_json(
        args.output_dir / 'sdv_metadata.json',
        mode='overwrite',
    )
    numerical_distributions = _load_numerical_distributions(
        args.numerical_distributions_file,
        numerical_columns,
    )
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        numerical_distributions=numerical_distributions,
        default_distribution=args.default_distribution,
    )

    fit_started = time.perf_counter()
    synthesizer.fit(train)
    fit_seconds = time.perf_counter() - fit_started
    save_json(
        args.output_dir / 'model_parameters.json',
        _jsonable(synthesizer.get_parameters()),
    )
    save_json(
        args.output_dir / 'learned_distributions.json',
        _jsonable(synthesizer.get_learned_distributions()),
    )

    sample_rows = len(train) if args.num_rows is None else args.num_rows
    sample_seed = args.seed if args.sample_seed is None else args.sample_seed
    sample_started = time.perf_counter()
    paths = _write_samples(
        synthesizer,
        args.output_dir,
        num_files=args.num_files,
        num_rows=sample_rows,
        batch_size=args.sample_batch_size,
        seed=sample_seed,
    )
    sample_seconds = time.perf_counter() - sample_started

    save_json(
        args.output_dir / 'train_config.json',
        _jsonable({
            'args': vars(args),
            'resolved': {
                'data_file': prepared.data_path,
                'train_file': prepared.train_path,
                'test_file': prepared.test_path,
                'info_file': prepared.info_path,
                'utility_feature_file': prepared.utility_feature_path,
                'metadata': experiment_metadata,
                'sdv_version': sdv.__version__,
                'missing_value_policy': 'drop_rows',
                'column_types_source': 'info.json',
                'fit_seconds': fit_seconds,
                'total_seconds': time.perf_counter() - started,
            },
        }),
    )
    save_json(
        args.output_dir / 'synthetic' / 'sample_config.json',
        _jsonable({
            'num_files': args.num_files,
            'num_rows': sample_rows,
            'sample_batch_size': args.sample_batch_size,
            'seed': sample_seed,
            'seeds': [sample_seed + index for index in range(args.num_files)],
            'sampling_seed_method': '_set_random_state',
            'sampling_mode': 'independent_per_file',
            'sample_seconds': sample_seconds,
            'synthetic_files': paths,
        }),
    )

    print(f'Fit seconds: {fit_seconds:.3f}')
    print(f'Sampling seconds: {sample_seconds:.3f}')
    for path in paths:
        print(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == '__main__':
    main()
