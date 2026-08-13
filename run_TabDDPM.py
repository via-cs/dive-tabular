"""Train and sample an unconstrained TAB-DDPM experiment.

When no command is supplied, ``train-sample`` is used. Column types default to
the complete categorical/numerical partition in ``info.json``; explicit column
overrides are available for deliberate experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from synthesizer_workflow import (
    WorkflowHelpFormatter,
    args_payload,
    encode_categoricals,
    load_label_maps,
    load_metadata,
    parse_device,
    parse_dims,
    prepare_data,
    save_json,
    set_seed,
    write_synthetic_samples,
    write_training_artifacts,
)
from synthetizers.TabDDPM.tabddpm import (
    TabDDPMPreprocessor,
    TabDDPMSynthesizer,
    infer_task_type,
    load_info_column_types,
)


COMMANDS = {'train', 'sample', 'train-sample'}
ARGPARSE_FORMATTER = WorkflowHelpFormatter
CHECKPOINT_NAME = 'tabddpm.pt'
PREPROCESSOR_NAME = 'tabddpm_preprocessor.pkl'


def _parse_columns(value: str | None, option: str) -> list[str] | None:
    if value is None:
        return None
    if value.strip().lower() in {'', 'none', 'null'}:
        return []
    columns = [column.strip() for column in value.split(',') if column.strip()]
    if len(columns) != len(set(columns)):
        raise ValueError(f'{option} contains duplicate column names.')
    return columns


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
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
            'Optional. Exact required column-type JSON; defaults to '
            '<data-dir>/info.json.'
        ),
    )
    inputs.add_argument(
        '--utility-feature-file',
        type=Path,
        default=None,
        help=(
            'Optional. Exact required target/utility JSON; defaults to '
            '<data-dir>/utility_feature.json.'
        ),
    )
    inputs.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=(
            'Required. Experiment directory receiving splits, transforms, '
            'checkpoint, losses, configuration, and optional samples.'
        ),
    )

    data = parser.add_argument_group(
        'data preparation',
        'Defaults to info.json types; overrides must partition all non-target columns.',
    )
    data.add_argument(
        '--categorical-columns',
        type=str,
        default=None,
        help=(
            'Optional. Comma-separated categorical feature override. Use "none" '
            'for no categorical features; defaults to info.json.'
        ),
    )
    data.add_argument(
        '--numerical-columns',
        type=str,
        default=None,
        help=(
            'Optional. Comma-separated numerical feature override. Use "none" '
            'for no numerical features; defaults to info.json.'
        ),
    )
    data.add_argument(
        '--test-size',
        type=float,
        default=0.3,
        help='Optional. Fraction of cleaned rows reserved for test.csv.',
    )
    data.add_argument(
        '--seed',
        type=int,
        default=42,
        help=(
            'Optional. Seed for row limiting, splitting, training, and the '
            'first synthetic file; file i uses seed + i.'
        ),
    )
    data.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Optional. Randomly limit cleaned data before splitting.',
    )
    data.add_argument(
        '--no-stratify',
        action='store_true',
        help='Optional. Disable target-stratified splitting for classification.',
    )
    data.add_argument(
        '--normalization',
        choices=['quantile', 'standard', 'minmax'],
        default='quantile',
        help='Optional. Train-fitted transformation for numerical variables.',
    )
    data.add_argument(
        '--comment',
        type=str,
        default='',
        help='Optional. Free-form note saved in train_config.json.',
    )

    model = parser.add_argument_group(
        'TAB-DDPM model',
        'MLP denoiser and Gaussian/multinomial diffusion parameters.',
    )
    model.add_argument(
        '--steps',
        type=int,
        default=30000,
        help='Optional. Number of optimizer updates (not epochs).',
    )
    model.add_argument(
        '--batch-size',
        type=int,
        default=4096,
        help='Optional. Maximum training rows per optimizer update.',
    )
    model.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Optional. Initial AdamW learning rate with linear decay.',
    )
    model.add_argument(
        '--weight-decay',
        type=float,
        default=0.0,
        help='Optional. AdamW weight decay.',
    )
    model.add_argument(
        '--hidden-dims',
        type=parse_dims,
        default=(256, 256),
        help='Optional. Comma-separated MLP widths, for example 256,512,256.',
    )
    model.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Optional. Dropout applied in every MLP hidden block.',
    )
    model.add_argument(
        '--time-embedding-dim',
        type=int,
        default=128,
        help='Optional. Sinusoidal timestep/label embedding width.',
    )
    model.add_argument(
        '--num-timesteps',
        type=int,
        default=1000,
        help='Optional. Number of forward and reverse diffusion steps.',
    )
    model.add_argument(
        '--scheduler',
        choices=['cosine', 'linear'],
        default='cosine',
        help='Optional. Diffusion beta schedule.',
    )
    model.add_argument(
        '--ema-decay',
        type=float,
        default=0.999,
        help='Optional. Exponential moving-average decay for sampling weights.',
    )
    model.add_argument(
        '--log-every',
        type=int,
        default=100,
        help='Optional. Optimizer-step interval written to train_loss.csv.',
    )
    model.add_argument(
        '--device',
        type=parse_device,
        default='auto',
        help=(
            'Optional. Torch device: auto, cpu, cuda, or cuda:N. Auto uses '
            'CUDA when available and otherwise CPU.'
        ),
    )


def _add_sampling_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_seed: bool,
) -> None:
    sampling = parser.add_argument_group(
        'sampling',
        'Controls for reverse diffusion and generated CSV files.',
    )
    sampling.add_argument(
        '--num-files',
        type=int,
        default=1,
        help='Optional. Number of synthetic_N.csv files to generate.',
    )
    sampling.add_argument(
        '--num-rows',
        type=int,
        default=None,
        help='Optional. Rows per file; defaults to the training row count.',
    )
    sampling.add_argument(
        '--sample-batch-size',
        type=int,
        default=10000,
        help='Optional. Maximum rows generated in one reverse-diffusion batch.',
    )
    sampling.add_argument(
        '--checkpoint-variant',
        choices=['ema', 'raw'],
        default='ema',
        help='Optional. Exponential-moving-average or raw training weights.',
    )
    if include_seed:
        sampling.add_argument(
            '--seed',
            type=int,
            default=42,
            help='Optional. Sampling seed for the first file; file i uses seed + i.',
        )
        sampling.add_argument(
            '--device',
            type=parse_device,
            default='auto',
            help='Optional. Torch device used for reverse diffusion.',
        )
    sampling.add_argument(
        '--sample-verbose',
        action='store_true',
        help='Optional. Display every reverse-diffusion timestep.',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=ARGPARSE_FORMATTER
    )
    subparsers = parser.add_subparsers(dest='command', metavar='COMMAND')

    train_sample = subparsers.add_parser(
        'train-sample',
        help='Train TAB-DDPM and sample it (default).',
        description='Train a new experiment and immediately generate CSV samples.',
        formatter_class=ARGPARSE_FORMATTER,
    )
    _add_training_arguments(train_sample)
    _add_sampling_arguments(train_sample, include_seed=False)

    train = subparsers.add_parser(
        'train',
        help='Train and save a TAB-DDPM checkpoint.',
        description='Train a new experiment without sampling.',
        formatter_class=ARGPARSE_FORMATTER,
    )
    _add_training_arguments(train)

    sample = subparsers.add_parser(
        'sample',
        help='Sample an existing TAB-DDPM experiment.',
        description='Load a checkpoint and replace its numbered synthetic CSV set.',
        formatter_class=ARGPARSE_FORMATTER,
    )
    inputs = sample.add_argument_group(
        'experiment inputs',
        'Required experiment location and optional exact-file overrides.',
    )
    inputs.add_argument(
        '--experiment-dir',
        type=Path,
        required=True,
        help='Required. Existing experiment directory and synthetic output root.',
    )
    inputs.add_argument(
        '--checkpoint-file',
        type=Path,
        default=None,
        help=f'Optional. Exact checkpoint; defaults to <experiment-dir>/{CHECKPOINT_NAME}.',
    )
    inputs.add_argument(
        '--preprocessor-file',
        type=Path,
        default=None,
        help=(
            f'Optional. Exact fitted preprocessor; defaults to '
            f'<experiment-dir>/{PREPROCESSOR_NAME}.'
        ),
    )
    inputs.add_argument(
        '--train-file',
        type=Path,
        default=None,
        help='Optional. Exact training CSV; defaults to <experiment-dir>/train.csv.',
    )
    inputs.add_argument(
        '--metadata-file',
        type=Path,
        default=None,
        help='Optional. Exact metadata JSON; defaults to <experiment-dir>/metadata.json.',
    )
    inputs.add_argument(
        '--label-map-file',
        type=Path,
        default=None,
        help='Optional. Exact label maps; defaults to <experiment-dir>/label_maps.json.',
    )
    _add_sampling_arguments(sample, include_seed=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or (
        arguments[0] not in COMMANDS and arguments[0] not in {'-h', '--help'}
    ):
        arguments.insert(0, 'train-sample')
    return build_parser().parse_args(arguments)


def _read_target(utility_path: Path, columns: list[str]) -> str:
    if not utility_path.exists():
        raise FileNotFoundError(f'TAB-DDPM requires utility_feature.json: {utility_path}')
    utility = json.loads(utility_path.read_text(encoding='utf-8'))
    target = utility.get('target_column') or utility.get('target')
    if not isinstance(target, str) or target not in columns:
        raise ValueError(f'{utility_path} must name a target column present in data.csv.')
    return target


def _resolve_types(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, str, list[str], list[str], dict[str, Any]]:
    data_path = args.data_file or args.train_file or args.data_dir / 'data.csv'
    info_path = args.info_file or args.data_dir / 'info.json'
    utility_path = args.utility_feature_file or args.data_dir / 'utility_feature.json'
    if not data_path.exists():
        raise FileNotFoundError(f'Data file not found: {data_path}')
    columns = pd.read_csv(data_path, nrows=0).columns.tolist()
    info_categorical, info_numerical, info = load_info_column_types(
        info_path, columns
    )
    target = _read_target(utility_path, columns)
    features = [column for column in columns if column != target]
    categorical_override = _parse_columns(
        args.categorical_columns, '--categorical-columns'
    )
    numerical_override = _parse_columns(args.numerical_columns, '--numerical-columns')
    if categorical_override is None and numerical_override is None:
        feature_categorical = [column for column in info_categorical if column != target]
        feature_numerical = [column for column in info_numerical if column != target]
    elif categorical_override is None:
        feature_numerical = numerical_override
        feature_categorical = [column for column in features if column not in feature_numerical]
    elif numerical_override is None:
        feature_categorical = categorical_override
        feature_numerical = [column for column in features if column not in feature_categorical]
    else:
        feature_categorical = categorical_override
        feature_numerical = numerical_override

    unknown = sorted((set(feature_categorical) | set(feature_numerical)) - set(features))
    overlap = sorted(set(feature_categorical) & set(feature_numerical))
    unassigned = [
        column for column in features
        if column not in set(feature_categorical) | set(feature_numerical)
    ]
    if unknown or overlap or unassigned:
        raise ValueError(
            'Column overrides must partition every non-target column. '
            f'Unknown: {unknown}; overlap: {overlap}; unassigned: {unassigned}.'
        )
    target_is_categorical = target in info_categorical
    if info.get('task') is None and info.get('task_type') is None:
        full_target = pd.read_csv(data_path, usecols=[target])[target]
        info = dict(info)
        info['task'] = infer_task_type(
            full_target,
            target_is_categorical,
            None,
        )
    categorical = [
        column for column in columns
        if column in feature_categorical or (column == target and target_is_categorical)
    ]
    numerical = [column for column in columns if column not in categorical]
    return data_path, info_path, utility_path, target, categorical, numerical, info


def _validate_training_args(args: argparse.Namespace) -> None:
    if not 0 < args.test_size < 1:
        raise ValueError('--test-size must be between 0 and 1.')
    positive = {
        '--steps': args.steps,
        '--batch-size': args.batch_size,
        '--time-embedding-dim': args.time_embedding_dim,
        '--num-timesteps': args.num_timesteps,
        '--log-every': args.log_every,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f'These arguments must be positive: {invalid}')
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError('Learning rate must be positive and weight decay nonnegative.')
    if not 0 <= args.dropout < 1 or not 0 <= args.ema_decay < 1:
        raise ValueError('Dropout and EMA decay must be in [0, 1).')


def _create_synthesizer(
    args: argparse.Namespace,
    prepared,
    categorical: list[str],
    numerical: list[str],
    info: dict[str, Any],
) -> TabDDPMSynthesizer:
    task = infer_task_type(
        prepared.train[prepared.target],
        prepared.target in categorical,
        info.get('task') or info.get('task_type'),
    )
    preprocessor = TabDDPMPreprocessor(
        column_order=list(prepared.train.columns),
        target=prepared.target,
        categorical_columns=categorical,
        numerical_columns=numerical,
        task_type=task,
        normalization=args.normalization,
        seed=args.seed,
    )
    return TabDDPMSynthesizer(
        preprocessor,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        time_embedding_dim=args.time_embedding_dim,
        num_timesteps=args.num_timesteps,
        scheduler=args.scheduler,
        device=args.device,
    )


def _sample_adapter(
    model: TabDDPMSynthesizer,
    *,
    batch_size: int,
    checkpoint_variant: str,
):
    def sample(num_rows: int, seed: int, verbose: bool):
        return model.sample(
            num_rows,
            batch_size=batch_size,
            seed=seed,
            checkpoint_variant=checkpoint_variant,
            verbose=verbose,
        )

    return sample


def _write_samples(
    args: argparse.Namespace,
    model: TabDDPMSynthesizer,
    train: pd.DataFrame,
    categorical_columns: list[str],
    label_maps: dict,
    output_dir: Path,
) -> list[Path]:
    return write_synthetic_samples(
        output_dir=output_dir,
        sample=_sample_adapter(
            model,
            batch_size=args.sample_batch_size,
            checkpoint_variant=args.checkpoint_variant,
        ),
        train=train,
        categorical_columns=categorical_columns,
        label_maps=label_maps,
        num_files=args.num_files,
        num_rows=args.num_rows,
        seed=args.seed,
        verbose=args.sample_verbose,
    )


def run_training(args: argparse.Namespace, *, sample_after_training: bool) -> None:
    _validate_training_args(args)
    (
        data_path,
        info_path,
        utility_path,
        _,
        categorical,
        numerical,
        info,
    ) = _resolve_types(args)
    set_seed(args.seed)
    prepared = prepare_data(
        data_dir=args.data_dir,
        data_file=data_path,
        train_file=args.train_file,
        test_file=args.test_file,
        info_file=info_path,
        utility_feature_file=utility_path,
        categorical_columns_override=','.join(categorical) if categorical else 'none',
        test_size=args.test_size,
        seed=args.seed,
        max_rows=args.max_rows,
        no_stratify=args.no_stratify,
    )
    metadata = write_training_artifacts(args.output_dir, prepared)
    model = _create_synthesizer(args, prepared, categorical, numerical, info)
    training_started = time.perf_counter()
    loss = model.fit(
        prepared.train,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        seed=args.seed,
        log_every=args.log_every,
    )
    training_seconds = time.perf_counter() - training_started
    print(f'Training seconds: {training_seconds:.3f}', flush=True)
    checkpoint_path = args.output_dir / CHECKPOINT_NAME
    preprocessor_path = args.output_dir / PREPROCESSOR_NAME
    model.save(checkpoint_path, preprocessor_path)
    loss.to_csv(args.output_dir / 'train_loss.csv', index=False)
    save_json(
        args.output_dir / 'train_config.json',
        args_payload(
            args,
            data_file=data_path,
            info_file=info_path,
            train_file=prepared.train_path,
            test_file=prepared.test_path,
            utility_feature_file=utility_path,
            metadata=metadata,
            task_type=model.preprocessor.task_type,
            categorical_columns=categorical,
            numerical_columns=numerical,
            checkpoint_file=checkpoint_path,
            preprocessor_file=preprocessor_path,
            training_seconds=training_seconds,
        ),
    )
    print(checkpoint_path)

    if sample_after_training:
        sampling_started = time.perf_counter()
        paths = _write_samples(
            args,
            model,
            prepared.train,
            categorical,
            prepared.label_maps,
            args.output_dir,
        )
        sampling_seconds = time.perf_counter() - sampling_started
        print(f'Sampling seconds: {sampling_seconds:.3f}', flush=True)
        save_json(
            args.output_dir / 'synthetic' / 'sample_config.json',
            args_payload(args, synthetic_files=paths),
        )
        for path in paths:
            print(path)


def run_sampling(args: argparse.Namespace) -> None:
    if args.num_files < 1 or args.sample_batch_size < 1:
        raise ValueError('--num-files and --sample-batch-size must be positive.')
    experiment_dir = args.experiment_dir
    checkpoint_path = args.checkpoint_file or experiment_dir / CHECKPOINT_NAME
    preprocessor_path = args.preprocessor_file or experiment_dir / PREPROCESSOR_NAME
    train_path = args.train_file or experiment_dir / 'train.csv'
    metadata_path = args.metadata_file or experiment_dir / 'metadata.json'
    label_map_path = args.label_map_file or experiment_dir / 'label_maps.json'
    for label, path in (
        ('TAB-DDPM checkpoint', checkpoint_path),
        ('TAB-DDPM preprocessor', preprocessor_path),
        ('training split', train_path),
        ('experiment metadata', metadata_path),
        ('categorical label maps', label_map_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f'{label} not found: {path}')

    train_raw = pd.read_csv(train_path)
    metadata = load_metadata(metadata_path, list(train_raw.columns))
    label_maps = load_label_maps(label_map_path)
    train, _ = encode_categoricals(
        train_raw, metadata['categorical_columns'], label_maps=label_maps
    )
    set_seed(args.seed)
    model = TabDDPMSynthesizer.load(
        checkpoint_path, preprocessor_path, device=args.device
    )
    sampling_started = time.perf_counter()
    paths = _write_samples(
        args,
        model,
        train,
        metadata['categorical_columns'],
        label_maps,
        experiment_dir,
    )
    sampling_seconds = time.perf_counter() - sampling_started
    print(f'Sampling seconds: {sampling_seconds:.3f}', flush=True)
    save_json(
        experiment_dir / 'synthetic' / 'sample_config.json',
        args_payload(
            args,
            checkpoint_file=checkpoint_path,
            preprocessor_file=preprocessor_path,
            train_file=train_path,
            metadata_file=metadata_path,
            label_map_file=label_map_path,
            synthetic_files=paths,
        ),
    )
    for path in paths:
        print(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == 'sample':
        run_sampling(args)
    else:
        run_training(args, sample_after_training=args.command == 'train-sample')


if __name__ == '__main__':
    main()
