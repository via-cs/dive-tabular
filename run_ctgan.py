"""Train and sample an unconstrained CTGAN experiment.

When no command is supplied, train-sample is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from synthesizer_workflow import (
    args_payload,
    encode_categoricals,
    load_label_maps,
    load_metadata,
    parse_device,
    parse_dims,
    prepare_data,
    save_json,
    set_seed,
    WorkflowHelpFormatter,
    write_synthetic_samples,
    write_training_artifacts,
)
from synthetizers.CTGAN.ctgan import CTGAN


COMMANDS = {'train', 'sample', 'train-sample'}
ARGPARSE_FORMATTER = WorkflowHelpFormatter


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_argument_group(
        'dataset inputs',
        'Required dataset/output locations and optional per-file overrides.',
    )
    inputs.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help=(
            'Required. Dataset directory used to infer data.csv, info.json, '
            'and the required utility_feature.json when explicit overrides are omitted.'
        ),
    )
    inputs.add_argument(
        '--data-file',
        type=Path,
        default=None,
        help=(
            'Optional. Exact input CSV. When omitted, inferred as '
            '<data-dir>/data.csv.'
        ),
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
            'Optional. Exact dataset information JSON. When omitted, '
            '<data-dir>/info.json is used if present; otherwise categorical '
            'columns are inferred from non-numeric dataframe dtypes.'
        ),
    )
    inputs.add_argument(
        '--utility-feature-file',
        type=Path,
        default=None,
        help=(
            'Optional. Exact required utility-feature JSON. When omitted, '
            '<data-dir>/utility_feature.json must exist.'
        ),
    )
    inputs.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help=(
            'Required. Experiment directory receiving splits, metadata, '
            'configuration, losses, the CTGAN checkpoint, and optional samples.'
        ),
    )

    data = parser.add_argument_group(
        'data preparation',
        'Optional overrides for automatically resolved dataset behavior.',
    )
    data.add_argument(
        '--categorical-columns',
        type=str,
        default=None,
        help=(
            'Optional. Comma-separated categorical-column override. Use "none" '
            'for no categorical columns. Otherwise inferred from info.json '
            'col_types, then from non-numeric dataframe dtypes.'
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
            'Optional. Seed for row limiting, train/test splitting, model '
            'training, and the first synthetic file; file i uses seed + i.'
        ),
    )
    data.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help=(
            'Optional. Randomly limit the cleaned dataset before splitting. '
            'When omitted, all cleaned rows are used.'
        ),
    )
    data.add_argument(
        '--no-stratify',
        action='store_true',
        help=(
            'Optional. Disable target-stratified splitting. By default, '
            'stratification is used only when the target has multiple values '
            'and every value has at least two rows.'
        ),
    )
    data.add_argument(
        '--comment',
        type=str,
        default='',
        help='Optional. Free-form note saved in train_config.json.',
    )

    model = parser.add_argument_group(
        'CTGAN training',
        'Optional CTGAN hyperparameters; constructor defaults are explicit here.',
    )
    model.add_argument(
        '--epochs',
        type=int,
        default=300,
        help='Optional. Number of complete CTGAN training epochs.',
    )
    model.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help=(
            'Optional. Number of training rows per CTGAN batch. Must be even '
            'and divisible by --pac.'
        ),
    )
    model.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='Optional. Dimension of the random vector passed to the generator.',
    )
    model.add_argument(
        '--generator-dims',
        type=parse_dims,
        default=(256, 256),
        help=(
            'Optional. Comma-separated generator hidden-layer widths, for '
            'example 256,256.'
        ),
    )
    model.add_argument(
        '--discriminator-dims',
        type=parse_dims,
        default=(256, 256),
        help=(
            'Optional. Comma-separated discriminator hidden-layer widths, for '
            'example 256,256.'
        ),
    )
    model.add_argument(
        '--generator-lr',
        type=float,
        default=2e-4,
        help='Optional. Generator optimizer learning rate.',
    )
    model.add_argument(
        '--generator-decay',
        type=float,
        default=1e-6,
        help='Optional. Generator optimizer weight decay.',
    )
    model.add_argument(
        '--discriminator-lr',
        type=float,
        default=2e-4,
        help='Optional. Discriminator optimizer learning rate.',
    )
    model.add_argument(
        '--discriminator-decay',
        type=float,
        default=1e-6,
        help='Optional. Discriminator optimizer weight decay.',
    )
    model.add_argument(
        '--discriminator-steps',
        type=int,
        default=1,
        help='Optional. Discriminator updates performed per generator update.',
    )
    model.add_argument(
        '--pac',
        type=int,
        default=10,
        help=(
            'Optional. Rows packed together for discriminator decisions. '
            '--batch-size must be divisible by this value.'
        ),
    )
    model.add_argument(
        '--optimizer',
        choices=['adam', 'rmsprop', 'sgd'],
        default='adam',
        help='Optional. Optimizer used for both generator and discriminator.',
    )
    model.add_argument(
        '--log-frequency',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Optional. Weight conditional category sampling by observed '
            'log-frequency. Use --no-log-frequency for uniform treatment.'
        ),
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
        'Optional controls for generated CSV files.',
    )
    sampling.add_argument(
        '--num-files',
        type=int,
        default=1,
        help=(
            'Optional. Number of synthetic_N.csv files to generate. Existing '
            'synthetic_*.csv files are replaced before writing the new set.'
        ),
    )
    sampling.add_argument(
        '--num-rows',
        type=int,
        default=None,
        help=(
            'Optional. Rows generated per synthetic CSV. When omitted, inferred '
            'from the number of rows in train.csv.'
        ),
    )
    if include_seed:
        sampling.add_argument(
            '--seed',
            type=int,
            default=42,
            help=(
                'Optional. Seed for the first synthetic file; file i uses '
                'seed + i.'
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=ARGPARSE_FORMATTER,
    )
    subparsers = parser.add_subparsers(dest='command', metavar='COMMAND')

    train_sample = subparsers.add_parser(
        'train-sample',
        help='Train a CTGAN and sample it (default).',
        description=(
            'Train a new CTGAN experiment and immediately generate synthetic '
            'CSVs. This command is selected implicitly when COMMAND is omitted.'
        ),
        formatter_class=ARGPARSE_FORMATTER,
    )
    _add_training_arguments(train_sample)
    _add_sampling_arguments(train_sample, include_seed=False)

    train = subparsers.add_parser(
        'train',
        help='Train and save a CTGAN checkpoint.',
        description='Train a new CTGAN experiment without generating synthetic CSVs.',
        formatter_class=ARGPARSE_FORMATTER,
    )
    _add_training_arguments(train)

    sample = subparsers.add_parser(
        'sample',
        help='Sample an existing CTGAN experiment.',
        description=(
            'Load a previously trained CTGAN experiment and replace its numbered '
            'synthetic CSV set.'
        ),
        formatter_class=ARGPARSE_FORMATTER,
    )
    sample_inputs = sample.add_argument_group(
        'experiment inputs',
        'Required experiment location and optional exact-file overrides.',
    )
    sample_inputs.add_argument(
        '--experiment-dir',
        type=Path,
        required=True,
        help=(
            'Required. Existing experiment directory used to infer the '
            'checkpoint, train split, metadata, label maps, and synthetic output.'
        ),
    )
    sample_inputs.add_argument(
        '--checkpoint-file',
        type=Path,
        default=None,
        help=(
            'Optional. Exact CTGAN checkpoint. When omitted, inferred as '
            '<experiment-dir>/ctgan.pkl.'
        ),
    )
    sample_inputs.add_argument(
        '--train-file',
        type=Path,
        default=None,
        help=(
            'Optional. Exact training split. When omitted, inferred as '
            '<experiment-dir>/train.csv.'
        ),
    )
    sample_inputs.add_argument(
        '--metadata-file',
        type=Path,
        default=None,
        help=(
            'Optional. Exact experiment metadata. When omitted, inferred as '
            '<experiment-dir>/metadata.json.'
        ),
    )
    sample_inputs.add_argument(
        '--label-map-file',
        type=Path,
        default=None,
        help=(
            'Optional. Exact categorical label maps. When omitted, inferred as '
            '<experiment-dir>/label_maps.json.'
        ),
    )
    _add_sampling_arguments(sample, include_seed=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments, inserting the default train-sample command."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or (
        arguments[0] not in COMMANDS and arguments[0] not in {'-h', '--help'}
    ):
        arguments.insert(0, 'train-sample')
    return build_parser().parse_args(arguments)


def _cuda_argument(device: str) -> bool | str:
    if device == 'auto':
        return True
    if device == 'cpu':
        return False
    if device == 'cuda':
        return True
    return device


def _create_model(args: argparse.Namespace, test: pd.DataFrame) -> CTGAN:
    return CTGAN(
        test,
        embedding_dim=args.embedding_dim,
        generator_dim=args.generator_dims,
        discriminator_dim=args.discriminator_dims,
        generator_lr=args.generator_lr,
        generator_decay=args.generator_decay,
        discriminator_lr=args.discriminator_lr,
        discriminator_decay=args.discriminator_decay,
        batch_size=args.batch_size,
        discriminator_steps=args.discriminator_steps,
        log_frequency=args.log_frequency,
        epochs=args.epochs,
        pac=args.pac,
        cuda=_cuda_argument(args.device),
        path=str(args.output_dir),
        version='unconstrained',
        verbose=True,
    )


def _train_model(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_columns: list[str],
) -> CTGAN:
    """Adapt the legacy CTGAN fitting API to an unconstrained runner."""
    model = _create_model(args, test)
    model.set_random_state(args.seed)
    legacy_fit_args = SimpleNamespace(
        version='unconstrained',
        optimiser=args.optimizer,
    )
    model.fit(legacy_fit_args, train, discrete_columns=categorical_columns)
    model.save(str(args.output_dir / 'ctgan.pkl'))
    if model.loss_history:
        pd.DataFrame(model.loss_history).to_csv(
            args.output_dir / 'train_loss.csv', index=False
        )
    return model


def _sample_adapter(model: CTGAN):
    def sample(num_rows: int, seed: int, _verbose: bool):
        model.set_random_state(seed)
        generated = model.sample(num_rows)
        return generated[0] if isinstance(generated, tuple) else generated

    return sample


def _write_samples(
    args: argparse.Namespace,
    model: CTGAN,
    train: pd.DataFrame,
    categorical_columns: list[str],
    label_maps: dict,
    output_dir: Path,
) -> list[Path]:
    return write_synthetic_samples(
        output_dir=output_dir,
        sample=_sample_adapter(model),
        train=train,
        categorical_columns=categorical_columns,
        label_maps=label_maps,
        num_files=args.num_files,
        num_rows=args.num_rows,
        seed=args.seed,
        verbose=False,
    )


def run_training(args: argparse.Namespace, *, sample_after_training: bool) -> None:
    if not 0 < args.test_size < 1:
        raise ValueError('--test-size must be between 0 and 1.')
    if args.epochs < 1:
        raise ValueError('--epochs must be at least 1.')
    if args.batch_size < 2 or args.batch_size % 2:
        raise ValueError('--batch-size must be a positive even number.')
    if args.pac < 1 or args.batch_size % args.pac:
        raise ValueError('--batch-size must be divisible by --pac.')
    if args.discriminator_steps < 1:
        raise ValueError('--discriminator-steps must be at least 1.')

    set_seed(args.seed)
    prepared = prepare_data(
        data_dir=args.data_dir,
        data_file=args.data_file,
        train_file=args.train_file,
        test_file=args.test_file,
        info_file=args.info_file,
        utility_feature_file=args.utility_feature_file,
        categorical_columns_override=args.categorical_columns,
        test_size=args.test_size,
        seed=args.seed,
        max_rows=args.max_rows,
        no_stratify=args.no_stratify,
    )
    metadata = write_training_artifacts(args.output_dir, prepared)
    save_json(
        args.output_dir / 'train_config.json',
        args_payload(
            args,
            data_file=prepared.data_path,
            train_file=prepared.train_path,
            test_file=prepared.test_path,
            info_file=prepared.info_path,
            utility_feature_file=prepared.utility_feature_path,
            metadata=metadata,
        ),
    )
    model = _train_model(
        args,
        prepared.train,
        prepared.test,
        prepared.categorical_columns,
    )
    print(args.output_dir / 'ctgan.pkl')

    if sample_after_training:
        paths = _write_samples(
            args,
            model,
            prepared.train,
            prepared.categorical_columns,
            prepared.label_maps,
            args.output_dir,
        )
        save_json(
            args.output_dir / 'synthetic' / 'sample_config.json',
            args_payload(args, synthetic_files=paths),
        )
        for path in paths:
            print(path)


def run_sampling(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir
    checkpoint_path = args.checkpoint_file or experiment_dir / 'ctgan.pkl'
    train_path = args.train_file or experiment_dir / 'train.csv'
    metadata_path = args.metadata_file or experiment_dir / 'metadata.json'
    label_map_path = args.label_map_file or experiment_dir / 'label_maps.json'

    for label, path in (
        ('CTGAN checkpoint', checkpoint_path),
        ('training split', train_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f'{label} not found: {path}')

    train_raw = pd.read_csv(train_path)
    metadata = load_metadata(metadata_path, list(train_raw.columns))
    label_maps = load_label_maps(label_map_path)
    train, _ = encode_categoricals(
        train_raw,
        metadata['categorical_columns'],
        label_maps=label_maps,
    )

    set_seed(args.seed)
    model = CTGAN.load(str(checkpoint_path))
    model._version = 'unconstrained'
    paths = _write_samples(
        args,
        model,
        train,
        metadata['categorical_columns'],
        label_maps,
        experiment_dir,
    )
    save_json(
        experiment_dir / 'synthetic' / 'sample_config.json',
        args_payload(
            args,
            checkpoint_file=checkpoint_path,
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
