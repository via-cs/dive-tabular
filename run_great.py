"""Train and sample an unconstrained GReaT synthetic-data experiment."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from synthesizer_workflow import (
    WorkflowHelpFormatter,
    args_payload,
    decode_categoricals,
    load_metadata,
    parse_device,
    prepare_data,
    save_json,
    set_seed,
    write_training_artifacts,
)
from synthesizer_workflow.decimal_formatting import format_synthetic_file
from synthesizer_workflow.great import (
    GreatValueCodec,
    format_great_row,
    validate_great_sample,
)


COMMANDS = {'train', 'sample', 'train-sample'}
DEFAULT_LLM = 'tabularisai/Qwen3-0.3B-distil'


def _optional_precision(value: str) -> int | None:
    if value.lower() in {'none', 'full'}:
        return None
    precision = int(value)
    if precision < 0:
        raise argparse.ArgumentTypeError('precision must be non-negative or none.')
    return precision


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_argument_group('dataset inputs')
    inputs.add_argument(
        '--data-dir', type=Path, required=True,
        help='Required. Dataset directory used for default input filenames.',
    )
    inputs.add_argument(
        '--data-file', type=Path, default=None,
        help='Optional. Exact input CSV; defaults to <data-dir>/data.csv.',
    )
    inputs.add_argument(
        '--info-file', type=Path, default=None,
        help='Optional. Exact info JSON; defaults to <data-dir>/info.json.',
    )
    inputs.add_argument(
        '--utility-feature-file', type=Path, default=None,
        help=(
            'Optional. Exact utility-feature JSON; defaults to '
            '<data-dir>/utility_feature.json.'
        ),
    )
    inputs.add_argument(
        '--output-dir', type=Path, required=True,
        help='Required. Experiment directory receiving model and data artifacts.',
    )

    data = parser.add_argument_group('data preparation')
    data.add_argument(
        '--categorical-columns', type=str, default=None,
        help='Optional. Comma-separated override; use none for no categoricals.',
    )
    data.add_argument(
        '--test-size', type=float, default=0.2,
        help='Optional. Fraction of rows reserved for the real test split.',
    )
    data.add_argument(
        '--seed', type=int, default=42,
        help='Optional. Seed for splitting, training, and the first sample file.',
    )
    data.add_argument(
        '--max-rows', type=int, default=None,
        help='Optional. Randomly limit rows before splitting; useful for smoke tests.',
    )
    data.add_argument(
        '--no-stratify', action='store_true',
        help='Optional. Disable target-stratified train/test splitting.',
    )
    data.add_argument(
        '--comment', type=str, default='',
        help='Optional. Free-form note stored in train_config.json.',
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    _add_dataset_arguments(parser)
    model = parser.add_argument_group('GReaT training')
    model.add_argument(
        '--llm', default=DEFAULT_LLM,
        help='Optional. Hugging Face causal-language-model checkpoint.',
    )
    model.add_argument(
        '--epochs', type=int, default=5,
        help='Optional. Number of complete fine-tuning epochs.',
    )
    model.add_argument(
        '--batch-size', type=int, default=32,
        help='Optional. Per-device training batch size.',
    )
    model.add_argument(
        '--gradient-accumulation-steps', type=int, default=1,
        help='Optional. Batches accumulated before each optimizer update.',
    )
    model.add_argument(
        '--learning-rate', type=float, default=5e-5,
        help='Optional. Hugging Face Trainer learning rate.',
    )
    model.add_argument(
        '--float-precision', type=_optional_precision, default=3,
        help='Optional. Float decimal places in model text, or none for full precision.',
    )
    model.add_argument(
        '--bf16', action=argparse.BooleanOptionalAction, default=True,
        help='Optional. Use BF16 mixed-precision training; use --no-bf16 for FP32.',
    )
    model.add_argument(
        '--checkpoint-dtype', choices=['float32', 'bfloat16'], default='bfloat16',
        help='Optional. Dtype used for the permanent model checkpoint.',
    )
    model.add_argument(
        '--dataloader-num-workers', type=int, default=4,
        help='Optional. Worker processes used by the training data loader.',
    )
    model.add_argument(
        '--logging-steps', type=int, default=10,
        help='Optional. Optimizer steps between loss log records.',
    )
    model.add_argument(
        '--device', type=parse_device, default='auto',
        help=(
            'Optional. Sampling device after training. Isolate training GPUs with '
            'CUDA_VISIBLE_DEVICES when running concurrent jobs.'
        ),
    )


def _add_sampling_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_seed: bool,
) -> None:
    sampling = parser.add_argument_group('sampling')
    sampling.add_argument(
        '--num-files', type=int, default=1,
        help='Optional. Number of synthetic_N.csv files to generate.',
    )
    sampling.add_argument(
        '--num-rows', type=int, default=None,
        help='Optional. Rows per file; defaults to the training-split size.',
    )
    if include_seed:
        sampling.add_argument(
            '--seed', type=int, default=42,
            help='Optional. Seed for the first file; file i uses seed + i.',
        )
    sampling.add_argument(
        '--temperature', type=float, default=0.7,
        help='Optional. Language-model generation temperature.',
    )
    sampling.add_argument(
        '--max-length', type=int, default=512,
        help='Optional. Maximum total tokens generated for each candidate row.',
    )
    sampling.add_argument(
        '--sampling-batch-size', type=int, default=64,
        help='Optional. Candidate rows generated together by legacy sampling.',
    )
    sampling.add_argument(
        '--guided-sampling', action=argparse.BooleanOptionalAction, default=False,
        help='Optional. Generate feature-by-feature; much slower for wide datasets.',
    )
    sampling.add_argument(
        '--max-sampling-rounds', type=int, default=8,
        help='Optional. Maximum oversampling rounds used to replace invalid rows.',
    )
    sampling.add_argument(
        '--inference-dtype', choices=['auto', 'float32', 'bfloat16'], default='auto',
        help='Optional. Model dtype used during generation.',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=WorkflowHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', metavar='COMMAND')

    train_sample = subparsers.add_parser(
        'train-sample', help='Train GReaT and immediately sample it (default).',
        formatter_class=WorkflowHelpFormatter,
    )
    _add_training_arguments(train_sample)
    _add_sampling_arguments(train_sample, include_seed=False)

    train = subparsers.add_parser(
        'train', help='Train and save a GReaT checkpoint.',
        formatter_class=WorkflowHelpFormatter,
    )
    _add_training_arguments(train)

    sample = subparsers.add_parser(
        'sample', help='Load and sample an existing GReaT experiment.',
        formatter_class=WorkflowHelpFormatter,
    )
    sample.add_argument(
        '--experiment-dir', type=Path, required=True,
        help='Required. Existing experiment directory.',
    )
    sample.add_argument(
        '--checkpoint-dir', type=Path, default=None,
        help='Optional. Checkpoint directory; defaults to <experiment>/great_model.',
    )
    sample.add_argument(
        '--train-file', type=Path, default=None,
        help='Optional. Training CSV; defaults to <experiment>/train.csv.',
    )
    sample.add_argument(
        '--metadata-file', type=Path, default=None,
        help='Optional. Metadata JSON; defaults to <experiment>/metadata.json.',
    )
    sample.add_argument(
        '--serialization-config-file', type=Path, default=None,
        help=(
            'Optional. Serialization JSON; defaults to '
            '<experiment>/serialization_config.json.'
        ),
    )
    sample.add_argument(
        '--device', type=parse_device, default='auto',
        help='Optional. Generation device: auto, cpu, cuda, or cuda:N.',
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


def _validate_training_args(args: argparse.Namespace) -> None:
    if not 0 < args.test_size < 1:
        raise ValueError('--test-size must be between 0 and 1.')
    for name in (
        'epochs', 'batch_size', 'gradient_accumulation_steps',
        'dataloader_num_workers', 'logging_steps',
    ):
        minimum = 0 if name == 'dataloader_num_workers' else 1
        if getattr(args, name) < minimum:
            raise ValueError(f'--{name.replace("_", "-")} must be at least {minimum}.')
    if args.learning_rate <= 0:
        raise ValueError('--learning-rate must be positive.')


def _token_length_stats(model, data: pd.DataFrame, precision: int | None, seed: int) -> dict:
    measured = data
    if len(measured) > 2000:
        measured = measured.sample(n=2000, random_state=seed)
    texts = [format_great_row(row, precision) for _, row in measured.iterrows()]
    lengths = []
    for start in range(0, len(texts), 128):
        encoded = model.tokenizer(
            texts[start:start + 128], padding=False, truncation=False,
        )['input_ids']
        lengths.extend(len(tokens) for tokens in encoded)
    values = np.asarray(lengths, dtype=int)
    return {
        'measured_rows': int(len(values)),
        'minimum': int(values.min()),
        'median': int(np.median(values)),
        'p95': int(np.quantile(values, 0.95)),
        'p99': int(np.quantile(values, 0.99)),
        'maximum': int(values.max()),
    }


def _save_checkpoint(model, checkpoint_dir: Path, dtype_name: str) -> dict:
    import torch

    dtype = torch.bfloat16 if dtype_name == 'bfloat16' else torch.float32
    model.model.to(dtype=dtype)
    model.save(str(checkpoint_dir))
    model.tokenizer.save_pretrained(checkpoint_dir / 'tokenizer')

    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    state_bytes = sum(
        value.numel() * value.element_size()
        for value in model.model.state_dict().values()
    )
    info = {
        'dtype': dtype_name,
        'parameter_count': int(parameter_count),
        'state_dict_bytes': int(state_bytes),
        'state_dict_gib': state_bytes / 1024**3,
        'checkpoint_dir': str(checkpoint_dir),
    }
    save_json(checkpoint_dir / 'checkpoint_info.json', info)
    return info


def _prepare_model_frames(prepared):
    output_train = decode_categoricals(prepared.train, prepared.label_maps)
    output_test = decode_categoricals(prepared.test, prepared.label_maps)
    codec = GreatValueCodec.fit(output_train, prepared.categorical_columns)
    return (
        codec.transform(output_train),
        codec.transform(output_test),
        output_train,
        output_test,
        codec,
    )


def _sampling_device_and_dtype(model, device: str, inference_dtype: str) -> str:
    import torch

    resolved_device = device
    if resolved_device == 'auto':
        resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_name = inference_dtype
    if dtype_name == 'auto':
        dtype_name = 'bfloat16' if resolved_device.startswith('cuda') else 'float32'
    dtype = torch.bfloat16 if dtype_name == 'bfloat16' else torch.float32
    model.model.to(dtype=dtype)
    return resolved_device


def _merge_diagnostics(total: dict, update: dict) -> None:
    for key, value in update.items():
        if key == 'missing_columns':
            total[key] = sorted(set(total.get(key, [])) | set(value))
        elif isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _write_samples(
    args: argparse.Namespace,
    *,
    model,
    model_train: pd.DataFrame,
    output_train: pd.DataFrame,
    categorical_columns: list[str],
    codec: GreatValueCodec,
    output_dir: Path,
) -> tuple[list[Path], dict]:
    if args.num_files < 1:
        raise ValueError('--num-files must be at least 1.')
    sample_size = len(output_train) if args.num_rows is None else args.num_rows
    if sample_size < 1:
        raise ValueError('--num-rows must be at least 1.')
    if args.sampling_batch_size < 1 or args.max_length < 1:
        raise ValueError('sampling batch size and max length must be positive.')

    device = _sampling_device_and_dtype(
        model, args.device, args.inference_dtype,
    )
    synthetic_dir = output_dir / 'synthetic'
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    for existing in synthetic_dir.glob('synthetic_*.csv'):
        existing.unlink()

    paths = []
    run_diagnostics = {'files': []}
    for file_index in range(args.num_files):
        start_time = time.monotonic()
        chunks = []
        accepted_count = 0
        diagnostics = {'file_index': file_index, 'rounds': 0}
        for round_index in range(args.max_sampling_rounds):
            remaining = sample_size - accepted_count
            if remaining <= 0:
                break
            requested = max(remaining, math.ceil(remaining * 1.25))
            set_seed(args.seed + file_index + round_index * 1_000_000)
            candidate = model.sample(
                n_samples=requested,
                temperature=args.temperature,
                k=args.sampling_batch_size,
                max_length=args.max_length,
                drop_nan=True,
                device=device,
                guided_sampling=args.guided_sampling,
            )
            valid, round_diagnostics = validate_great_sample(
                candidate,
                model_train=model_train,
                output_train=output_train,
                categorical_columns=categorical_columns,
                codec=codec,
            )
            diagnostics['rounds'] += 1
            _merge_diagnostics(diagnostics, round_diagnostics)
            if len(valid):
                chunks.append(valid.head(remaining))
                accepted_count += min(len(valid), remaining)

        if accepted_count < sample_size:
            raise RuntimeError(
                f'Accepted only {accepted_count}/{sample_size} rows after '
                f'{args.max_sampling_rounds} sampling rounds.'
            )
        synthetic = pd.concat(chunks, ignore_index=True).head(sample_size)
        synthetic = synthetic.loc[:, output_train.columns]
        path = synthetic_dir / f'synthetic_{file_index}.csv'
        synthetic.to_csv(path, index=False)
        format_synthetic_file(output_train, path)
        diagnostics['written_rows'] = int(len(synthetic))
        diagnostics['elapsed_seconds'] = time.monotonic() - start_time
        diagnostics['acceptance_rate'] = (
            diagnostics['accepted_rows'] / diagnostics['candidate_rows']
            if diagnostics.get('candidate_rows') else 0.0
        )
        run_diagnostics['files'].append(diagnostics)
        paths.append(path)

    save_json(synthetic_dir / 'sampling_diagnostics.json', run_diagnostics)
    return paths, run_diagnostics


def run_training(args: argparse.Namespace, *, sample_after_training: bool) -> None:
    _validate_training_args(args)
    set_seed(args.seed)
    prepared = prepare_data(
        data_dir=args.data_dir,
        data_file=args.data_file,
        info_file=args.info_file,
        utility_feature_file=args.utility_feature_file,
        categorical_columns_override=args.categorical_columns,
        test_size=args.test_size,
        seed=args.seed,
        max_rows=args.max_rows,
        no_stratify=args.no_stratify,
    )
    metadata = write_training_artifacts(args.output_dir, prepared)
    model_train, _, output_train, _, codec = _prepare_model_frames(prepared)
    save_json(
        args.output_dir / 'serialization_config.json',
        {
            'float_precision': args.float_precision,
            'value_codec': codec.as_dict(),
        },
    )
    save_json(
        args.output_dir / 'train_config.json',
        args_payload(
            args,
            data_file=prepared.data_path,
            info_file=prepared.info_path,
            utility_feature_file=prepared.utility_feature_path,
            metadata=metadata,
        ),
    )

    from be_great import GReaT

    model = GReaT(
        llm=args.llm,
        experiment_dir=str(args.output_dir / 'trainer'),
        epochs=args.epochs,
        batch_size=args.batch_size,
        float_precision=args.float_precision,
        bf16=args.bf16,
        fp16=False,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        save_strategy='no',
        seed=args.seed,
        data_seed=args.seed,
    )
    token_stats = _token_length_stats(
        model, model_train, args.float_precision, args.seed,
    )
    save_json(args.output_dir / 'token_length_stats.json', token_stats)
    if args.command == 'train-sample' and args.max_length <= token_stats['p99']:
        raise ValueError(
            f'--max-length {args.max_length} is not above the measured p99 '
            f'training length {token_stats["p99"]}.'
        )

    started = time.monotonic()
    trainer = model.fit(
        model_train,
        conditional_col=prepared.target,
        random_conditional_col=False,
    )
    training_seconds = time.monotonic() - started
    history = [
        record for record in trainer.state.log_history
        if 'loss' in record or 'train_loss' in record
    ]
    if history:
        pd.DataFrame(history).to_csv(args.output_dir / 'train_loss.csv', index=False)
    checkpoint_info = _save_checkpoint(
        model, args.output_dir / 'great_model', args.checkpoint_dtype,
    )
    save_json(
        args.output_dir / 'training_summary.json',
        {
            'elapsed_seconds': training_seconds,
            'global_steps': int(trainer.state.global_step),
            'training_loss': trainer.state.log_history[-1].get('train_loss')
            if trainer.state.log_history else None,
            'checkpoint': checkpoint_info,
        },
    )
    print(args.output_dir / 'great_model')

    if sample_after_training:
        paths, diagnostics = _write_samples(
            args,
            model=model,
            model_train=model_train,
            output_train=output_train,
            categorical_columns=prepared.categorical_columns,
            codec=codec,
            output_dir=args.output_dir,
        )
        save_json(
            args.output_dir / 'synthetic' / 'sample_config.json',
            args_payload(args, synthetic_files=paths, diagnostics=diagnostics),
        )
        for path in paths:
            print(path)


def run_sampling(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir
    checkpoint_dir = args.checkpoint_dir or experiment_dir / 'great_model'
    train_path = args.train_file or experiment_dir / 'train.csv'
    metadata_path = args.metadata_file or experiment_dir / 'metadata.json'
    serialization_path = (
        args.serialization_config_file
        or experiment_dir / 'serialization_config.json'
    )
    for label, path in (
        ('checkpoint', checkpoint_dir),
        ('training split', train_path),
        ('metadata', metadata_path),
        ('serialization config', serialization_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f'{label} not found: {path}')

    output_train = pd.read_csv(train_path)
    metadata = load_metadata(metadata_path, list(output_train.columns))
    serialization = json.loads(serialization_path.read_text(encoding='utf-8'))
    codec = GreatValueCodec.from_dict(serialization['value_codec'])
    model_train = codec.transform(output_train)

    from be_great import GReaT

    model = GReaT.load_from_dir(str(checkpoint_dir))
    paths, diagnostics = _write_samples(
        args,
        model=model,
        model_train=model_train,
        output_train=output_train,
        categorical_columns=metadata['categorical_columns'],
        codec=codec,
        output_dir=experiment_dir,
    )
    save_json(
        experiment_dir / 'synthetic' / 'sample_config.json',
        args_payload(
            args,
            checkpoint_dir=checkpoint_dir,
            train_file=train_path,
            metadata_file=metadata_path,
            serialization_config_file=serialization_path,
            synthetic_files=paths,
            diagnostics=diagnostics,
        ),
    )
    for path in paths:
        print(path)
    del model
    gc.collect()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == 'sample':
        run_sampling(args)
    else:
        run_training(args, sample_after_training=args.command == 'train-sample')


if __name__ == '__main__':
    main()
