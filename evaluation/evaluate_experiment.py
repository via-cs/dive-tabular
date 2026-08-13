"""Evaluate synthetic data with quality, utility, and optional constraints."""

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __package__:
    from .metrics.quality import evaluate_quality, sdmetrics_metadata
else:
    from metrics.quality import evaluate_quality, sdmetrics_metadata


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = DEFAULT_ROOT / 'experiments' / 'heloc_tvae'
DEFAULT_SYNTHETIC_PATH = DEFAULT_EXPERIMENT_DIR / 'synthetic.csv'
CONSTRAINT_DETAILS_FILENAME = 'constraint_evaluation_details.json'
REAL_METRICS_CACHE_SCHEMA_VERSION = 1
REAL_METRICS = {'constraint', 'utility'}
CONSTRAINT_FILENAMES = (
    'categorical_dependency_constraint.json',
    'equational_constraint.json',
    'linear_constraint.json',
)
SYNTHETIC_UTILITY_REGIMES = ('tstr', 'tsrtr')


def read_csv(path):
    return pd.read_csv(
        path,
        keep_default_na=False,
        float_precision='round_trip',
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _real_metrics_fingerprint(
    experiment_dir, selected_metrics, constraints_expert
):
    cached_metrics = sorted(set(selected_metrics) & REAL_METRICS)
    files = {'train.csv': experiment_dir / 'train.csv'}
    if 'utility' in cached_metrics:
        files.update({
            'test.csv': experiment_dir / 'test.csv',
            'metadata.json': experiment_dir / 'metadata.json',
        })
    if 'constraint' in cached_metrics:
        constraint_dir = Path(constraints_expert)
        files.update({
            f'constraints/{name}': constraint_dir / name
            for name in CONSTRAINT_FILENAMES
            if (constraint_dir / name).is_file()
        })
    payload = {
        'schema_version': REAL_METRICS_CACHE_SCHEMA_VERSION,
        'metrics': cached_metrics,
        'files': {name: _sha256_file(path) for name, path in files.items()},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def _load_real_metrics_cache(path, fingerprint):
    if path is None or not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if (
        not isinstance(document, dict)
        or document.get('schema_version') != REAL_METRICS_CACHE_SCHEMA_VERSION
        or document.get('fingerprint') != fingerprint
        or not isinstance(document.get('real_metrics'), dict)
    ):
        return {}
    return document['real_metrics']


def _write_real_metrics_cache(path, fingerprint, real_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        'schema_version': REAL_METRICS_CACHE_SCHEMA_VERSION,
        'fingerprint': fingerprint,
        'real_metrics': real_metrics,
    }
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(
        json.dumps(document, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def log_progress(message):
    print(f'[evaluate_experiment] {message}', flush=True)


def resolve_utility_columns(metadata, available_columns):
    """Resolve the utility task entirely from experiment metadata."""
    if metadata is None:
        return {
            'target': None,
            'categorical_feature_columns': [],
            'numerical_feature_columns': [],
        }, [], []

    utility_target = metadata.get('target')
    utility_columns = metadata.get('feature_columns')
    if not isinstance(utility_target, str) or not utility_target:
        raise ValueError('metadata.json must contain a non-empty string "target".')
    if not isinstance(utility_columns, list) or not utility_columns:
        raise ValueError('metadata.json must contain a non-empty "feature_columns" list.')
    if utility_target in utility_columns:
        raise ValueError(
            f'Target column {utility_target!r} must not appear in feature_columns.'
        )
    if len(utility_columns) != len(set(utility_columns)):
        raise ValueError('metadata.json feature_columns contains duplicates.')
    required = utility_columns + [utility_target]
    missing = [column for column in required if column not in available_columns]
    if missing:
        raise ValueError(f'Utility columns are missing from experiment data: {missing}')

    all_categorical = list(metadata.get('categorical_columns', []))
    all_numerical = list(metadata.get('numerical_columns', []))
    utility_categorical = [
        column for column in utility_columns if column in all_categorical
    ]
    utility_numerical = [
        column for column in utility_columns if column in all_numerical
    ]
    untyped = [
        column
        for column in utility_columns
        if column not in utility_categorical and column not in utility_numerical
    ]
    if untyped:
        raise ValueError(
            'Utility feature columns are not classified as categorical or '
            f'numerical in metadata.json: {untyped}'
        )

    output = {
        'target': utility_target,
        'categorical_feature_columns': utility_categorical,
        'numerical_feature_columns': utility_numerical,
    }
    return output, utility_categorical, utility_numerical


def _safe_call(label, fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return fn()
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}', 'metric': label}


def _synthetic_sort_key(path):
    stem = path.stem
    prefix, _, suffix = stem.rpartition('_')
    if prefix == 'synthetic' and suffix.isdigit():
        return (0, int(suffix), path.name)
    return (1, path.name)


def synthetic_files(experiment_dir, synthetic_path=None):
    if synthetic_path is not None:
        synthetic_path = Path(synthetic_path)
        if synthetic_path.is_file():
            return [synthetic_path]
        if not synthetic_path.is_dir():
            raise FileNotFoundError(f'Synthetic data path not found: {synthetic_path}.')
        synthetic_dir = synthetic_path
    else:
        synthetic_dir = Path(experiment_dir) / 'synthetic'

    files = sorted(synthetic_dir.glob('*.csv'), key=_synthetic_sort_key)
    if not files:
        raise FileNotFoundError(f'No synthetic CSV files found in {synthetic_dir}.')
    return files


def _merge_values(values):
    first = values[0]

    if isinstance(first, dict) and all(isinstance(value, dict) for value in values):
        keys = first.keys()
        return {
            child_key: _merge_values(
                [value[child_key] for value in values],
            )
            for child_key in keys
            if all(child_key in value for value in values)
        }

    if isinstance(first, list) and all(isinstance(value, list) for value in values):
        if not all(len(value) == len(first) for value in values):
            return values
        return [
            _merge_values([value[index] for value in values])
            for index in range(len(first))
        ]

    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return values

    return first


def _merge_utility(trtr_result, synthetic_results):
    if not isinstance(trtr_result, dict) or 'models' not in trtr_result:
        return trtr_result

    synthetic_utility = _merge_values(synthetic_results)
    synthetic_models = (
        synthetic_utility.get('models', {})
        if isinstance(synthetic_utility, dict)
        else {}
    )
    result = {
        'task': trtr_result['task'],
        'models': {},
    }
    for model_name, trtr_model in trtr_result['models'].items():
        result['models'][model_name] = {
            **trtr_model,
            **synthetic_models.get(model_name, {}),
        }
    if 'classes' in trtr_result:
        result['classes'] = trtr_result['classes']
    if isinstance(synthetic_utility, dict) and 'error' in synthetic_utility:
        result['synthetic_error'] = synthetic_utility
    return result


def evaluate_experiment(
    experiment_dir,
    metrics=None,
    synthetic_path=None,
    constraints_expert=None,
    constraint_details_path=None,
    real_metrics_cache_path=None,
    utility_regimes=None,
):
    """Run requested metrics on a trained experiment.

    Constraint metrics are included only when ``constraints_expert`` is
    provided. No constraint-directory discovery is performed.
    """
    all_metrics = {'constraint', 'quality', 'utility'}
    if metrics is None:
        selected_metrics = set(all_metrics)
    else:
        selected_metrics = {str(metric).lower() for metric in metrics}
        unknown_metrics = selected_metrics - all_metrics
        if unknown_metrics:
            raise ValueError(
                f'Unknown metric(s): {sorted(unknown_metrics)}. '
                f'Expected one or more of {sorted(all_metrics)}.'
            )
        if not selected_metrics:
            selected_metrics = set(all_metrics)

    if constraints_expert is None and 'constraint' in selected_metrics:
        log_progress(
            'Skipping constraint metrics because --constraints-expert was not provided.'
        )
        selected_metrics.remove('constraint')

    if utility_regimes is None:
        selected_utility_regimes = SYNTHETIC_UTILITY_REGIMES
    else:
        selected_utility_regimes = tuple(
            dict.fromkeys(str(regime).lower() for regime in utility_regimes)
        )
        unknown_regimes = (
            set(selected_utility_regimes) - set(SYNTHETIC_UTILITY_REGIMES)
        )
        if unknown_regimes:
            raise ValueError(
                f'Unknown utility regime(s): {sorted(unknown_regimes)}. '
                f'Expected one or more of {list(SYNTHETIC_UTILITY_REGIMES)}.'
            )
        if not selected_utility_regimes:
            raise ValueError('At least one utility regime is required.')

    experiment_dir = Path(experiment_dir)
    synthetic_search_path = (
        Path(synthetic_path) if synthetic_path is not None else experiment_dir / 'synthetic'
    )
    log_progress(f'Looking for synthetic CSV files in {synthetic_search_path}')
    synthetic_paths = synthetic_files(experiment_dir, synthetic_path=synthetic_path)
    log_progress(f'Found {len(synthetic_paths)} synthetic CSV file(s).')

    log_progress('Loading metadata.')
    metadata = json.loads((experiment_dir / 'metadata.json').read_text(encoding='utf-8'))

    log_progress('Loading train/test splits.')
    real_train = read_csv(experiment_dir / 'train.csv')
    real_test = None
    metric_metadata = None
    if 'utility' in selected_metrics:
        real_test = read_csv(experiment_dir / 'test.csv')
    if 'quality' in selected_metrics:
        if metadata is None:
            raise FileNotFoundError(
                f'metadata.json is required for quality metrics in {experiment_dir}.'
            )
        metric_metadata = sdmetrics_metadata(
            real_train.columns,
            metadata['numerical_columns'],
            metadata.get('categorical_columns', []),
        )

    utility_columns = None
    utility_categorical = []
    utility_numerical = []
    utility_target = None
    if 'utility' in selected_metrics:
        utility_columns, utility_categorical, utility_numerical = (
            resolve_utility_columns(metadata, real_train.columns)
        )
        utility_target = utility_columns['target']
    cache_path = (
        Path(real_metrics_cache_path)
        if real_metrics_cache_path is not None
        else None
    )
    cache_fingerprint = None
    real_metrics_cache = {}
    if cache_path is not None and selected_metrics & REAL_METRICS:
        cache_fingerprint = _real_metrics_fingerprint(
            experiment_dir, selected_metrics, constraints_expert
        )
        real_metrics_cache = _load_real_metrics_cache(
            cache_path, cache_fingerprint
        )
        if real_metrics_cache:
            log_progress(f'Reusing real metrics from {cache_path}.')

    paths = {
        'experiment_dir': str(experiment_dir),
        'synthetic_files': [str(path) for path in synthetic_paths],
        'metrics': sorted(selected_metrics),
        'constraints_expert': (
            str(constraints_expert) if constraints_expert is not None else None
        ),
    }
    if 'utility' in selected_metrics:
        paths['utility_regimes'] = ['trtr', *selected_utility_regimes]
    trtr_utility = real_metrics_cache.get('utility')
    if 'utility' in selected_metrics and trtr_utility is None:
        if __package__:
            from .metrics import evaluate_trtr
        else:
            from metrics import evaluate_trtr

        log_progress('Evaluating TRTR utility once.')
        trtr_utility = _safe_call(
            'evaluate_trtr',
            lambda: evaluate_trtr(
                real_train=real_train.copy(),
                real_test=real_test.copy(),
                target=utility_target,
                categorical_columns=utility_categorical,
                numerical_columns=utility_numerical,
            ),
        )
        real_metrics_cache['utility'] = trtr_utility
        if cache_path is not None:
            _write_real_metrics_cache(
                cache_path, cache_fingerprint, real_metrics_cache
            )

    cached_constraint = real_metrics_cache.get('constraint')
    if cached_constraint is not None:
        cached_summary = cached_constraint.get('summary', {})
        linear_status = cached_summary.get('family_status', {}).get('linear')
        if (
            linear_status == 'evaluated'
            and 'linear_feasibility_distance' not in cached_summary
        ):
            # Constraint caches written before LFD was introduced are missing a
            # required real-data reference metric. Recompute only constraints;
            # cached utility metrics remain reusable.
            cached_constraint = None
        else:
            cached_summary.setdefault('linear_feasibility_distance', None)
    if 'constraint' in selected_metrics and cached_constraint is None:
        if __package__:
            from .metrics.constraint import evaluate_real_constraint_dataset
        else:
            from metrics.constraint import evaluate_real_constraint_dataset

        log_progress('Evaluating real-data constraint metrics once.')
        real_summary, real_details = evaluate_real_constraint_dataset(
            real_train=real_train.copy(),
            real_train_path=experiment_dir / 'train.csv',
            constraints_expert=constraints_expert,
        )
        cached_constraint = {
            'summary': {
                key: value
                for key, value in real_summary.items()
                if key != 'path'
            },
            'details': {
                key: value
                for key, value in real_details.items()
                if key != 'source'
            },
        }
        real_metrics_cache['constraint'] = cached_constraint
        if cache_path is not None:
            _write_real_metrics_cache(
                cache_path, cache_fingerprint, real_metrics_cache
            )

    synthetic_results = []
    constraint_datasets = []
    for index, path in enumerate(synthetic_paths, start=1):
        log_progress(f'[{index}/{len(synthetic_paths)}] Loading {path.name}.')
        synthetic = read_csv(path)
        result = {'path': str(path)}
        if 'quality' in selected_metrics:
            log_progress(f'[{index}/{len(synthetic_paths)}] Evaluating quality metrics.')
            result['quality'] = _safe_call(
                'QualityReport',
                lambda synthetic=synthetic: evaluate_quality(
                    real_train.copy(), synthetic.copy(), metric_metadata
                ),
            )
        if 'utility' in selected_metrics:
            if __package__:
                from .metrics import evaluate_synthetic_utility
            else:
                from metrics import evaluate_synthetic_utility

            log_progress(f'[{index}/{len(synthetic_paths)}] Evaluating synthetic utility.')
            result['utility'] = _safe_call(
                'evaluate_synthetic_utility',
                lambda synthetic=synthetic: evaluate_synthetic_utility(
                    real_train=real_train.copy(),
                    real_test=real_test.copy(),
                    synthetic=synthetic.copy(),
                    target=utility_target,
                    categorical_columns=utility_categorical,
                    numerical_columns=utility_numerical,
                    regimes=selected_utility_regimes,
                ),
            )
        if 'constraint' in selected_metrics:
            constraint_datasets.append((path, synthetic.copy()))
        synthetic_results.append(result)
        log_progress(f'[{index}/{len(synthetic_paths)}] Finished {path.name}.')

    log_progress('Merging per-file evaluation results.')
    results = {
        'paths': paths,
        'per_file': synthetic_results,
    }
    if utility_columns is not None:
        results['utility_columns'] = utility_columns
    if 'quality' in selected_metrics:
        results['quality'] = _merge_values([
            result['quality']
            for result in synthetic_results
        ])
    if 'utility' in selected_metrics:
        results['utility'] = _merge_utility(
            trtr_utility,
            [result['utility'] for result in synthetic_results],
        )
    if 'constraint' in selected_metrics:
        if __package__:
            from .metrics.constraint import evaluate_constraint_datasets
        else:
            from metrics.constraint import evaluate_constraint_datasets

        log_progress('Evaluating expert constraint metrics.')
        real_path = str(experiment_dir / 'train.csv')
        constraint_summary, constraint_details = evaluate_constraint_datasets(
            real_train=real_train.copy(),
            real_train_path=real_path,
            synthetic_datasets=constraint_datasets,
            constraints_expert=constraints_expert,
            real_summary={'path': real_path, **cached_constraint['summary']},
            real_details={'source': real_path, **cached_constraint['details']},
        )
        details_output = (
            Path(constraint_details_path)
            if constraint_details_path is not None
            else experiment_dir / CONSTRAINT_DETAILS_FILENAME
        )
        details_output.parent.mkdir(parents=True, exist_ok=True)
        details_output.write_text(
            json.dumps(constraint_details, indent=2, allow_nan=False) + '\n',
            encoding='utf-8',
        )
        constraint_summary['details_path'] = str(details_output)
        results['constraint'] = constraint_summary
        constraint_by_path = {
            item['path']: item
            for item in constraint_summary['synthetic']
        }
        for result in synthetic_results:
            constraint_result = constraint_by_path.get(result['path'])
            if constraint_result is not None:
                result['constraint'] = constraint_result
        log_progress(f'Wrote constraint details to {details_output}.')
    log_progress('Evaluation complete.')
    return results


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_dir', type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help=(
            'Optional path for the evaluation JSON dump. Overrides the '
            'default filename inside <experiment_dir>.'
        ),
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=['quality', 'utility', 'constraint'],
        default=None,
        help=(
            'Metric groups to evaluate. Defaults to quality and utility, plus '
            'constraint when --constraints-expert is provided.'
        ),
    )
    parser.add_argument(
        '--synthetic-data',
        type=Path,
        default=None,
        help=(
            'Override synthetic data path. May be a CSV file or a directory '
            'containing synthetic CSV files. Defaults to <experiment_dir>/synthetic.'
        ),
    )
    parser.add_argument(
        '--utility-regimes',
        nargs='+',
        choices=list(SYNTHETIC_UTILITY_REGIMES),
        default=None,
        help=(
            'Synthetic utility training regimes to evaluate. Defaults to '
            'both tstr and tsrtr; TRTR is always included as the baseline.'
        ),
    )
    parser.add_argument(
        '--constraints-expert',
        type=Path,
        default=None,
        help=(
            'Optional constraints_expert directory. Constraint metrics are '
            'skipped when this argument is omitted.'
        ),
    )
    parser.add_argument(
        '--real-metrics-cache',
        type=Path,
        default=None,
        help=(
            'Optional shared cache for TRTR and real-data constraint metrics. '
            'The cache is content-fingerprinted and reused when current.'
        ),
    )
    parser.add_argument(
        '--constraint-details-output',
        type=Path,
        default=None,
        help=(
            'Constraint detail JSON override. Defaults to '
            f'<experiment_dir>/{CONSTRAINT_DETAILS_FILENAME}.'
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    results = evaluate_experiment(
        args.experiment_dir,
        metrics=args.metrics,
        synthetic_path=args.synthetic_data,
        constraints_expert=args.constraints_expert,
        constraint_details_path=args.constraint_details_output,
        real_metrics_cache_path=args.real_metrics_cache,
        utility_regimes=args.utility_regimes,
    )
    if args.output is not None:
        output_path = args.output
    else:
        output_path = args.experiment_dir / f'evaluation.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
