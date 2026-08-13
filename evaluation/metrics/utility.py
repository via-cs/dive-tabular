"""ML-utility metrics for synthetic tabular data.

Up to three training regimes are evaluated, all scored on the *real* held-out
test set:

- ``trtr``  -- train on real training data only.
- ``tstr``  -- train on synthetic data only.
- ``tsrtr`` -- train on synthetic concatenated with the real training
  data.

Each regime is run with a small ensemble of estimators selected by task
type (see :func:`evaluation.metrics.utils.models.models_for_task`).
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .utils.models import models_for_task

_MULTICLASS_STRING_LIMIT = 20
SYNTHETIC_UTILITY_REGIMES = ('tstr', 'tsrtr')


def infer_task_type(train_data, target):
    """Infer the ML task type from the target column's distinct values.

    Returns one of ``'binary_classification'``,
    ``'multiclass_classification'``, or ``'regression'``.
    """
    series = train_data[target].dropna()
    if series.empty:
        return 'regression'
    n_unique = series.nunique()
    is_string = pd.api.types.is_string_dtype(series)
    if n_unique == 2:
        return 'binary_classification'
    if is_string or n_unique <= _MULTICLASS_STRING_LIMIT:
        return 'multiclass_classification'
    return 'regression'


def _coerce_numerical(df, numerical_columns):
    """Cast numerical feature columns to ``float``, filling NaNs with 0.

    The evaluation CSVs are read with ``keep_default_na=False`` so empty
    cells survive as the literal empty string. We coerce them here so
    sklearn doesn't choke on object-typed columns.
    """
    df = df.copy()
    for column in numerical_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0.0)
    return df


def _fit_feature_encoder(train_frame, categorical, numerical):
    """Fit a ``ColumnTransformer`` on the given training frame only.

    Preprocessing is treated as part of the model pipeline: the scaler's
    mean/std and the one-hot vocabulary are learned from whatever set of
    rows is used to train the downstream estimator, and never from the
    held-out test frame. Unseen categorical values at transform time are
    encoded as all-zero one-hot rows via ``handle_unknown='ignore'``.
    """
    transformer = ColumnTransformer([
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            categorical,
        ),
        ('num', StandardScaler(), numerical),
    ])
    transformer.fit(train_frame)
    return transformer


def _classification_metrics(y_true, y_pred, y_proba, task):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(
            f1_score(
                y_true,
                y_pred,
                average='binary' if task == 'binary_classification' else 'macro',
                zero_division=0,
            )
        ),
    }
    if task == 'binary_classification' and y_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            # Happens e.g. when y_true has a single class in the test set.
            metrics['roc_auc'] = None
    return metrics


def _regression_metrics(y_true, y_pred):
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _safe(label, fn):
    """Run ``fn`` and return its result, or an error dict on exception."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return fn()
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}', 'metric': label}


def _encode_targets(y_real_train, y_real_test, y_synthetic, task):
    """Encode targets consistently across train / test / synthetic.

    For classification, fit a :class:`LabelEncoder` on the union so that
    no class drops out of any of the three sets (XGBoost in particular
    requires contiguous 0-indexed integer labels).

    For regression, coerce to float and fill NaNs with 0.
    """
    if task in ('binary_classification', 'multiclass_classification'):
        all_targets = pd.concat(
            [y_real_train, y_real_test, y_synthetic],
        ).astype(str)
        encoder = LabelEncoder()
        encoder.fit(all_targets)
        return (
            encoder.transform(y_real_train.astype(str)),
            encoder.transform(y_real_test.astype(str)),
            encoder.transform(y_synthetic.astype(str)),
            encoder.classes_.tolist(),
        )
    return (
        pd.to_numeric(y_real_train, errors='coerce').fillna(0.0).to_numpy(),
        pd.to_numeric(y_real_test, errors='coerce').fillna(0.0).to_numpy(),
        pd.to_numeric(y_synthetic, errors='coerce').fillna(0.0).to_numpy(),
        None,
    )


def _encode_real_targets(y_real_train, y_real_test, task):
    """Encode targets for metrics that only depend on real data."""
    if task in ('binary_classification', 'multiclass_classification'):
        all_targets = pd.concat([y_real_train, y_real_test]).astype(str)
        encoder = LabelEncoder()
        encoder.fit(all_targets)
        return (
            encoder.transform(y_real_train.astype(str)),
            encoder.transform(y_real_test.astype(str)),
            encoder.classes_.tolist(),
        )
    return (
        pd.to_numeric(y_real_train, errors='coerce').fillna(0.0).to_numpy(),
        pd.to_numeric(y_real_test, errors='coerce').fillna(0.0).to_numpy(),
        None,
    )


def _evaluate_model(model_factory, X_train, y_train, X_test, y_test, task):
    model = model_factory()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if task == 'regression':
        return _regression_metrics(y_test, y_pred)
    y_proba = None
    if task == 'binary_classification' and hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    return _classification_metrics(y_test, y_pred, y_proba, task)


def evaluate_trtr(
    real_train,
    real_test,
    target,
    categorical_columns,
    numerical_columns,
):
    """Evaluate utility for the real-train / real-test regime once."""
    task = infer_task_type(real_train, target)

    feature_categorical = [c for c in categorical_columns if c != target]
    feature_numerical = [c for c in numerical_columns if c != target]

    real_train = _coerce_numerical(real_train, feature_numerical)
    real_test = _coerce_numerical(real_test, feature_numerical)

    feature_columns = feature_categorical + feature_numerical
    X_real_train_raw = real_train[feature_columns]
    X_real_test_raw = real_test[feature_columns]

    encoder = _fit_feature_encoder(
        X_real_train_raw,
        categorical=feature_categorical,
        numerical=feature_numerical,
    )

    X_real_train = encoder.transform(X_real_train_raw)
    X_real_test = encoder.transform(X_real_test_raw)
    y_real_train, y_real_test, classes = _encode_real_targets(
        real_train[target], real_test[target], task,
    )

    per_model = {}
    for model_name, factory in models_for_task(task).items():
        per_model[model_name] = {
            'trtr': _safe(
                f'{model_name}/trtr',
                lambda factory=factory: _evaluate_model(
                    factory, X_real_train, y_real_train, X_real_test, y_real_test, task,
                ),
            )
        }

    result = {'task': task, 'models': per_model}
    if classes is not None:
        result['classes'] = classes
    return result


def evaluate_synthetic_utility(
    real_train,
    real_test,
    synthetic,
    target,
    categorical_columns,
    numerical_columns,
    regimes=None,
):
    """Evaluate selected utility regimes that depend on synthetic data."""
    if regimes is None:
        selected_regimes = SYNTHETIC_UTILITY_REGIMES
    else:
        selected_regimes = tuple(
            dict.fromkeys(str(regime).lower() for regime in regimes)
        )
        unknown_regimes = set(selected_regimes) - set(SYNTHETIC_UTILITY_REGIMES)
        if unknown_regimes:
            raise ValueError(
                f'Unknown synthetic utility regime(s): {sorted(unknown_regimes)}. '
                f'Expected one or more of {list(SYNTHETIC_UTILITY_REGIMES)}.'
            )
        if not selected_regimes:
            raise ValueError('At least one synthetic utility regime is required.')

    task = infer_task_type(real_train, target)

    feature_categorical = [c for c in categorical_columns if c != target]
    feature_numerical = [c for c in numerical_columns if c != target]

    real_train = _coerce_numerical(real_train, feature_numerical)
    real_test = _coerce_numerical(real_test, feature_numerical)
    synthetic = _coerce_numerical(synthetic, feature_numerical)

    feature_columns = feature_categorical + feature_numerical
    X_real_train_raw = real_train[feature_columns]
    X_real_test_raw = real_test[feature_columns]
    X_synthetic_raw = synthetic[feature_columns]

    y_real_train, y_real_test, y_synthetic, _classes = _encode_targets(
        real_train[target], real_test[target], synthetic[target], task,
    )

    # Each regime gets its own encoder, fit only on that regime's training
    # rows. real_test is transformed (never fit) by every selected regime.
    regime_specs = {}
    if 'tstr' in selected_regimes:
        regime_specs['tstr'] = (X_synthetic_raw, y_synthetic)
    if 'tsrtr' in selected_regimes:
        regime_specs['tsrtr'] = (
            pd.concat([X_synthetic_raw, X_real_train_raw], ignore_index=True),
            np.concatenate([y_synthetic, y_real_train]),
        )

    train_sets = {}
    for regime, (train_raw, y_train) in regime_specs.items():
        regime_encoder = _fit_feature_encoder(
            train_raw,
            categorical=feature_categorical,
            numerical=feature_numerical,
        )
        train_sets[regime] = (
            regime_encoder.transform(train_raw),
            y_train,
            regime_encoder.transform(X_real_test_raw),
        )

    per_model = {}
    for model_name, factory in models_for_task(task).items():
        per_model[model_name] = {
            regime: _safe(
                f'{model_name}/{regime}',
                lambda X=X, y=y, X_test=X_test, factory=factory: _evaluate_model(
                    factory, X, y, X_test, y_real_test, task,
                ),
            )
            for regime, (X, y, X_test) in train_sets.items()
        }

    return {'models': per_model}


def _combine_utility_results(trtr_result, synthetic_result):
    result = {
        'task': trtr_result['task'],
        'models': {},
    }
    for model_name, model_result in trtr_result['models'].items():
        result['models'][model_name] = {
            **model_result,
            **synthetic_result['models'].get(model_name, {}),
        }
    if 'classes' in trtr_result:
        result['classes'] = trtr_result['classes']
    return result


def evaluate_utility(
    real_train,
    real_test,
    synthetic,
    target,
    categorical_columns,
    numerical_columns,
    synthetic_regimes=None,
):
    """Evaluate ML utility under the TRTR / TSTR / TSRTR regimes.

    Parameters
    ----------
    real_train, real_test, synthetic : pandas.DataFrame
        DataFrames sharing the same column schema.
    target : str
        Name of the target column.
    categorical_columns, numerical_columns : sequence of str
        Column-type partitioning of all columns. The target is removed
        from the feature lists automatically.

    Returns
    -------
    dict
        ``{'task': <task_type>, 'models': {model_name:
        {regime: metrics_or_error}}, 'classes': [...]?}``. The
        ``classes`` field is only present for classification tasks and
        records the LabelEncoder class order (so e.g. ``f1`` for binary
        classification is computed on the *second* class in this list).
    """
    trtr_result = evaluate_trtr(
        real_train=real_train,
        real_test=real_test,
        target=target,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )
    synthetic_result = evaluate_synthetic_utility(
        real_train=real_train,
        real_test=real_test,
        synthetic=synthetic,
        target=target,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        regimes=synthetic_regimes,
    )
    return _combine_utility_results(trtr_result, synthetic_result)
