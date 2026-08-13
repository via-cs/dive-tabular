"""Per-task model factories used by :mod:`evaluation.metrics.utility`.

Each task type maps to a small ensemble of estimators. The factories
return *fresh, unfitted* sklearn-compatible estimators so the same
factory can be reused across the TRTR / TSTR / TSRTR training regimes
without leaking state between fits.
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

_RANDOM_STATE = 42


def models_for_task(task):
    """Return ``{model_name: zero_arg_factory}`` for ``task``.

    Supported tasks:
    - ``'binary_classification'``: logistic regression, decision tree,
      XGBoost classifier.
    - ``'multiclass_classification'``: decision tree, XGBoost classifier.
    - ``'regression'``: linear regression, XGBoost regressor.
    """
    if task == 'binary_classification':
        return {
            'logistic_regression': lambda: LogisticRegression(
                max_iter=1000, random_state=_RANDOM_STATE,
            ),
            'decision_tree': lambda: DecisionTreeClassifier(
                random_state=_RANDOM_STATE,
            ),
            'xgboost': lambda: XGBClassifier(
                eval_metric='logloss',
                random_state=_RANDOM_STATE,
                tree_method='hist',
            ),
        }
    if task == 'multiclass_classification':
        return {
            'decision_tree': lambda: DecisionTreeClassifier(
                random_state=_RANDOM_STATE,
            ),
            'xgboost': lambda: XGBClassifier(
                eval_metric='mlogloss',
                random_state=_RANDOM_STATE,
                tree_method='hist',
            ),
        }
    if task == 'regression':
        return {
            'linear_regression': lambda: LinearRegression(),
            'xgboost': lambda: XGBRegressor(
                random_state=_RANDOM_STATE,
                tree_method='hist',
            ),
        }
    raise ValueError(f'Unknown task type: {task!r}.')
