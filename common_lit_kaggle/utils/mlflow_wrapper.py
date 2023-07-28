from unittest.mock import MagicMock

# pylint: disable=invalid-name
is_mlflow_available = False

_module_mlflow = None
try:
    import mlflow

    is_mlflow_available = True
    _module_mlflow = mlflow
except ImportError:
    pass


if not is_mlflow_available:
    _module_mlflow = MagicMock()

mlflow = _module_mlflow
