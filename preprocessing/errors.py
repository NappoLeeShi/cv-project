"""Shared exceptions for dataset loaders."""


class DatasetError(RuntimeError):
    """Raised when a dataset layout is invalid or a sample cannot be resolved."""