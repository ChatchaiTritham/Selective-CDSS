"""Repository-local constants for BASICS-CDSS."""

from typing import Final, Tuple

DEFAULT_CONFIDENCE_LEVEL: Final[float] = 0.95
DEFAULT_DPI: Final[int] = 300
DEFAULT_EPSILON: Final[float] = 1e-9
DEFAULT_EXPORT_IMAGE_FORMAT: Final[str] = "png"
DEFAULT_EXPORT_VECTOR_FORMAT: Final[str] = "svg"
DEFAULT_FIGURE_SIZE: Final[Tuple[float, float]] = (10.0, 6.0)
DEFAULT_RANDOM_SEED: Final[int] = 42
ISO_DATE_FORMAT: Final[str] = "%Y-%m-%d"
ISO_DATETIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"
ISO_TIMEZONE_UTC: Final[str] = "UTC"

PACKAGE_NAME: Final[str] = "basics_cdss"
PACKAGE_VERSION: Final[str] = "2.1.0"
FRAMEWORK_NAME: Final[str] = "BASICS-CDSS"

SUPPORTED_PYTHON_VERSION_MIN: Final[str] = "3.9"

FIGURE_EXPORT_FORMATS: Final[Tuple[str, str]] = (
    DEFAULT_EXPORT_IMAGE_FORMAT,
    DEFAULT_EXPORT_VECTOR_FORMAT,
)

__all__ = [
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_DPI",
    "DEFAULT_EPSILON",
    "DEFAULT_FIGURE_SIZE",
    "DEFAULT_RANDOM_SEED",
    "FIGURE_EXPORT_FORMATS",
    "FRAMEWORK_NAME",
    "ISO_DATE_FORMAT",
    "ISO_DATETIME_FORMAT",
    "ISO_TIMEZONE_UTC",
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "SUPPORTED_PYTHON_VERSION_MIN",
]
