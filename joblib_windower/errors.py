from __future__ import annotations


class InvalidDTypeError(ValueError):
    """Raised when the dtype is invalid."""


class InvalidMinFracError(ValueError):
    """Raised when the minimum fraction is invalid."""


class InvalidLengthError(ValueError):
    """Raised when the length is invalid."""


class InvalidStepError(ValueError):
    """Raised when the step is invalid."""


class InvalidWindowError(ValueError):
    """Raised when the window is invalid."""


class NoSlicersError(ValueError):
    """Raised when no slicers are valid."""
