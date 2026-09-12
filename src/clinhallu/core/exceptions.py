"""ClinHallu-specific exceptions."""


class ClinHalluError(RuntimeError):
    """Base error for failures that should be shown directly to CLI users."""


class ProtocolViolation(ClinHalluError):
    """Raised when data use would violate the fixed-split protocol."""
