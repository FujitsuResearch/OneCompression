"""Errors raised by the formal ARC profile.

Copyright 2025-2026 Fujitsu Ltd.
"""


class FormalArcError(Exception):
    """Base class for fail-closed formal ARC errors."""


class ArchiveCapacityExceeded(FormalArcError):
    """The declared fixed-width occurrence namespace is exhausted."""


class ArchiveConfigurationMismatch(FormalArcError):
    """An existing archive was opened with incompatible settings."""


class ArchiveIntegrityError(FormalArcError):
    """Persisted content or metadata failed an integrity check."""


class SourceOccurrenceConflict(FormalArcError):
    """One host event key was reused for different occurrence data."""


class UnknownObservation(FormalArcError):
    """No archived observation has the requested identifier."""


class AmbiguousObservation(FormalArcError):
    """A human-entered identifier prefix matches multiple records."""


class RecallRequiresChunking(FormalArcError):
    """A full recall cannot fit its declared response envelope."""


class ExactRecoveryError(FormalArcError):
    """A codec or chunk operation cannot guarantee exact recovery."""


class BudgetExceeded(FormalArcError):
    """A model invocation exceeds its declared context envelope."""


class InvalidAction(FormalArcError):
    """A model-facing ARC action does not match the strict grammar."""


class LegacyMigrationError(FormalArcError):
    """An experimental WAL cannot be migrated without losing guarantees."""
